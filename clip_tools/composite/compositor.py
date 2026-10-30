"""Compositor engine — port of ``psd_tools.composite.composite``.

Produces ``(color, shape, alpha)`` float32 triples using the full PDF 1.7
§11.4 group compositing model (shape vs alpha distinction, pass-through
vs isolated groups, knockout, Porter-Duff over, backdrop-aware color).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from clip_tools.composite import utils as cu
from clip_tools.composite.blend import BLEND_FUNC, normal
from clip_tools.constants import LayerComposite, LayerFolderBit

try:
    from clip_tools.composite._composite_fast import (
        apply_source_normal as _apply_source_normal_fast,
    )
except ImportError:  # pragma: no cover
    # The Cython extension is optional at import time — ``_apply_source``
    # falls back to the pure-numpy path when the compiled module is
    # unavailable (fresh checkout without a build, PyPy, non-x86 arch
    # without wheels, etc.).
    _apply_source_normal_fast = None

if TYPE_CHECKING:
    from clip_tools.api.clip_layer import ClipLayer

logger = logging.getLogger(__name__)

Viewport = Tuple[int, int, int, int]  # (left, top, right, bottom)
LayerFilter = Callable[["ClipLayer"], bool]


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def composite(
    group,  # ClipLayer
    color: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
    viewport: Optional[Viewport] = None,
    layer_filter: Optional[LayerFilter] = None,
    force: bool = False,
    as_layer: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite ``group`` and return ``(color, shape, alpha)`` as float32.

    - ``viewport`` defaults to the full canvas of ``group``.
    - ``color`` / ``alpha`` are the initial backdrop (defaults to black / zero).
    - ``as_layer=True`` composites ``group`` as a single layer source
      (used when the caller asked for one specific group's output rather
      than its parent's output).
    """
    if viewport is None:
        h, w = group._canvas_size  # (H, W)
        viewport = (0, 0, int(w), int(h))

    vw = viewport[2] - viewport[0]
    vh = viewport[3] - viewport[1]
    if color is None:
        color = np.zeros((vh, vw, 3), dtype=np.float32)
    if alpha is None:
        alpha = np.zeros((vh, vw, 1), dtype=np.float32)

    # A top-level composite always isolates (there is no outer backdrop to
    # pass through); nested calls set this explicitly based on blend mode.
    isolated = True
    if _is_group(group):
        bm = _blend_mode(group)
        if bm != LayerComposite.PASS_THROUGH:
            isolated = True
        else:
            isolated = False

    compositor = Compositor(
        viewport=viewport,
        color=color,
        alpha=alpha,
        isolated=isolated,
        layer_filter=layer_filter,
        force=force,
    )
    target = group._children if (_is_group(group) and not as_layer) else [group]
    for layer in target:
        compositor.apply(layer)
    return compositor.finish()


def composite_pil(
    group,
    viewport: Optional[Viewport] = None,
    layer_filter: Optional[LayerFilter] = None,
    force: bool = False,
    as_layer: bool = False,
) -> Image.Image:
    """``composite()`` + float32→uint8 PIL conversion. Returns RGBA."""
    color, _shape, alpha = composite(
        group,
        viewport=viewport,
        layer_filter=layer_filter,
        force=force,
        as_layer=as_layer,
    )
    rgba = np.concatenate([color, alpha], axis=2)
    rgba = np.clip(rgba * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------


class Compositor:
    """Accumulates `(color, shape, alpha)` across a group's children."""

    def __init__(
        self,
        viewport: Viewport,
        color: np.ndarray,
        alpha: np.ndarray,
        isolated: bool,
        layer_filter: Optional[LayerFilter] = None,
        force: bool = False,
    ) -> None:
        self._viewport = viewport
        self._layer_filter = layer_filter
        self._force = force

        vw = viewport[2] - viewport[0]
        vh = viewport[3] - viewport[1]

        self._color_0 = np.zeros_like(color) if isolated else color
        self._alpha_0 = np.zeros_like(alpha) if isolated else alpha

        # Group's accumulated geometric and effective coverage.
        self._shape_g = np.zeros((vh, vw, 1), dtype=np.float32)
        self._alpha_g = np.zeros((vh, vw, 1), dtype=np.float32)

        # Running composite starts at backdrop.
        self._color = self._color_0.copy()
        self._alpha = self._alpha_0.copy()

    # ---- Public: apply one child ---------------------------------------
    def apply(self, layer) -> None:
        if self._layer_filter is not None and not self._layer_filter(layer):
            return
        if not getattr(layer, "visible", True) and not self._force:
            return

        knockout = False  # clip-tools has no knockout tag yet; always False.

        if _is_group(layer):
            color, shape, alpha = self._get_group(layer, knockout)
        else:
            color, shape, alpha = self._get_object(layer)

        if color is None:
            return

        shape_mask, opacity_mask = self._get_mask(layer)
        shape_const, opacity_const = self._get_const(layer)
        shape = shape * shape_mask
        alpha = alpha * shape_mask * opacity_mask * opacity_const

        self._apply_source(
            color,
            shape * shape_const,
            alpha * shape_const,
            _blend_mode(layer),
            knockout,
        )

    # ---- Source extraction --------------------------------------------
    def _get_object(
        self, layer
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Produce (color, shape, alpha) from a leaf layer."""
        raster = _leaf_raster(layer)
        if raster is None:
            # Fall back to solid-color fill layers (CSP's "用紙" / paper layer
            # exposed via draw_color_enable / draw_color_main_* fields).
            solid = _solid_fill_rgba(layer)
            if solid is None:
                return None, None, None
            rgba = _crop_to_viewport(solid, self._viewport)
        else:
            rgba = _place_raster_on_canvas(raster, layer._canvas_size, layer.bbox[:2])
            rgba = _crop_to_viewport(rgba, self._viewport)
        color = rgba[..., :3].astype(np.float32, copy=False)
        alpha = rgba[..., 3:4].astype(np.float32, copy=False)
        shape = alpha.copy()  # Leaves: shape == alpha before opacity.
        return color, shape, alpha

    def _get_group(
        self, group, knockout: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recursively composite a child group within this parent's viewport."""
        if knockout:
            backdrop_color = self._color_0
            backdrop_alpha = self._alpha_0
        else:
            backdrop_color = self._color
            backdrop_alpha = self._alpha

        return composite(
            group,
            color=backdrop_color.copy(),
            alpha=backdrop_alpha.copy(),
            viewport=self._viewport,
            layer_filter=self._layer_filter,
            force=self._force,
            as_layer=False,
        )

    def _get_mask(self, layer) -> Tuple[np.ndarray, np.ndarray]:
        """(shape_mask, opacity_mask). clip-tools has no raster mask yet, so ones."""
        vw = self._viewport[2] - self._viewport[0]
        vh = self._viewport[3] - self._viewport[1]
        ones = np.ones((vh, vw, 1), dtype=np.float32)
        return ones, ones

    def _get_const(self, layer) -> Tuple[float, float]:
        """(shape_const=fill_opacity, opacity_const=layer_opacity).

        clip-tools has no fill_opacity concept; return (1.0, layer_opacity).

        We bypass ``ClipLayer.opacity`` (which divides by 256, matching
        CSP's internal 0-256 integer scale) and use /255 effective scale
        here. CSP stores LayerOpacity as 0-256 but when it exports to PSD
        the same byte value shifts semantics to /255. For byte-exact
        parity with psd-tools output we need the same /255 interpretation
        at composite time; the public ``layer.opacity`` property keeps
        /256 for CSP-format fidelity.
        """
        raw = getattr(layer.metadata, "layer_opacity", 256)
        try:
            raw_i = int(raw) if raw is not None else 256
        except (TypeError, ValueError):
            raw_i = 256
        opacity = min(raw_i, 255) / 255.0
        return 1.0, opacity

    # ---- Porter-Duff + blend merge ------------------------------------
    def _apply_source(
        self,
        color: np.ndarray,
        shape: np.ndarray,
        alpha: np.ndarray,
        blend_mode: LayerComposite,
        knockout: bool,
    ) -> None:
        # Fast path: NORMAL + non-knockout handled by a single Cython kernel
        # that fuses shape/alpha/color accumulation into one pass over the
        # viewport, avoiding ~15 numpy temporary arrays per call. Falls
        # through to the full numpy expression below when the compiled
        # module isn't available, when a non-NORMAL blend mode is in
        # play, or when the group is in knockout.
        if (
            _apply_source_normal_fast is not None
            and blend_mode == LayerComposite.NORMAL
            and not knockout
        ):
            _apply_source_normal_fast(
                self._color,
                self._shape_g,
                self._alpha_g,
                self._alpha,
                self._alpha_0,
                color,
                shape,
                alpha,
            )
            return

        self._shape_g = cu.union(self._shape_g, shape)
        if knockout:
            self._alpha_g = (
                (1.0 - shape) * self._alpha_g + (shape - alpha) * self._alpha_0 + alpha
            )
        else:
            self._alpha_g = cu.union(self._alpha_g, alpha)

        alpha_previous = self._alpha
        self._alpha = cu.union(self._alpha_0, self._alpha_g)

        alpha_b = self._alpha_0 if knockout else alpha_previous
        color_b = self._color_0 if knockout else self._color

        blend_fn = BLEND_FUNC.get(blend_mode, normal)
        color_t = (shape - alpha) * alpha_b * color_b + alpha * (
            (1.0 - alpha_b) * color + alpha_b * blend_fn(color_b, color)
        )
        self._color = cu.clip(
            cu.divide(
                (1.0 - shape) * alpha_previous * self._color + color_t,
                self._alpha,
            )
        )

    # ---- Finish -------------------------------------------------------
    def finish(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.color, self.shape, self.alpha

    @property
    def color(self) -> np.ndarray:
        # Adobe "group color correction": undo backdrop contribution.
        return cu.clip(
            self._color
            + (self._color - self._color_0)
            * (cu.divide(self._alpha_0, self._alpha_g) - self._alpha_0)
        )

    @property
    def shape(self) -> np.ndarray:
        return self._shape_g

    @property
    def alpha(self) -> np.ndarray:
        return self._alpha_g


# ---------------------------------------------------------------------------
# clip-tools layer adapters
# ---------------------------------------------------------------------------


def _is_group(layer) -> bool:
    return bool(layer.metadata.layer_folder & LayerFolderBit.IS_FOLDER)


def _blend_mode(layer) -> LayerComposite:
    try:
        return LayerComposite(layer.metadata.layer_composite)
    except (ValueError, TypeError):
        return LayerComposite.NORMAL


def _leaf_raster(layer) -> Optional[np.ndarray]:
    """Return the leaf raster for ``layer`` as a numpy ndarray (RGBA uint8)."""
    entry = layer._raster.get(layer.layer_id)
    if entry is not None and entry.type in ("raster", "vector"):
        return entry.image
    return None


def _place_raster_on_canvas(
    arr: np.ndarray,
    canvas_size,
    offset_xy: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Convert a uint8 raster to canvas-sized float32 (H, W, 4) in [0, 1],
    placing it at ``offset_xy = (offset_x, offset_y)`` within the canvas.

    Ports the clipping logic from the old ``composit_layers``: when the
    raster extends past a canvas edge (negative offset or image larger
    than remaining space), the overflowing portion is cropped so the
    visible region is placed at the right spot. CSP writes layer images
    larger than the canvas when the painted content extends off-canvas,
    so a top-left-origin paste would wrongly show the off-canvas region.
    """
    arr_f = arr.astype(np.float32)
    if arr.dtype == np.uint8:
        arr_f = arr_f / 255.0
    elif arr_f.max() > 1.0 + 1e-6:
        arr_f = arr_f / 255.0

    # Normalize channel count to RGBA.
    if arr_f.ndim == 2:
        arr_f = np.stack([arr_f, arr_f, arr_f, np.full_like(arr_f, 1.0)], axis=2)
    elif arr_f.shape[2] == 3:
        alpha = np.ones((*arr_f.shape[:2], 1), dtype=np.float32)
        arr_f = np.concatenate([arr_f, alpha], axis=2)

    h_c, w_c = canvas_size
    canvas = np.zeros((h_c, w_c, 4), dtype=np.float32)

    offset_x, offset_y = int(offset_xy[0]), int(offset_xy[1])
    img_h, img_w = arr_f.shape[:2]

    # X clipping — mirror the old composit_layers logic for signed offsets.
    if offset_x < 0:
        src_x0 = -offset_x
        dst_x0 = 0
    else:
        src_x0 = 0
        dst_x0 = offset_x
    src_x1 = min(img_w, src_x0 + (w_c - dst_x0))
    width = src_x1 - src_x0
    if width <= 0:
        return canvas
    dst_x1 = dst_x0 + width

    # Y clipping — same pattern.
    if offset_y < 0:
        src_y0 = -offset_y
        dst_y0 = 0
    else:
        src_y0 = 0
        dst_y0 = offset_y
    src_y1 = min(img_h, src_y0 + (h_c - dst_y0))
    height = src_y1 - src_y0
    if height <= 0:
        return canvas
    dst_y1 = dst_y0 + height

    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = arr_f[src_y0:src_y1, src_x0:src_x1]
    return canvas


def _crop_to_viewport(arr: np.ndarray, viewport: Viewport) -> np.ndarray:
    left, top, right, bottom = viewport
    return arr[top:bottom, left:right]


def _solid_fill_rgba(layer) -> Optional[np.ndarray]:
    """Return canvas-sized float32 RGBA for a draw_color_enable solid-fill layer.

    CSP's paper layer ("用紙") has no raster entry; its content is a flat
    color stored in draw_color_main_red/green/blue scaled by 2^32-1.
    """
    meta = layer.metadata
    if float(getattr(meta, "draw_color_enable", 0.0) or 0.0) != 1.0:
        return None
    scale = 1.0 / (2**32 - 1)
    r = scale * float(meta.draw_color_main_red or 0.0)
    g = scale * float(meta.draw_color_main_green or 0.0)
    b = scale * float(meta.draw_color_main_blue or 0.0)
    h, w = layer._canvas_size
    rgba = np.empty((h, w, 4), dtype=np.float32)
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = 1.0
    return rgba
