from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple, Optional, Iterator
import numpy as np
import pandas as pd
from PIL import Image

import logging

from clip_tools.blending import composite_layer
from clip_tools.composite import composite_pil as _composite_pil
from clip_tools.constants import DEBUG, LayerComposite, LayerFolderBit, LayerLockBit
from clip_tools.structs import (
    process_text_attributes,
    process_resizable_image_attributes,
)
from clip_tools.types import LayerEntry, LayerRecord
from clip_tools.utils import (
    calculate_homography,
    backward_mapping,
    alpha_blend,
    arr_to_pil,
)

logger = logging.getLogger(__name__)


class ClipLayer:
    def __init__(
        self,
        record: pd.DataFrame,
        idx: int,
        raster: Dict[int, LayerEntry],
        canvas_size: Tuple[int, int],
        layer_map: Optional[Dict[int, int]] = None,
        instance_map: Optional[Dict[int, "ClipLayer"]] = None,
    ):
        """
        Args:
            record (pd.DataFrame): Layer dataframe.
            idx (int): Index of the layer in the layer dataframe.
            raster (dict): Map from layer_id to the raster image.
            canvas_size (Tuple[int, int]): Size of the canvas.
            layer_map (Optional[Dict[int, int]]): Map from layer_id to index in the layer dataframe.
            instance_map (Optional[Dict[int, ClipLayer]]): Shared map from
                layer_id to the single ``ClipLayer`` instance for that layer.
                Populated during the recursive tree build; ``parent`` / tree
                walks look up here so every reference to the same layer is
                the same Python object (identity semantics match psd-tools).
                Passing ``None`` at the root starts a fresh map.
        """
        self._record = record
        self._idx = idx
        self._raster = raster
        self._canvas_size = canvas_size

        self._text_attrs = None
        self._resizable_image_attrs = None
        self._composited = None

        # Snapshot the layer's row into a typed LayerRecord once, at
        # construction time. Reads (``.name``, ``.visible``, ``.opacity``,
        # ``.blend_mode``, ``.bbox``) go through this snapshot instead of
        # doing a fresh ``_record.loc[idx]`` + ``LayerRecord.from_row()``
        # per access. The compositor hot path reads several attributes per
        # layer per pass, so eliminating that per-attribute DataFrame
        # dispatch is the main speedup for Issue #28. Setters update
        # ``_meta`` and ``_record`` together (see ``visible.setter``,
        # ``opacity.setter``) so the DataFrame remains a source of truth
        # for anything that needs to round-trip the file back to .clip.
        self._meta = LayerRecord.from_row(record.loc[idx])

        if layer_map is None:
            self._layer_map = {}
            for index, row in self._record.iterrows():
                self._layer_map[row["MainId"]] = index
        else:
            self._layer_map = layer_map

        # Register this instance in the shared instance map before
        # recursing to children, so a child's ``parent`` lookup can find
        # us. The map is shared across the whole tree, so one layer is
        # represented by exactly one ``ClipLayer`` object: this is the
        # identity contract that lets ``layer.parent is layer.parent``
        # return True and matches psd-tools' tree semantics.
        if instance_map is None:
            self._instance_map = {}
        else:
            self._instance_map = instance_map
        self._instance_map[self.layer_id] = self

        self._children = []
        for child_id in self.children_ids:
            self._children.append(
                ClipLayer(
                    self._record,
                    self._layer_map[child_id],
                    self._raster,
                    self._canvas_size,
                    self._layer_map,
                    self._instance_map,
                )
            )

    @property
    def metadata(self) -> LayerRecord:
        """Typed view of this layer's row in the dataframe.

        Returns the snapshot built at construction time
        (``__init__`` set ``self._meta`` once). Reads are a plain attribute
        access with no DataFrame lookup. Writes (``visible.setter``,
        ``opacity.setter``) mutate this snapshot and the DataFrame together,
        so the DataFrame remains in sync for round-trip serialization
        while reads stay fast.
        """
        return self._meta

    @property
    def layer_id(self) -> int:
        return int(self._meta.main_id)

    @property
    def name(self) -> str:
        return self.metadata.layer_name

    @property
    def layer_type(self) -> str:
        if self.layer_id in self._raster:
            return self._raster[self.layer_id].type
        else:
            return "raster"

    @property
    def parent_id(self) -> int:
        return self.metadata.parent_layer

    @property
    def children_ids(self) -> List[int]:
        return list(
            self._record[self._record["ParentLayer"] == self.layer_id]["MainId"].values
        )

    @property
    def children_names(self) -> List[str]:
        return [child.name for child in self._children]

    @property
    def parent(self) -> Optional[ClipLayer]:
        """Return the parent ``ClipLayer``, or ``None`` if this is the root.

        Looks up the shared instance map built during the recursive tree
        construction, so every call returns the *same* object:
        ``layer.parent is layer.parent`` → ``True``. This matches
        psd-tools' tree semantics, where every layer in the tree has
        exactly one Python object representing it, and lets callers rely
        on identity comparisons (``layer.parent is other_layer``).

        The root layer (``ClipImage``) has no parent and returns ``None``;
        the previous behaviour of raising on a missing parent came from
        creating a new ``ClipLayer`` for every call, which also raised
        when ``parent_id`` had no mapping. Returning ``None`` for the
        root makes the ascending walk safe.
        """
        return self._instance_map.get(self.parent_id)

    @property
    def children(self) -> List[ClipLayer]:
        return self._children

    @property
    def visible(self) -> bool:
        return self.metadata.layer_visibility != 0

    @visible.setter
    def visible(self, value: bool) -> None:
        # Update both the snapshot and the DataFrame together: reads come
        # from the snapshot, but the DataFrame is still the source of
        # truth for anything that round-trips back to the .clip file.
        v = 1 if value else 0
        self._meta.layer_visibility = v
        self._record.loc[self._idx, "LayerVisibility"] = v

    @property
    def opacity(self) -> float:
        return self.metadata.layer_opacity / 256.0

    @opacity.setter
    def opacity(self, value: float) -> None:
        # Same two-write pattern as ``visible`` — see the comment there.
        v = int(value * 256)
        self._meta.layer_opacity = v
        self._record.loc[self._idx, "LayerOpacity"] = v

    @property
    def blend_mode(self) -> LayerComposite:
        """Per-layer blend mode (``LayerComposite`` enum).

        psd-tools exposes the equivalent as ``.blend_mode`` returning
        ``psd_tools.constants.BlendMode``. The two enums use different
        value types (CSP numeric IDs vs PSD 4-byte codes) but their
        member *names* overlap for every common mode:
        ``NORMAL / MULTIPLY / SCREEN / OVERLAY / SOFT_LIGHT /
        HARD_LIGHT / DIFFERENCE / HUE / SATURATION / COLOR / LUMINOSITY
        / DARKEN / LIGHTEN / COLOR_BURN / LINEAR_BURN / COLOR_DODGE /
        VIVID_LIGHT / LINEAR_LIGHT / PIN_LIGHT / HARD_MIX / EXCLUSION
        / SUBTRACT / DIVIDE / PASS_THROUGH / DARKER_COLOR / LIGHTER_COLOR``.
        Downstream code that only cares about blend mode by name can
        write ``layer.blend_mode.name`` and get strings that compare
        1:1 across libraries.

        Falls back to ``LayerComposite.NORMAL`` when the record's
        ``layer_composite`` field holds a value outside the known
        enum (matches the fallback inside ``composit_layers``).
        """
        try:
            return LayerComposite(self.metadata.layer_composite)
        except ValueError:
            return LayerComposite.NORMAL

    @property
    def size(self) -> Tuple[int, int]:
        """Canvas size as ``(width, height)``.

        Matches ``psd_tools.PSDImage.size`` and PIL's ``Image.size`` so
        downstream consumers reading ``layer.size`` behave the same
        across PSDImage ↔ ClipImage swaps.

        Note that the backing ``_canvas_size`` stores ``(height, width)``
        internally (numpy / cv2 convention — see ``ClipImage.open``
        which assigns ``canvas_size = (canvas_height, canvas_width)``).
        This property re-orders to the PIL convention so API consumers
        don't need to know about the internal storage order.
        """
        return self.width, self.height

    @property
    def width(self) -> int:
        """Canvas width in pixels.

        Matches ``psd_tools.PSDImage.width`` / ``Layer.width`` so downstream
        consumers that iterate ``layer.width`` interchangeably across both
        libraries work unchanged. For now descendant layers inherit the
        canvas width — a per-layer bbox-derived width will follow once
        bbox returns proper right/bottom coordinates (see
        ``ClipLayer.bbox`` which currently returns zeros for the extent).
        """
        return int(self._canvas_size[1])

    @property
    def height(self) -> int:
        """Canvas height in pixels. See :pyattr:`width` for scope notes."""
        return int(self._canvas_size[0])

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        meta = self.metadata

        if meta.text_layer_type is not None and meta.text_layer_attributes is not None:
            if DEBUG:
                if not os.path.exists("temp/text_attributes"):
                    os.makedirs("temp/text_attributes")
                with open(
                    f"temp/text_attributes/{self.layer_id}_text_attribute.binary",
                    mode="wb",
                ) as f:
                    f.write(meta.text_layer_attributes)

            try:
                if self._text_attrs is None:
                    self._text_attrs = process_text_attributes(
                        meta.text_layer_attributes
                    )

                attr_ds = self._text_attrs
                layer_offset_x = meta.layer_offset_x + attr_ds.general_offset_x
                layer_offset_y = meta.layer_offset_y + attr_ds.general_offset_y
            except Exception:
                print(
                    f"WARNING: Text attribute layer load failed for layer_id: {self.layer_id}"
                )
                layer_offset_x = meta.layer_offset_x
                layer_offset_y = meta.layer_offset_y
        else:
            layer_offset_x = 0
            layer_offset_y = 0

        # Once the layer's offscreen has been refreshed, render-offscreen
        # offsets are already baked into the offscreen and we shouldn't add
        # them again. DrawToRenderOffscreenType is the dirty/applied flag.
        if meta.draw_to_render_offscreen_type is None:
            layer_offset_x += meta.layer_render_offscr_offset_x
            layer_offset_y += meta.layer_render_offscr_offset_y

        layer_offset_x += meta.layer_offset_x
        layer_offset_y += meta.layer_offset_y

        return (layer_offset_x, layer_offset_y, 0, 0)

    def is_group(self) -> bool:
        return bool(self.metadata.layer_folder & LayerFolderBit.IS_FOLDER)

    def __iter__(self) -> Iterator[ClipLayer]:
        return iter(self._children)

    def __len__(self) -> int:
        return len(self._children)

    def descendants(self, include_clip: bool = True) -> Iterator[ClipLayer]:
        """Depth-first iterator yielding every descendant layer.

        Matches ``psd_tools.PSDImage.descendants`` / ``Group.descendants``
        in both signature and traversal order: pre-order DFS, children
        before grandchildren, yields groups and leaves alike. Does not
        yield ``self``.

        :param include_clip: Kept for signature parity with psd-tools.
            In psd-tools this filters out adjustment-layer clipping
            masks; CSP's layer model doesn't have a matching concept
            so clip-tools accepts and ignores this parameter.

        Example::

            for layer in img.descendants():
                print(layer.name)
        """
        for layer in self:
            yield layer
            if layer.is_group():
                yield from layer.descendants(include_clip=include_clip)

    def find(self, name: str) -> Optional[ClipLayer]:
        """Depth-first search for the first descendant with ``name``.

        Matches ``psd_tools.PSDImage.find`` / ``Group.find``: returns the
        first layer whose ``.name == name`` in DFS pre-order, or ``None``
        if nothing matches. First match wins — no ambiguity resolution.

        For the "find all layers with this name" case use a comprehension
        over ``descendants()`` (``[l for l in img.descendants() if l.name == name]``);
        psd-tools exposes this as ``findall()`` but kakidashi-kun's
        current consumers only need first-match semantics, so clip-tools
        starts with the smaller surface here.
        """
        for layer in self.descendants():
            if layer.name == name:
                return layer
        return None

    def composite(
        self,
        viewport: Optional[Tuple[int, int, int, int]] = None,
        force: bool = False,
        layer_filter: Optional[Callable[["ClipLayer"], bool]] = None,
        prepended_layers: Optional[List["ClipLayer"]] = None,
    ) -> Optional[Image.Image]:
        """Composite this layer and return the result as a PIL image.

        psd-tools parity: this matches ``PSDImage.composite`` /
        ``Group.composite`` in both the shape of the keyword arguments and
        their semantics. Downstream consumers (e.g. kakidashi-kun) can call
        ``img.composite(viewport=..., force=True, layer_filter=lambda l: ...)``
        identically against a PSDImage or a ClipImage.

        :param viewport: ``(left, top, right, bottom)`` crop applied to the
            composited result. ``None`` returns the full canvas.
        :param force: when True, do not use the per-layer cached recomposite
            (``self._composited``) nor the group's pre-merged raster as a
            fallback. Leaf raster/vector entries ARE the layer's content
            and are always used.
        :param layer_filter: predicate called on each candidate layer.
            Only layers for which it returns True participate. Passed down
            recursively so nested groups also see the filter. ``None``
            means "no filtering" (all layers participate).
        :param prepended_layers: layers composited UNDER the children,
            as a base. Kept for backward compat with the internal recursive
            composite path.
        """
        # Delegate to the new clip_tools.composite engine (port of psd-tools'
        # Compositor). The engine handles leaf / group / pass-through / knockout
        # uniformly with (color, shape, alpha) float32 + Porter-Duff over, so
        # the only paths we still inline here are:
        #   - draw_color_enable solid fills (no raster entry to feed the engine)
        #   - the group-raster fallback (ClipStudio flattens group strokes into
        #     the group's own cached raster and leaves child leaves empty)
        #
        # `prepended_layers` is kept for backward compat with the internal
        # recursion but is no longer used by the engine; a non-empty value
        # falls back to the old composit_layers path.
        prepended = list(prepended_layers) if prepended_layers else []
        entry = self._raster.get(self.layer_id)
        meta = self.metadata

        # Solid-color fill: no raster to feed the engine.
        if entry is None and meta.draw_color_enable == 1.0:
            scale = 255.0 / (2**32 - 1)
            red = scale * (meta.draw_color_main_red or 0.0)
            green = scale * (meta.draw_color_main_green or 0.0)
            blue = scale * (meta.draw_color_main_blue or 0.0)
            bg_color = np.array([red, green, blue, 255], dtype=np.uint8)
            composited = np.zeros((*self._canvas_size, 4), dtype=np.uint8)
            composited[..., :4] = bg_color
            # Apply self.opacity for parity with group-call semantics.
            if self.opacity < 1.0:
                composited = composited.copy()
                composited[..., 3] = (
                    composited[..., 3].astype(np.float32) * float(self.opacity)
                ).astype(np.uint8)
            result = arr_to_pil(composited)
            if viewport is not None:
                result = result.crop(viewport)
            return result

        # Prepended-layers path: internal recursive callers rely on the
        # old composit_layers behaviour (compositing a stack of ad-hoc
        # layers under the group's children). Keep the pre-engine code
        # for those cases; external callers never pass prepended_layers.
        if prepended:
            children = list(self._children)
            if layer_filter is not None:
                children = [c for c in children if layer_filter(c)]
            if not children:
                return None
            layers_to_composit = prepended + children
            composited = self.composit_layers(
                layers_to_composit, layer_filter=layer_filter, force=force
            )
            if composited.shape[-1] >= 4 and self.opacity < 1.0:
                composited = composited.copy()
                composited[..., 3] = (
                    composited[..., 3].astype(np.float32) * float(self.opacity)
                ).astype(np.uint8)
            result = arr_to_pil(composited)
            if viewport is not None:
                result = result.crop(viewport)
            return result

        # Main path: hand off to the new engine. For a regular group we pass
        # as_layer=True (psd-tools' Group.composite semantics — applies the
        # group's own opacity). For the image root (ClipImage, parity with
        # psd-tools' PSDImage.composite), we pass as_layer=False so the engine
        # iterates the root's children directly instead of applying the root
        # as a single layer: a layer_filter passed by the caller is a
        # children-selector, and treating the root as a single layer would
        # wrongly gate the whole render on filter(root).
        from clip_tools.api.clip_image import ClipImage  # noqa: PLC0415

        as_layer = not isinstance(self, ClipImage)
        try:
            result = _composite_pil(
                self,
                viewport=viewport,
                layer_filter=layer_filter,
                force=force,
                as_layer=as_layer,
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("composite engine failed on layer %s: %s", self.name, exc)
            raise

        # Group-raster fallback: when a group's child composite produced a
        # fully transparent image but ClipStudio cached a pre-merged raster,
        # prefer the cached raster (unless `force` requested a fresh render).
        if (
            not force
            and entry is not None
            and entry.type == "group"
            and isinstance(result, Image.Image)
            and result.mode == "RGBA"
        ):
            alpha_arr = np.asarray(result.split()[-1])
            if not alpha_arr.any():
                fallback = entry.image
                if self.opacity < 1.0:
                    fallback = fallback.copy()
                    fallback[..., 3] = (
                        fallback[..., 3].astype(np.float32) * float(self.opacity)
                    ).astype(np.uint8)
                result = arr_to_pil(fallback)
                if viewport is not None:
                    result = result.crop(viewport)

        return result

    def composit_layers(
        self,
        layer_list: List[ClipLayer],
        layer_filter: Optional[Callable[["ClipLayer"], bool]] = None,
        force: bool = False,
    ) -> np.ndarray:
        """Composite a list of layers against a white backdrop and return RGBA.

        Internal method called by :py:meth:`composite` when a group needs to
        render its children. ``layer_filter`` and ``force`` are forwarded to
        each child's own ``composite()`` call so filtering and cache-skipping
        propagate through nested groups.
        """

        def layer_composite(meta: LayerRecord) -> LayerComposite:
            try:
                return LayerComposite(meta.layer_composite)
            except ValueError:
                return LayerComposite.NORMAL

        def layer_alpha_lock(meta: LayerRecord) -> bool:
            return bool(meta.layer_lock & LayerLockBit.ALPHA)

        def unpremultiply_rgba(arr: np.ndarray) -> np.ndarray:
            if arr.shape[-1] < 4:
                return arr
            a = arr[..., 3:4].astype(np.float32) / 255.0
            safe_a = np.where(a == 0, 1.0, a)
            rgb = np.clip(arr[..., :3].astype(np.float32) / safe_a, 0, 255)
            out = arr.copy()
            out[..., :3] = np.where(a > 0, rgb, 0).round().astype(np.uint8)
            return out

        # Transparent backdrop so the output RGBA reflects actual layer
        # coverage, matching psd-tools's convention (callers flatten onto
        # white when they want an opaque image — see save_jpg's convert('RGB')).
        # Previously the buffer was pre-filled with RGB=255, which silently
        # pre-mixed white into the composite: a semi-transparent BG at 45%
        # opacity blended with the white backdrop produced
        # 0.45*BG_rgb + 0.55*white (light pastel) instead of the raw BG
        # content with alpha=115. When that intermediate was then alpha-
        # blended with LO on top, the result diverged visibly from
        # psd-tools's output (seen as mean pixel diff ≈ 45 in the
        # kakidashi-kun BG_LO parity check).
        buffer = np.zeros((*self._canvas_size, 4), dtype=np.uint8)

        for i, layer in enumerate(layer_list):
            layer_id = layer.layer_id
            layer_type = layer.layer_type
            layer_img = np.array(
                layer.composite(layer_filter=layer_filter, force=force)
            )

            if len(layer_img.shape) == 0:
                continue

            meta = layer.metadata

            if meta.resizable_image_info is not None:
                if DEBUG:
                    if not os.path.exists("temp/resizable_image_info"):
                        os.makedirs("temp/resizable_image_info")
                    with open(
                        f"temp/resizable_image_info/{layer_id}_resizable_image_info.binary",
                        mode="wb",
                    ) as f:
                        f.write(meta.resizable_image_info)

                layer._resizable_image_attrs = process_resizable_image_attributes(
                    meta.resizable_image_info
                )

                layer_buffer = np.zeros((*self._canvas_size, 4), dtype=np.uint8)

                source_coords = layer._resizable_image_attrs.source_coords
                polygon_coords = layer._resizable_image_attrs.polygon_coords
                transform = calculate_homography(source_coords, polygon_coords)
                layer_buffer = backward_mapping(
                    transform, layer_img, layer_buffer, polygon_coords
                )
                mode = layer_composite(meta)
                if mode == LayerComposite.NORMAL:
                    buffer = alpha_blend(buffer, layer_buffer, premultiplied=False)
                else:
                    buffer = composite_layer(
                        buffer,
                        layer_buffer,
                        mode,
                        preserve_transparency=layer_alpha_lock(meta),
                    )

            else:
                if layer_img is None:
                    logger.debug(f"Skipping None layer idx: {i}, id: {layer_id}")
                    continue

                if meta.layer_visibility == 0:
                    logger.debug(f"Skipping {layer.name}")
                    continue

                layer_offset_x, layer_offset_y = layer.bbox[:2]
                layer_height, layer_width = layer_img.shape[:2]

                if layer_offset_x < 0:
                    layer_width = layer_img.shape[1] + layer_offset_x
                    layer_offset_x = 0
                    layer_img = layer_img[:, -layer_width:]
                else:
                    if layer_width + layer_offset_x > buffer.shape[1]:
                        layer_width = buffer.shape[1] - layer_offset_x
                    layer_img = layer_img[:, :layer_width]

                if layer_offset_y < 0:
                    layer_height = layer_img.shape[0] + layer_offset_y
                    layer_offset_y = 0
                    layer_img = layer_img[-layer_height:, :]
                else:
                    if layer_height + layer_offset_y > buffer.shape[0]:
                        layer_height = buffer.shape[0] - layer_offset_y
                    layer_img = layer_img[:layer_height, :]

                if layer_height > buffer.shape[0]:
                    layer_height = buffer.shape[0]
                    layer_img = layer_img[:layer_height]
                if layer_width > buffer.shape[1]:
                    layer_width = buffer.shape[1]
                    layer_img = layer_img[:, :layer_width]

                premultiplied = layer_type == "vector"
                mode = layer_composite(meta)
                base_slice = buffer[
                    layer_offset_y : layer_offset_y + layer_height,
                    layer_offset_x : layer_offset_x + layer_width,
                ]
                if mode == LayerComposite.NORMAL:
                    composited = alpha_blend(
                        base_slice, layer_img, premultiplied=premultiplied
                    )
                else:
                    blend_input = (
                        unpremultiply_rgba(layer_img) if premultiplied else layer_img
                    )
                    composited = composite_layer(
                        base_slice,
                        blend_input,
                        mode,
                        preserve_transparency=layer_alpha_lock(meta),
                    )
                buffer[
                    layer_offset_y : layer_offset_y + layer_height,
                    layer_offset_x : layer_offset_x + layer_width,
                ] = composited

        return buffer
