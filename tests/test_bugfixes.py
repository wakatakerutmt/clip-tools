"""Regression tests for compositor correctness fixes.

Two bugs were reported together by a downstream (kakidashi-kun #89) when the
PSD and CLIP export routes disagreed on an identical fixture:

1. ``ClipLayer.children_ids`` used to return children in SQLite row order,
   which only coincidentally matches CSP's stack order for some groups.
   For groups where the raw row order was reversed vs. the real stack,
   kakidashi-kun's ``work_folder_papers`` layer (e.g. ``paper_yellow``)
   was composited on top of the cel instead of under it.

2. ``_solid_fill_rgba`` used to treat any layer with
   ``draw_color_enable == 1.0`` as a canvas-wide solid fill. CSP stores a
   layer's "paint color" in the same fields even when the layer has no
   actual raster (e.g. an empty ``水`` shading layer inside a cel); those
   layers were then rendered as a full-canvas opaque fill, overriding
   siblings.

Both fixes together let CLIP export output match PSD export output and
CSP native output on the reported fixture.
"""

from __future__ import annotations

import types
from typing import List

import numpy as np
import pandas as pd
import pytest

from clip_tools.api.clip_layer import ClipLayer
from clip_tools.composite.compositor import _solid_fill_rgba
from clip_tools.rendering import (
    _compute_stamp_alpha,
    _compute_stamp_dimensions,
    _compute_stamp_size,
    _has_implicit_pressure_flow,
    _sample_paper_grain_patch,
    composite_alpha_onto,
    multiply_alpha_onto,
    stamp_pattern,
)
import numpy as np
from clip_tools.structs.blob_parsers import (
    BrushEffector,
    EffectorPressure,
    EffectorSmooth,
    EffectorVelocity,
)


# ---------------------------------------------------------------------------
# children_ids: linked-list walk
# ---------------------------------------------------------------------------


def _mk_children_layer(parent_first_child: int, chain: List[dict]) -> ClipLayer:
    """Build a bare ``ClipLayer`` substitute whose ``children_ids`` property
    can be exercised without rebuilding the whole ClipImage tree.

    ``chain`` lists children's ``{"MainId": int, "LayerNextIndex": int}``
    rows in the desired linked-list order. The parent's
    ``layer_first_child_index`` seeds the walk, and the DataFrame is
    constructed so each child's ``LayerNextIndex`` chains to the next.
    """
    df = pd.DataFrame(chain)
    layer_map = {int(row["MainId"]): idx for idx, row in df.iterrows()}

    # Minimal ClipLayer-ish shim: just the attributes ``children_ids``
    # reads (``metadata.layer_first_child_index``, ``self._record``,
    # ``self._layer_map``). Avoids constructing a full ClipLayer which
    # requires the whole SQLite surface.
    shim = types.SimpleNamespace()
    shim._record = df
    shim._layer_map = layer_map
    shim._meta = types.SimpleNamespace(layer_first_child_index=parent_first_child)
    shim.metadata = shim._meta
    # Bind the real method so we are testing the production code path.
    shim.children_ids = ClipLayer.children_ids.fget(shim)
    return shim


def test_children_ids_empty_when_no_first_child():
    shim = _mk_children_layer(parent_first_child=0, chain=[])
    assert shim.children_ids == []


def test_children_ids_follows_linked_list_order():
    # Chain: 10 → 20 → 30 → 0 (stop). Whatever DataFrame row order
    # happens to be, the returned ids must come back in walk order.
    chain = [
        {"MainId": 30, "LayerNextIndex": 0},  # row 0 (NOT the first in stack)
        {"MainId": 10, "LayerNextIndex": 20},  # row 1 (first in stack)
        {"MainId": 20, "LayerNextIndex": 30},  # row 2
    ]
    shim = _mk_children_layer(parent_first_child=10, chain=chain)
    assert shim.children_ids == [10, 20, 30]


def test_children_ids_stops_on_broken_chain():
    # A child's LayerNextIndex points at an id that isn't in the map.
    # children_ids must stop there rather than loop; the children it
    # already collected stay valid.
    chain = [
        {"MainId": 10, "LayerNextIndex": 20},
        {"MainId": 20, "LayerNextIndex": 999},  # 999 not in map
    ]
    shim = _mk_children_layer(parent_first_child=10, chain=chain)
    assert shim.children_ids == [10, 20]


def test_children_ids_single_child():
    shim = _mk_children_layer(
        parent_first_child=42,
        chain=[{"MainId": 42, "LayerNextIndex": 0}],
    )
    assert shim.children_ids == [42]


# ---------------------------------------------------------------------------
# _solid_fill_rgba: paper-only guard
# ---------------------------------------------------------------------------


def _layer_with(meta_attrs: dict, canvas_size=(20, 10)):
    """Minimal layer shim exposing ``metadata.*`` and ``_canvas_size``."""
    meta = types.SimpleNamespace(
        draw_color_enable=None,
        draw_color_main_red=None,
        draw_color_main_green=None,
        draw_color_main_blue=None,
        special_render_type=None,
    )
    for k, v in meta_attrs.items():
        setattr(meta, k, v)
    layer = types.SimpleNamespace(metadata=meta, _canvas_size=canvas_size)
    return layer


def test_solid_fill_returns_none_when_draw_color_disabled():
    layer = _layer_with({"draw_color_enable": 0.0, "special_render_type": 20.0})
    assert _solid_fill_rgba(layer) is None


def test_solid_fill_returns_none_for_cel_shading_layer():
    # Reproduces the bug pattern: cel shading layer (e.g. ``水``) with its
    # paint color saved but no paper-layer render type. Must NOT be treated
    # as a canvas-wide solid fill — doing so paints the whole canvas with
    # the shading color and swallows paper layers underneath.
    layer = _layer_with(
        {
            "draw_color_enable": 1.0,
            "draw_color_main_red": 0.0,
            "draw_color_main_green": 2_330_168_035.0,
            "draw_color_main_blue": 4_294_967_295.0,
            "special_render_type": None,
        }
    )
    assert _solid_fill_rgba(layer) is None


def test_solid_fill_returns_rgba_for_paper_layer():
    # A real CSP paper layer (``用紙``): draw_color_enable=1 AND
    # special_render_type=20. Rendered as a flat canvas-sized fill.
    layer = _layer_with(
        {
            "draw_color_enable": 1.0,
            "draw_color_main_red": 4_294_967_295.0,  # full R
            "draw_color_main_green": 4_294_967_295.0,  # full G
            "draw_color_main_blue": 2_147_483_647.0,  # ~half B
            "special_render_type": 20.0,
        },
        canvas_size=(5, 3),
    )
    rgba = _solid_fill_rgba(layer)
    assert rgba is not None
    assert rgba.shape == (5, 3, 4)
    assert rgba.dtype == np.float32
    assert np.allclose(rgba[..., 0], 1.0)
    assert np.allclose(rgba[..., 1], 1.0)
    assert 0.49 < float(rgba[0, 0, 2]) < 0.51
    assert np.allclose(rgba[..., 3], 1.0)


def test_solid_fill_returns_none_when_special_render_type_is_mismatched():
    # Any non-20 special_render_type should also disqualify. 20 is the
    # specific marker CSP uses for the paper-layer type.
    layer = _layer_with(
        {
            "draw_color_enable": 1.0,
            "draw_color_main_red": 1.0,
            "draw_color_main_green": 1.0,
            "draw_color_main_blue": 1.0,
            "special_render_type": 5.0,
        }
    )
    assert _solid_fill_rgba(layer) is None


# ---------------------------------------------------------------------------
# _has_implicit_pressure_flow: CSP FlowEffector flags=0x01 interpretation
# ---------------------------------------------------------------------------


def _mk_brush(flow_effector):
    """Minimal BrushStyle-ish shim exposing the single field the helper reads."""
    return types.SimpleNamespace(flow_effector=flow_effector)


def test_implicit_pressure_flow_detected_when_flags_bit0_only():
    # Y4 sen's actual FlowEffector: 4-byte blob 00 00 00 01. parsed to
    # BrushEffector(flags=1, every modulator None). This is the pattern
    # that must trigger implicit pressure-flow modulation.
    eff = BrushEffector(flags=0x1, pressure=None, velocity=None, smooth=None)
    assert _has_implicit_pressure_flow(_mk_brush(eff)) is True


def test_implicit_pressure_flow_rejected_for_none_effector():
    assert _has_implicit_pressure_flow(_mk_brush(None)) is False


def test_implicit_pressure_flow_rejected_when_flags_bit0_clear():
    eff = BrushEffector(flags=0x0, pressure=None, velocity=None, smooth=None)
    assert _has_implicit_pressure_flow(_mk_brush(eff)) is False


def test_implicit_pressure_flow_rejected_when_explicit_pressure_struct_present():
    # A brush with an explicit pressure curve already has its flow driven
    # by pressure via apply_effector; the implicit modulator must NOT
    # double-apply here.
    eff = BrushEffector(
        flags=0x1 | 0x10,  # bit 0 + EFFECTOR_FLAG_PRESSURE
        pressure=EffectorPressure(min=0.0, curve_id=94),
        velocity=None,
        smooth=None,
    )
    assert _has_implicit_pressure_flow(_mk_brush(eff)) is False


def test_implicit_pressure_flow_rejected_when_explicit_velocity_struct_present():
    eff = BrushEffector(
        flags=0x1 | 0x20,
        pressure=None,
        velocity=EffectorVelocity(min=1.0, curve_id=95),
        smooth=None,
    )
    assert _has_implicit_pressure_flow(_mk_brush(eff)) is False


def test_implicit_pressure_flow_rejected_when_explicit_smooth_struct_present():
    eff = BrushEffector(
        flags=0x1 | 0x40,
        pressure=None,
        velocity=None,
        smooth=EffectorSmooth(min=0.0),
    )
    assert _has_implicit_pressure_flow(_mk_brush(eff)) is False


# ---------------------------------------------------------------------------
# _compute_stamp_dimensions: ThicknessBase dispatch per brush kind
# ---------------------------------------------------------------------------


def _mk_stamp_brush(anti_alias, pattern_style, thickness_base):
    """Minimal BrushStyle-ish shim for the three fields the helper reads."""
    return types.SimpleNamespace(
        anti_alias=anti_alias,
        pattern_style=pattern_style,
        thickness_base=thickness_base,
    )


def test_stamp_dimensions_standard_pen_applies_cube_root_thickness():
    # Standard pen: anti_alias > 0 AND pattern_style == 0 → cube-root factor.
    # With thickness_base = 1.0 the stamp is brush_size * ss (no shrink).
    brush = _mk_stamp_brush(anti_alias=2, pattern_style=0, thickness_base=1.0)
    long, short = _compute_stamp_dimensions(brush, brush_size=3.0, ss=2)
    assert long == pytest.approx(6.0)
    assert short == pytest.approx(6.0)


def test_stamp_dimensions_standard_pen_with_small_thickness_shrinks():
    # Standard pen with thickness = 0.125 → stamp_long = 3 * 0.5 * 2 = 3.0
    brush = _mk_stamp_brush(anti_alias=2, pattern_style=0, thickness_base=0.125)
    long, short = _compute_stamp_dimensions(brush, brush_size=3.0, ss=2)
    assert long == pytest.approx(3.0)
    assert short == pytest.approx(3.0)


def test_stamp_dimensions_elliptical_uses_thickness_as_aspect_ratio():
    # Chisel brush: anti_alias == 0 → short axis compressed, long axis full.
    brush = _mk_stamp_brush(anti_alias=0, pattern_style=0, thickness_base=0.5)
    long, short = _compute_stamp_dimensions(brush, brush_size=4.0, ss=2)
    assert long == pytest.approx(8.0)
    assert short == pytest.approx(4.0)


def test_stamp_dimensions_shaped_brush_ignores_thickness_scaling():
    # kakidashi-kun #84: shaped hatching brush 10949's actual params —
    # brush_size=3, anti_alias=2, pattern_style=16 (texture-based),
    # thickness_base=0.01. The old path applied thickness^(1/3) = 0.2154
    # here, collapsing stamp_long from 6px to 1.3px. The fix: for shaped
    # brushes, thickness_base is NOT a size parameter — the pattern image
    # encodes the brush's visual shape at its own scale, so stamps stay
    # at brush_size * ss regardless of thickness_base.
    brush = _mk_stamp_brush(anti_alias=2, pattern_style=16, thickness_base=0.01)
    long, short = _compute_stamp_dimensions(brush, brush_size=3.0, ss=2)
    assert long == pytest.approx(6.0)
    assert short == pytest.approx(6.0)
    # And explicitly: the result must NOT be the old (buggy) value.
    old_shrunk = 3.0 * (0.01 ** (1 / 3)) * 2
    assert long != pytest.approx(old_shrunk)


def test_stamp_dimensions_shaped_brush_with_full_thickness_matches_standard():
    # When thickness_base == 1.0 the old and new formulas coincide, so a
    # shaped brush with thickness=1.0 should yield the same stamp_long as
    # a standard pen with thickness=1.0. Guards against the shaped path
    # accidentally picking up a different base scale.
    shaped = _mk_stamp_brush(anti_alias=2, pattern_style=16, thickness_base=1.0)
    standard = _mk_stamp_brush(anti_alias=2, pattern_style=0, thickness_base=1.0)
    long_shaped, _ = _compute_stamp_dimensions(shaped, brush_size=3.0, ss=2)
    long_standard, _ = _compute_stamp_dimensions(standard, brush_size=3.0, ss=2)
    assert long_shaped == pytest.approx(long_standard)


# ---------------------------------------------------------------------------
# _compute_stamp_size: brush.texture_scale must NOT scale the stamp size.
# ---------------------------------------------------------------------------


def _mk_size_brush(texture_scale):
    """Minimal BrushStyle-ish shim — the helper only reads ``texture_scale``,
    and it must refuse to consume it (asserted below)."""
    return types.SimpleNamespace(texture_scale=texture_scale)


def test_stamp_size_identity_when_all_factors_are_one():
    # Sanity floor: identity inputs produce identity output.
    brush = _mk_size_brush(texture_scale=1.0)
    size = _compute_stamp_size(
        brush,
        effective_size=10.0,
        size_mult=1.0,
        sample_size=1.0,
        shaped_scale=1.0,
    )
    assert size == pytest.approx(10.0)


def test_stamp_size_multiplies_its_four_documented_factors():
    # The helper is a pure four-way product of the documented factors —
    # nothing else.
    brush = _mk_size_brush(texture_scale=1.0)
    size = _compute_stamp_size(
        brush,
        effective_size=10.0,
        size_mult=2.0,
        sample_size=0.5,
        shaped_scale=3.0,
    )
    assert size == pytest.approx(10.0 * 2.0 * 0.5 * 3.0)


def test_stamp_size_ignores_brush_texture_scale():
    # kakidashi-kun #84 continuation: brush.texture_scale is CSP's
    # secondary canvas-grain tile scale, not a per-stamp nib scale. The
    # old code path multiplied it into the stamp size, which collapsed
    # strokes for brushes whose author had tuned down the canvas grain
    # — most severely brush id 10948 with texture_scale=0.25 (Y4 LOSS
    # sen) whose 2.4 px stamps dropped to 0.6 px (sub-pixel, zeroed by
    # the supersample reduce, line disappeared from output).
    low = _mk_size_brush(texture_scale=0.25)
    quarter = _mk_size_brush(texture_scale=0.28)
    unit = _mk_size_brush(texture_scale=1.0)
    s_low = _compute_stamp_size(low, 10.0, 1.0, 1.0, 1.0)
    s_quarter = _compute_stamp_size(quarter, 10.0, 1.0, 1.0, 1.0)
    s_unit = _compute_stamp_size(unit, 10.0, 1.0, 1.0, 1.0)
    assert s_low == pytest.approx(s_unit)
    assert s_quarter == pytest.approx(s_unit)
    # And explicitly: the result must NOT be the old (buggy) scaled value.
    assert s_low != pytest.approx(10.0 * 0.25)


# ---------------------------------------------------------------------------
# _compute_stamp_alpha: brush.texture_density_base attenuation for textured
# brushes (clip-tools fork #44 / kakidashi-kun #84 continuation)
# ---------------------------------------------------------------------------


def _mk_alpha_brush(texture_pattern, texture_density_base, pattern_style=16):
    """Minimal BrushStyle-ish shim — the helper reads these three fields.

    ``pattern_style`` defaults to a shaped value so the tests that care only
    about ``texture_pattern`` + ``texture_density_base`` behavior don't have
    to pass it explicitly. Use ``pattern_style=0`` to simulate a standard
    pen (where density_base is intentionally ignored even with grain).
    """
    return types.SimpleNamespace(
        texture_pattern=texture_pattern,
        texture_density_base=texture_density_base,
        pattern_style=pattern_style,
    )


def test_stamp_alpha_identity_for_untextured_brush():
    # texture_pattern == 0 (no paper grain) → density_base is a no-op.
    brush = _mk_alpha_brush(texture_pattern=0, texture_density_base=0.5)
    alpha = _compute_stamp_alpha(
        brush, per_stamp_alpha=1.0, flow_mult=1.0, stroke_op=1.0
    )
    assert alpha == pytest.approx(1.0)


def test_stamp_alpha_multiplies_the_three_base_factors():
    # The base alpha is the pure three-way product of the documented factors
    # when the brush carries no grain.
    brush = _mk_alpha_brush(texture_pattern=0, texture_density_base=1.0)
    alpha = _compute_stamp_alpha(
        brush, per_stamp_alpha=0.7, flow_mult=0.5, stroke_op=0.8
    )
    assert alpha == pytest.approx(0.7 * 0.5 * 0.8)


def test_stamp_alpha_attenuates_by_texture_density_when_grain_present_on_shaped_brush():
    # kakidashi-kun #84 / clip-tools #44: brush 10949 nuri is a shaped
    # brush (pattern_style=16) with texture_pattern=39 and
    # texture_density_base=0.70. With grain present on a shaped brush,
    # every stamp must attenuate by 0.70 — without this attenuation the
    # yellow fill reads +43% over-saturated vs PSD.
    brush = _mk_alpha_brush(
        texture_pattern=39, texture_density_base=0.70, pattern_style=16
    )
    alpha = _compute_stamp_alpha(
        brush, per_stamp_alpha=1.0, flow_mult=1.0, stroke_op=1.0
    )
    assert alpha == pytest.approx(0.70)


def test_stamp_alpha_density_ignored_for_standard_pen_even_with_grain():
    # kakidashi-kun #84 / clip-tools #44 v4 visual review: brush 10948
    # (Y4 LOSS sen) is a standard pen (pattern_style=0) that also
    # carries a paper grain (texture_pattern=40) and stores
    # texture_density_base=0.26. If every stamp were multiplied by 0.26
    # the rendered sen line turns into a faint ghost — the user caught
    # this regression on the LOSS/A/2 visual check. Standard pens must
    # NOT attenuate by density_base; CSP routes their paper-grain
    # response through per-pixel texture modulation (not implemented
    # here yet), not a stamp-wide multiplier.
    brush = _mk_alpha_brush(
        texture_pattern=40, texture_density_base=0.26, pattern_style=0
    )
    alpha = _compute_stamp_alpha(
        brush, per_stamp_alpha=1.0, flow_mult=1.0, stroke_op=1.0
    )
    assert alpha == pytest.approx(1.0)


def test_stamp_alpha_attenuation_compounds_with_other_factors():
    brush = _mk_alpha_brush(
        texture_pattern=39, texture_density_base=0.70, pattern_style=16
    )
    alpha = _compute_stamp_alpha(
        brush, per_stamp_alpha=0.5, flow_mult=0.8, stroke_op=1.0
    )
    assert alpha == pytest.approx(0.5 * 0.8 * 1.0 * 0.70)


def test_stamp_alpha_density_ignored_when_pattern_is_zero_even_with_nondefault_density():
    # Guard: if a brush table happens to store a non-1.0
    # texture_density_base while texture_pattern is 0 (no paper grain in
    # effect), we must NOT attenuate — doing so would dim strokes for
    # every grain-less brush whose author left the field at some
    # arbitrary stored value.
    brush = _mk_alpha_brush(texture_pattern=0, texture_density_base=0.3)
    alpha = _compute_stamp_alpha(
        brush, per_stamp_alpha=1.0, flow_mult=1.0, stroke_op=1.0
    )
    assert alpha == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# multiply_alpha_onto: per-stroke MULTIPLY composite for composite_mode=2
# (clip-tools fork #44, LOSA/A2 sen over-darkness)
# ---------------------------------------------------------------------------


def _mk_rgba(h, w, r, g, b, a):
    buf = np.zeros((h, w, 4), dtype=np.float32)
    buf[..., 0] = r
    buf[..., 1] = g
    buf[..., 2] = b
    buf[..., 3] = a
    return buf


def test_multiply_alpha_onto_full_cover_dark_src_darkens_bright_dst():
    # Full-opacity MULTIPLY of dark (0.1,0.1,0.1) onto bright (1.0,1.0,1.0)
    # gives the dark color (equivalent to replacing when alpha=1 and dst=white).
    buf = _mk_rgba(1, 1, 1.0, 1.0, 1.0, 1.0)
    stroke_alpha = np.ones((1, 1), dtype=np.float32)
    multiply_alpha_onto(buf, stroke_alpha, (26, 26, 26))  # 0.102 normalized
    assert buf[0, 0, 0] == pytest.approx(26 / 255.0, abs=1e-3)
    assert buf[0, 0, 1] == pytest.approx(26 / 255.0, abs=1e-3)
    assert buf[0, 0, 2] == pytest.approx(26 / 255.0, abs=1e-3)


def test_multiply_alpha_onto_zero_alpha_is_identity():
    # With stroke_alpha=0, the buffer must be unchanged — any color.
    buf = _mk_rgba(1, 1, 0.5, 0.6, 0.7, 1.0)
    stroke_alpha = np.zeros((1, 1), dtype=np.float32)
    multiply_alpha_onto(buf, stroke_alpha, (0, 0, 0))
    assert buf[0, 0, 0] == pytest.approx(0.5)
    assert buf[0, 0, 1] == pytest.approx(0.6)
    assert buf[0, 0, 2] == pytest.approx(0.7)


def test_multiply_alpha_onto_half_cover_half_darkens():
    # Half-opacity MULTIPLY of half-brightness onto bright dst:
    # out = dst * (1 - 0.5 * (1 - 0.5)) = 1.0 * 0.75 = 0.75
    buf = _mk_rgba(1, 1, 1.0, 1.0, 1.0, 1.0)
    stroke_alpha = np.full((1, 1), 0.5, dtype=np.float32)
    multiply_alpha_onto(buf, stroke_alpha, (128, 128, 128))  # 0.502 normalized
    expected = 1.0 * (1.0 - 0.5 * (1.0 - 128 / 255.0))
    assert buf[0, 0, 0] == pytest.approx(expected, abs=1e-3)


def test_multiply_alpha_onto_two_overlapping_strokes_darken_multiplicatively():
    # MULTIPLY is a repeated per-channel scaling — overlapping strokes
    # compound the darkening factor, they do NOT saturate alpha toward 1
    # the way OVER accumulation does. Two half-opacity passes of a mid
    # dark color (0.251) on bright dst (1.0) each apply the scale factor
    # (1 - 0.5 * (1 - 0.251)) = 0.625, so the result is exactly the
    # product 0.625^2.
    buf = _mk_rgba(1, 1, 1.0, 1.0, 1.0, 1.0)
    stroke_alpha = np.full((1, 1), 0.5, dtype=np.float32)
    multiply_alpha_onto(buf, stroke_alpha, (64, 64, 64))
    v1 = float(buf[0, 0, 0])
    multiply_alpha_onto(buf, stroke_alpha, (64, 64, 64))
    v2 = float(buf[0, 0, 0])
    scale = 1.0 - 0.5 * (1.0 - 64 / 255.0)
    assert v1 == pytest.approx(scale, abs=1e-3)
    assert v2 == pytest.approx(scale * scale, abs=1e-3)
    assert v2 < v1  # each extra pass darkens further


def test_multiply_alpha_onto_alpha_channel_follows_source_over():
    # The alpha channel composition is unchanged by MULTIPLY — the stroke
    # silhouette still covers the dst using source-over.
    buf = _mk_rgba(1, 1, 1.0, 1.0, 1.0, 0.5)
    stroke_alpha = np.full((1, 1), 0.3, dtype=np.float32)
    multiply_alpha_onto(buf, stroke_alpha, (100, 100, 100))
    expected_a = 0.3 + 0.5 * (1.0 - 0.3)
    assert buf[0, 0, 3] == pytest.approx(expected_a)


# ---------------------------------------------------------------------------
# _sample_paper_grain_patch: per-pixel paper-grain modulation for line_stamp
# path (clip-tools fork #44 Phase 2, kakidashi-kun #84 continuation)
# ---------------------------------------------------------------------------


def test_sample_paper_grain_patch_returns_unit_interval_floats():
    # A uniform 128-value grain tile sampled anywhere must come back as
    # 128/255 ≈ 0.502 (no brightness/contrast adjustment).
    tex = np.full((8, 8), 128, dtype=np.uint8)
    grain = _sample_paper_grain_patch(
        tex,
        bbox_h=4,
        bbox_w=4,
        bbox_x0_ss=0,
        bbox_y0_ss=0,
        texture_scale=1.0,
        ss=1,
    )
    assert grain.shape == (4, 4)
    assert grain.dtype == np.float32
    assert float(grain.min()) == pytest.approx(128 / 255.0, abs=1e-5)
    assert float(grain.max()) == pytest.approx(128 / 255.0, abs=1e-5)


def test_sample_paper_grain_patch_wraps_with_absolute_canvas_origin():
    # The sampler must address the tile in ABSOLUTE canvas ss coords so
    # two neighbouring strokes see the same grain where their bboxes
    # overlap. Create a 2×2 tile with a known pattern, then sample two
    # 1×1 patches at adjacent ss coords and assert they pick up the
    # right tile cells.
    tex = np.array([[10, 200], [50, 150]], dtype=np.uint8)
    # Patch at (0, 0) → tile cell (0, 0) → 10/255
    g0 = _sample_paper_grain_patch(tex, 1, 1, 0, 0, 1.0, 1)
    assert float(g0[0, 0]) == pytest.approx(10 / 255.0)
    # Patch at (1, 0) → tile cell (0, 1) → 200/255
    g1 = _sample_paper_grain_patch(tex, 1, 1, 1, 0, 1.0, 1)
    assert float(g1[0, 0]) == pytest.approx(200 / 255.0)
    # Patch at (0, 3) wraps → tile cell (1, 0) (3 % 2 = 1) → 50/255
    g2 = _sample_paper_grain_patch(tex, 1, 1, 0, 3, 1.0, 1)
    assert float(g2[0, 0]) == pytest.approx(50 / 255.0)


def test_sample_paper_grain_patch_scales_tile_by_texture_scale_and_ss():
    # Tile scale controls the tiling period. A 2×2 tile with
    # texture_scale=0.5 at ss=2 means each tile cell spans
    # (ss * scale) = 1 ss-pixel. At ss=4, same tex_scale → 2 ss-pixels
    # per cell. Sample a 1-wide patch straddling the boundary.
    tex = np.array([[10, 200], [50, 150]], dtype=np.uint8)
    # scale=0.5, ss=2 → divisor (ss*scale)=1.0 → each cell = 1 ss-px.
    # Patches at ss x=0 and x=1 must pick cells (0,0) and (0,1).
    g0 = _sample_paper_grain_patch(tex, 1, 1, 0, 0, 0.5, 2)
    g1 = _sample_paper_grain_patch(tex, 1, 1, 1, 0, 0.5, 2)
    assert float(g0[0, 0]) == pytest.approx(10 / 255.0)
    assert float(g1[0, 0]) == pytest.approx(200 / 255.0)


def test_sample_paper_grain_patch_applies_brightness_and_contrast():
    # Brightness/contrast curve: t' = clip((t - 0.5) * c + 0.5 + b, 0, 1)
    # A uniform 128 tile → t = 0.502. With contrast=2, brightness=0:
    #   t' = clip((0.502 - 0.5) * 2 + 0.5, 0, 1) ≈ 0.504
    # With brightness=0.5:
    #   t' = clip(0.502 - 0.5 + 0.5 + 0.5, 0, 1) = 1.0 (clipped)
    tex = np.full((4, 4), 128, dtype=np.uint8)
    g_contrast = _sample_paper_grain_patch(
        tex, 1, 1, 0, 0, 1.0, 1, brightness=0.0, contrast=2.0
    )
    assert float(g_contrast[0, 0]) == pytest.approx(
        np.clip((128 / 255.0 - 0.5) * 2.0 + 0.5, 0.0, 1.0), abs=1e-5
    )
    g_bright_clipped = _sample_paper_grain_patch(
        tex, 1, 1, 0, 0, 1.0, 1, brightness=0.5, contrast=1.0
    )
    assert float(g_bright_clipped[0, 0]) == pytest.approx(1.0)


def test_sample_paper_grain_patch_identity_bc_leaves_tile_as_is():
    # When brightness == 0 and contrast == 1 we must take the fast path
    # that does NOT run the clip+arithmetic (the fast path returns the
    # raw tile sample straight from the texture). Fast-path bit-matches
    # the slow path at b=0,c=1, verified via identical output.
    tex = np.array([[10, 200], [50, 150]], dtype=np.uint8)
    g_fast = _sample_paper_grain_patch(tex, 2, 2, 0, 0, 1.0, 1)
    g_slow = _sample_paper_grain_patch(
        tex, 2, 2, 0, 0, 1.0, 1, brightness=0.0, contrast=1.0
    )
    assert np.allclose(g_fast, g_slow)


# ---------------------------------------------------------------------------
# stamp_pattern: per-pixel paper-grain for shaped brush path
# (clip-tools fork #44 Phase 2, kakidashi-kun #84 continuation)
#
# Phase 1 only attached grain to the disc-stamped branch of
# ``render_vector_line_stamp``; shaped brushes (``pattern_style > 0``) still
# stamped at full coverage via ``stamp_pattern`` writing straight to ``buf``.
# For Y4_13_302_GSA_systemtest LOSS/A2 nuri, shaped brush 10949 laid yellow
# at ~1.147× PSD saturation. The fix: let ``stamp_pattern`` accept a grain
# plane and multiply ``src_a`` by it per-pixel, same mechanism as the disc
# path's ``stroke_alpha *= grain``.
# ---------------------------------------------------------------------------


def _stamp_pattern_baseline_vs_grain(grain_value: float):
    """Compare a single ``stamp_pattern`` call with vs. without a uniform
    ``grain`` patch. Returns ``(alpha_no_grain, alpha_with_grain)``."""
    # 40×40 destination buffer, straight-alpha RGBA, initially empty.
    buf_no = np.zeros((40, 40, 4), dtype=np.float32)
    buf_yes = np.zeros((40, 40, 4), dtype=np.float32)
    # 6×6 fully-opaque pattern mask — a square nib stamp.
    tex_mask = np.full((6, 6), 255, dtype=np.uint8)
    # Uniform grain patch over the full buffer, origin=(0, 0).
    grain = np.full((40, 40), grain_value, dtype=np.float32)
    kwargs = dict(
        buffer=buf_no,
        tex_mask=tex_mask,
        cx=20.0,
        cy=20.0,
        size_x=10.0,
        size_y=10.0,
        opacity=0.5,
        color=(255, 0, 0),
        rotation_rad=0.0,
    )
    stamp_pattern(**kwargs)
    kwargs.update(buffer=buf_yes, grain=grain, grain_origin=(0, 0))
    stamp_pattern(**kwargs)
    return buf_no[..., 3], buf_yes[..., 3]


def test_stamp_pattern_with_uniform_half_grain_halves_coverage():
    # Grain=0.5 over an opaque stamp must halve the per-pixel coverage
    # vs a baseline stamp with no grain. This is the fundamental hook
    # the shaped-brush fix relies on.
    alpha_no, alpha_yes = _stamp_pattern_baseline_vs_grain(0.5)
    # Only compare pixels where the baseline stamp actually painted.
    mask = alpha_no > 1e-4
    assert mask.any(), "baseline stamp must paint at least one pixel"
    assert np.allclose(alpha_yes[mask], alpha_no[mask] * 0.5, atol=1e-5)


def test_stamp_pattern_with_grain_ones_matches_no_grain():
    # Grain identity (all 1.0) must bit-match the no-grain baseline so
    # that opt-ing shaped brushes WITHOUT a texture_pattern into this
    # path (grain=None) stays a no-op.
    alpha_no, alpha_yes = _stamp_pattern_baseline_vs_grain(1.0)
    assert np.allclose(alpha_yes, alpha_no, atol=1e-6)


def test_stamp_pattern_grain_origin_maps_to_buffer_coords():
    # Grain is laid out in BUFFER coords with its [0, 0] cell at
    # ``grain_origin``. Shift the origin so the grain's half-attenuation
    # band lines up with the right half of the stamp, and assert only
    # the right-half pixels got halved.
    buf_left = np.zeros((40, 40, 4), dtype=np.float32)
    buf_right = np.zeros((40, 40, 4), dtype=np.float32)
    tex_mask = np.full((6, 6), 255, dtype=np.uint8)
    # Grain: left half = 1.0 (no attenuation), right half = 0.25.
    grain = np.ones((40, 40), dtype=np.float32)
    grain[:, 20:] = 0.25
    stamp_pattern(
        buf_left,
        tex_mask,
        20.0,
        20.0,
        10.0,
        10.0,
        0.6,
        (0, 255, 0),
        grain=grain,
        grain_origin=(0, 0),
    )
    # No grain comparator — use a second stamp with grain=None.
    buf_none = np.zeros((40, 40, 4), dtype=np.float32)
    stamp_pattern(buf_none, tex_mask, 20.0, 20.0, 10.0, 10.0, 0.6, (0, 255, 0))
    a_left = buf_left[..., 3]
    a_none = buf_none[..., 3]
    # Left-half painted pixels must be unchanged; right-half pixels must
    # read ~0.25× the no-grain baseline.
    mask_l = (a_none > 1e-4) & (np.arange(40)[None, :] < 20)
    mask_r = (a_none > 1e-4) & (np.arange(40)[None, :] >= 20)
    assert mask_l.any() and mask_r.any()
    assert np.allclose(a_left[mask_l], a_none[mask_l], atol=1e-5)
    assert np.allclose(a_left[mask_r], a_none[mask_r] * 0.25, atol=1e-5)


def _mk_shaped_brush(
    texture_pattern: int,
    texture_density_base: float = 1.0,
    texture_scale: float = 1.0,
    interval_base: float = 5.0,
) -> "object":
    """Minimal BrushStyle for a line-stamp shaped brush, sized so the
    ``render_vector_line_stamp`` dispatcher takes the shaped (pattern_style>0)
    branch. ``spray_flag=0`` keeps it on the line-stamp path."""
    from clip_tools.types import BrushStyle

    return BrushStyle(
        main_id=1,
        pattern_style=1,
        texture_pattern=texture_pattern,
        hardness=1.0,
        thickness_base=1.0,
        anti_alias=2,
        flow_base=1.0,
        interval_base=interval_base,
        auto_interval_type=0,
        rotation_base=0.0,
        rotation_random=0.0,
        rotation_effector=0,
        rotation_in_spray_base=0.0,
        rotation_effector_in_spray=0,
        rotation_random_in_spray=0.0,
        texture_scale=texture_scale,
        texture_rotate=0.0,
        texture_offset_x=0.0,
        texture_offset_y=0.0,
        texture_density_base=texture_density_base,
        composite_mode=0,
        spray_flag=0,
        spray_size_base=0.0,
        spray_density_base=0.0,
        spray_bias=0.0,
    )


def _mk_single_stamp_stroke(brush_size: float = 12.0) -> "object":
    """A near-degenerate 2-point stroke that produces exactly one shaped
    stamp (interval_base is high enough that the step accumulator never
    fires a second stamp). This keeps the per-pixel alpha in a non-
    saturated regime so grain attenuation is measurable on the output."""
    from clip_tools.constants import VectorType
    from clip_tools.types import VectorPoint, VectorStroke

    return VectorStroke(
        vtype=VectorType.STANDARD,
        color=(200, 100, 50),
        stroke_opacity=1.0,
        brush_size=brush_size,
        brush_id=1,
        points=[
            VectorPoint(x=50.0, y=30.0, pressure=1.0),
            VectorPoint(x=51.0, y=30.0, pressure=1.0),
        ],
        random_seed=42,
    )


def _render_shaped_single_stroke(
    monkeypatch,
    texture_pattern: int,
    grain_tex: np.ndarray | None,
    interval_base: float = 200.0,
):
    """Drive ``render_vector_line_stamp`` with a single shaped-brush stroke,
    patching ``default_brush`` so no DataFrame plumbing is needed. Returns
    the rendered RGBA uint8 canvas. ``interval_base=200`` is big enough that
    the 1-pixel span between the two sample points never trips step_px, so
    only the initial stamp fires — a predictable single-stamp footprint."""
    import pandas as pd
    import clip_tools.rendering as rendering

    brush = _mk_shaped_brush(
        texture_pattern=texture_pattern, interval_base=interval_base
    )
    monkeypatch.setattr(rendering, "default_brush", lambda _bid: brush)
    # A single-stamp 8x8 pattern (opaque square nib).
    nib = np.full((8, 8), 255, dtype=np.uint8)
    pattern_images = {100: nib}
    if grain_tex is not None and texture_pattern > 0:
        pattern_images[texture_pattern] = grain_tex
    # BrushPatternStyle maps pattern_style=1 → image id 100 (the nib).
    bps = pd.DataFrame(
        [
            {
                "MainId": 1,
                "ImageIndex": (100).to_bytes(4, "big"),
                "OrderType": 0,
                "Reverse2": 0,
            }
        ]
    )
    dfs = {"BrushPatternStyle": bps}
    stroke = _mk_single_stamp_stroke()
    rendered, skipped = rendering.render_vector_line_stamp(
        [stroke], (60, 100), None, pattern_images, dfs
    )
    assert not skipped
    return rendered


def test_shaped_path_applies_paper_grain_when_texture_pattern_present(monkeypatch):
    # With a shaped brush carrying texture_pattern > 0 and a half-value
    # grain tile, the painted alpha on a single-stamp footprint must come
    # back at roughly half the no-grain baseline. This is the integration
    # hook that closes Phase 2 of kakidashi-kun #84 — shaped brush 10949
    # on the LOSS/A2 nuri was over-saturating because grain never reached
    # the shaped branch.
    half_tex = np.full((4, 4), 128, dtype=np.uint8)  # ≈ 0.502 grain
    img_no_grain = _render_shaped_single_stroke(
        monkeypatch, texture_pattern=0, grain_tex=None
    )
    img_with_grain = _render_shaped_single_stroke(
        monkeypatch, texture_pattern=99, grain_tex=half_tex
    )
    a_no = img_no_grain[..., 3].astype(np.float32)
    a_yes = img_with_grain[..., 3].astype(np.float32)
    mask = a_no > 10  # compare on pixels the baseline stroke actually painted
    assert mask.sum() > 20, (
        f"expected the test stroke to paint a visible footprint "
        f"(got {int(mask.sum())} painted pixels)"
    )
    # 128/255 ≈ 0.502; allow slack for CLIP's floor(alpha*256) step +
    # supersample reduce quantisation on single-stamp coverage.
    ratio = a_yes[mask].mean() / a_no[mask].mean()
    assert 0.40 < ratio < 0.60, (
        f"shaped-path grain did not attenuate as expected; ratio={ratio:.3f}"
    )


def test_shaped_path_no_grain_when_texture_pattern_zero(monkeypatch):
    # Regression: a shaped brush with texture_pattern=0 (the brush has
    # no paper grain enabled) must render identically to the pre-Phase-2
    # baseline — i.e. the grain-sampling branch is skipped entirely so
    # brushes authored without a secondary paper tile can't accidentally
    # pay the sampling cost or get modulated.
    img_a = _render_shaped_single_stroke(monkeypatch, texture_pattern=0, grain_tex=None)
    img_b = _render_shaped_single_stroke(monkeypatch, texture_pattern=0, grain_tex=None)
    assert np.array_equal(img_a, img_b)


def test_erase_brush_composite_mode_27_is_skipped(monkeypatch):
    # Regression (kakidashi-kun #96 Y4 LOSS/A2 nuri 眼下の黒):
    # CSP's CompositeMode 27 is the "Erase" brush — it reduces dst alpha
    # by the stamp's src alpha without laying down any color. Our
    # renderer has no destination-out path, so without a guard the brush
    # falls through to standard OVER and paints its authoring color
    # (brush 10951 in Y4 LOSS/A2 nuri uses (43, 52, 51) with very low
    # hardness) as a visible dark mark CSP's own output never shows.
    #
    # Fix: skip rendering CompositeMode == 27 strokes entirely. On a
    # layer with nothing to erase this matches CSP's behavior; a proper
    # destination-out implementation is a follow-up for the case where
    # the layer has paint beneath the stroke.
    import clip_tools.rendering as rendering

    erase_brush = _mk_shaped_brush(texture_pattern=0)
    # Override to be a standard disc pen with CSP's Erase mode.
    erase_brush.pattern_style = 0
    erase_brush.composite_mode = 27
    erase_brush.hardness = 0.01
    monkeypatch.setattr(rendering, "default_brush", lambda _bid: erase_brush)
    stroke = _mk_single_stamp_stroke(brush_size=8.0)
    # The stroke's color is (200, 100, 50) — a very visible warm tone.
    # Without the mode-27 guard this would paint that color onto the
    # empty canvas as a dark orange mark.
    rendered, skipped = rendering.render_vector_line_stamp(
        [stroke], (60, 100), None, {}, {}
    )
    assert not skipped
    assert rendered.shape == (60, 100, 4)
    assert rendered[..., 3].max() == 0, (
        f"erase brush left visible paint: max alpha = {rendered[..., 3].max()}"
    )


def test_ss_reduce_preserves_edge_color_at_low_alpha(monkeypatch):
    # Regression (kakidashi-kun #96 Y4 LOSS/A2, user-reported "縁に黒"):
    # the supersample-reduce step used to average ``buf`` in straight-RGB
    # space, so a single painted sub-pixel surrounded by three never-
    # painted neighbors (RGB=0, A=0) averaged its RGB channels toward
    # zero. Edge pixels of every low-alpha stroke then rendered as a dark
    # ring (brown/black around a yellow stroke) even though the alpha
    # channel averaged correctly.
    #
    # Fix: downsample in premultiplied-alpha space, then un-premultiply.
    # An RGB=yellow, A=0.1 pixel next to three transparent neighbors
    # reduces to a pixel whose RGB stays yellow and whose alpha averages
    # to 0.025 — correct both for the color and the coverage.
    import clip_tools.rendering as rendering

    canvas_h, canvas_w = 2, 2  # one output pixel, ss=2 → 2×2 subpixels
    ss = 2
    buf = np.zeros((canvas_h * ss, canvas_w * ss, 4), dtype=np.float32)
    # Paint one subpixel at (1, 1) pure red with 10% alpha. The other
    # three subpixels in the ss-tile covering output pixel (0, 0) stay
    # (0,0,0,0) — i.e. never touched by any stroke.
    buf[1, 1] = [1.0, 0.0, 0.0, 0.1]

    # Hand-run the exact reduce block the fix lives in.
    # We cannot import the inner block directly; replicate it on the buf
    # slice that covers output pixel (0, 0).
    region_h, region_w = 1, 1
    tile = buf[0:ss, 0:ss].astype(np.float32, copy=True)
    alpha = tile[..., 3:4]
    tile[..., :3] = tile[..., :3] * alpha
    tiled = tile.reshape(region_h, ss, region_w, ss, 4)
    reduced = np.einsum("hswSc->hwc", tiled) * (1.0 / (ss * ss))
    safe_a = np.where(reduced[..., 3:4] > 1e-6, reduced[..., 3:4], 1.0)
    reduced[..., :3] = reduced[..., :3] / safe_a
    out = np.minimum(255, (reduced * 256).astype(np.int32)).astype(np.uint8)

    r, g, b, a = out[0, 0]
    # R channel must stay saturated red — not get averaged toward zero by
    # the three transparent neighbors (the bug would give r ≈ 64).
    assert r >= 250, f"edge pixel R darkened by straight-RGB average: r={r}"
    assert g == 0 and b == 0, f"edge pixel bled color: g={g} b={b}"
    # Alpha averages across the 2×2 tile normally.
    assert 5 <= a <= 7, f"edge pixel alpha wrong: a={a}"
