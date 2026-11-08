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
