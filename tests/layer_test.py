from typing import List, Tuple

import numpy as np

from psd_tools import PSDImage

from clip_tools import ClipImage, ClipLayer


def _walk(layers, depth: int = 0) -> List[Tuple[int, ClipLayer]]:
    out: List[Tuple[int, ClipLayer]] = []
    for layer in layers:
        out.append((depth, layer))
        if layer.is_group():
            out.extend(_walk(layer, depth + 1))
    return out


class TestTest000:
    def test_canvas_size(self, test000_clip: ClipImage):
        assert test000_clip._canvas_size == (700, 1400)

    def test_top_level_layer_count(self, test000_clip: ClipImage):
        assert len(test000_clip) == 3

    def test_total_layer_count(self, test000_clip: ClipImage):
        assert len(_walk(test000_clip)) == 3

    def test_layer_names(self, test000_clip: ClipImage):
        assert [l.name for l in test000_clip] == ["用紙", "layer1", "layer2"]

    def test_layer_ids(self, test000_clip: ClipImage):
        assert [int(l.layer_id) for l in test000_clip] == [4, 3, 5]

    def test_layer_types(self, test000_clip: ClipImage):
        assert [l.layer_type for l in test000_clip] == ["raster"] * 3

    def test_no_groups(self, test000_clip: ClipImage):
        assert all(not l.is_group() for l in test000_clip)

    def test_all_visible(self, test000_clip: ClipImage):
        assert all(bool(l.visible) for l in test000_clip)

    def test_width_height(self, test000_clip: ClipImage):
        # test000 canvas is stored as _canvas_size == (height=700, width=1400).
        # psd-tools parity: expose width / height as ints matching the PIL
        # convention so downstream code can swap PSDImage ↔ ClipImage.
        assert test000_clip.width == 1400
        assert test000_clip.height == 700
        assert isinstance(test000_clip.width, int)
        assert isinstance(test000_clip.height, int)

    def test_width_height_on_descendants(self, test000_clip: ClipImage):
        # Descendants inherit the canvas width/height for now. Per-layer
        # bbox-derived size is future work tied to ClipLayer.bbox returning
        # proper right/bottom coordinates (currently (x, y, 0, 0)).
        for layer in test000_clip:
            assert layer.width == 1400
            assert layer.height == 700

    def test_size_tuple_order(self, test000_clip: ClipImage):
        # psd-tools / PIL contract: .size == (width, height). The internal
        # _canvas_size is (height, width) numpy-style, so this test guards
        # against a regression where .size silently flips to (h, w) and
        # produces transposed outputs downstream.
        assert test000_clip.size == (1400, 700)
        assert test000_clip.size == (test000_clip.width, test000_clip.height)
        # width-first ordering: index 0 is width, index 1 is height.
        assert test000_clip.size[0] == test000_clip.width
        assert test000_clip.size[1] == test000_clip.height

    def test_size_tuple_order_on_descendants(self, test000_clip: ClipImage):
        # Same contract on every layer.
        for layer in test000_clip:
            assert layer.size == (layer.width, layer.height)

    def test_descendants_flat_tree_matches_iter(self, test000_clip: ClipImage):
        # test000 has 3 top-level pixel layers and no groups, so
        # descendants() must yield the same layers as plain iteration.
        # Deeper nesting is covered by test_descendants_with_fake_group.
        assert list(test000_clip.descendants()) == list(test000_clip)

    def test_descendants_include_clip_flag_is_noop(self, test000_clip: ClipImage):
        # CSP's layer model doesn't have the psd-tools "clipping" concept,
        # so include_clip is accepted-and-ignored for signature parity.
        # Both values must yield identical sequences.
        assert list(test000_clip.descendants(include_clip=True)) == list(
            test000_clip.descendants(include_clip=False)
        )

    def test_descendants_does_not_yield_self(self, test000_clip: ClipImage):
        # Matches psd-tools semantics: descendants() iterates descendants,
        # not the receiver itself. Caller does `[root, *root.descendants()]`
        # when it wants the full tree including root.
        assert test000_clip not in list(test000_clip.descendants())

    def test_find_returns_matching_layer(self, test000_clip: ClipImage):
        # test000 layer names: ["用紙", "layer1", "layer2"].
        found = test000_clip.find("layer1")
        assert found is not None
        assert found.name == "layer1"

    def test_find_returns_none_when_absent(self, test000_clip: ClipImage):
        assert test000_clip.find("nonexistent_layer_name") is None

    def test_composite_backward_compat_no_kwargs(
        self, test000_clip: ClipImage, test000_ground_truth: np.ndarray
    ):
        # Explicit backward-compat guard: composite() with no kwargs still
        # produces the same golden output as before the signature change.
        assert np.array_equal(np.array(test000_clip.composite()), test000_ground_truth)

    def test_composite_viewport_crops_output(self, test000_clip: ClipImage):
        # viewport=(left, top, right, bottom) crops the composited result.
        # test000 canvas is 1400x700, so a 100x50 crop should yield a 100x50 image.
        img = test000_clip.composite(viewport=(10, 20, 110, 70))
        assert img is not None
        assert img.size == (100, 50)

    def test_composite_layer_filter_all_pass_matches_default(
        self, test000_clip: ClipImage, test000_ground_truth: np.ndarray
    ):
        # layer_filter=lambda _: True is equivalent to no filter — every
        # layer still participates, output matches the golden.
        all_passed = test000_clip.composite(layer_filter=lambda layer: True)
        assert all_passed is not None
        assert np.array_equal(np.array(all_passed), test000_ground_truth)

    def test_composite_layer_filter_all_block_returns_none(
        self, test000_clip: ClipImage
    ):
        # When every candidate layer is filtered out and there's no cached
        # group raster or prepended base, there's nothing to composite —
        # returns None, matching the existing "empty children" code path.
        all_blocked = test000_clip.composite(layer_filter=lambda layer: False)
        assert all_blocked is None

    def test_composite_force_runs_without_error(self, test000_clip: ClipImage):
        # Smoke test: force=True on a flat 3-pixel tree completes successfully
        # (deeper cache-skipping assertions need a group-with-cached-raster
        # fixture; same deferral as #3's deep-nesting case).
        img = test000_clip.composite(force=True)
        assert img is not None
        assert img.size == test000_clip.size

    def test_blend_mode_returns_layer_composite_enum(self, test000_clip: ClipImage):
        from clip_tools.constants import LayerComposite

        for layer in test000_clip:
            assert isinstance(layer.blend_mode, LayerComposite)

    def test_blend_mode_default_is_normal(self, test000_clip: ClipImage):
        from clip_tools.constants import LayerComposite

        # test000 has only plain raster layers with no explicit blend mode set
        # — they should all return the default NORMAL.
        for layer in test000_clip:
            assert layer.blend_mode == LayerComposite.NORMAL

    def test_blend_mode_name_overlaps_psd_tools(self, test000_clip: ClipImage):
        # psd-tools parity: the enum member *names* overlap even though the
        # value types differ. Downstream code compares via ``.name`` or
        # ``.name == "MULTIPLY"`` and gets portable semantics.
        from psd_tools.constants import BlendMode

        for layer in test000_clip:
            assert layer.blend_mode.name in BlendMode.__members__

    def test_find_first_match_wins(self, test000_clip: ClipImage):
        # test000 has unique names so this only checks the identity of
        # the returned layer, not the "first" semantics; proper first-
        # match-wins coverage needs a fixture with duplicate names
        # (deferred — same fixture gap as descendants' deep-nesting case).
        found = test000_clip.find("用紙")
        assert found is test000_clip._children[0]

    def test_parent_returns_root_identity(self, test000_clip: ClipImage):
        # Issue #28 B-1: `layer.parent` returns the single shared instance
        # from the tree's instance map, not a freshly constructed ClipLayer.
        # Every child's parent must therefore be `test000_clip` by identity.
        for layer in test000_clip:
            assert layer.parent is test000_clip

    def test_parent_is_stable_across_calls(self, test000_clip: ClipImage):
        # Issue #28 B-1 identity contract: calling `.parent` twice returns
        # the same Python object. Previously each access built a new
        # ClipLayer and this comparison was False.
        for layer in test000_clip:
            assert layer.parent is layer.parent

    def test_root_parent_is_none(self, test000_clip: ClipImage):
        # Issue #28 B-1: the root has no parent in the instance map and
        # safely returns None. The previous `self._layer_map[parent_id]`
        # raised when parent_id (0 for the root) had no entry.
        assert test000_clip.parent is None

    def test_find_returns_tree_instance(self, test000_clip: ClipImage):
        # Issue #28 B-1: `find` walks the tree so the returned layer must
        # be the instance registered under its layer_id, i.e. identical
        # to the one found by iteration.
        found = test000_clip.find("layer1")
        assert found is not None
        by_iter = [layer for layer in test000_clip if layer.name == "layer1"][0]
        assert found is by_iter

    def test_composite_cython_matches_numpy_fallback(self, test000_clip: ClipImage):
        # Issue #28 Round 2a: the Cython fast path for _apply_source NORMAL
        # non-knockout must produce byte-identical output to the numpy
        # fallback. Monkey-patch the Cython symbol to None so the fallback
        # runs, compare against the normal (Cython-enabled) output.
        from clip_tools.composite import compositor as cm

        fast = cm._apply_source_normal_fast
        try:
            cython_output = np.array(test000_clip.composite())
            cm._apply_source_normal_fast = None
            # Force a fresh composite (bypass per-layer _composited cache).
            numpy_output = np.array(test000_clip.composite(force=True))
        finally:
            cm._apply_source_normal_fast = fast
        assert np.array_equal(cython_output, numpy_output), (
            "Cython fast path diverges from numpy fallback — the kernel's "
            "operation order must match numpy's per-array materialization "
            "bit-for-bit. Check -ffp-contract=off is honored and no FMA "
            "is being auto-emitted."
        )

    def test_visible_setter_keeps_dataframe_in_sync(self, test000_clip: ClipImage):
        # Issue #28 B-1: writes must update both the snapshot and the
        # DataFrame so round-trip save paths (and anything reading the
        # DataFrame directly) stay consistent.
        layer = test000_clip._children[1]  # "layer1"
        original = layer.visible
        try:
            layer.visible = not original
            # Snapshot reflects the new value.
            assert layer.visible is (not original)
            # DataFrame reflects the new value.
            df_value = int(layer._record.loc[layer._idx, "LayerVisibility"])
            assert bool(df_value) is (not original)
        finally:
            layer.visible = original

    def test_bboxes(self, test000_clip: ClipImage):
        assert [tuple(int(x) for x in l.bbox) for l in test000_clip] == [
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ]

    def test_composite_matches_golden(
        self, test000_clip: ClipImage, test000_ground_truth: np.ndarray
    ):
        composited = np.array(test000_clip.composite())
        assert composited.shape == test000_ground_truth.shape
        assert np.array_equal(composited, test000_ground_truth)


class TestTest001:
    def test_canvas_size(self, test001_clip: ClipImage):
        assert test001_clip._canvas_size == (1200, 1600)

    def test_top_level_layer_count(self, test001_clip: ClipImage):
        assert len(test001_clip) == 3

    def test_total_layer_count(self, test001_clip: ClipImage):
        assert len(_walk(test001_clip)) == 3

    def test_layer_names(self, test001_clip: ClipImage):
        assert [l.name for l in test001_clip] == ["Paper", "Layer 1", "Layer 2"]

    def test_layer_ids(self, test001_clip: ClipImage):
        assert [int(l.layer_id) for l in test001_clip] == [4, 3, 5]

    def test_layer_types(self, test001_clip: ClipImage):
        assert [l.layer_type for l in test001_clip] == ["raster", "raster", "vector"]

    def test_has_vector_layer(self, test001_clip: ClipImage):
        assert any(l.layer_type == "vector" for l in test001_clip)

    def test_no_groups(self, test001_clip: ClipImage):
        assert all(not l.is_group() for l in test001_clip)

    def test_all_visible(self, test001_clip: ClipImage):
        assert all(bool(l.visible) for l in test001_clip)

    def test_composite_close_to_psd_golden(
        self, test001_clip: ClipImage, test001_ground_truth: np.ndarray
    ):
        # test001.png is the PSD's composite — i.e. what CLIP Studio itself
        # renders. Our renderer now does proper spray scatter for brush 6
        # (5 of 6 strokes) plus pattern-bitmap line stamping for brush 8.
        # Per-pixel agreement with CSP isn't possible for spray strokes:
        # CSP and our LCG seeds + pattern-cycling indices produce different
        # stamp placements, so pixel-level diff is dominated by random
        # speckle even when the splatter cloud is visually faithful.
        composited = np.array(test001_clip.composite())

        # Flatten both onto white to make the metric meaningful regardless of
        # alpha-channel encoding differences.
        def on_white(im: np.ndarray) -> np.ndarray:
            alpha = im[..., 3:4].astype(np.float32) / 255
            return (im[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)

        ours = on_white(composited)
        golden = on_white(test001_ground_truth)
        diff = np.abs(ours.astype(int) - golden.astype(int))

        # Current spray baseline: mean diff ~30, ~38% pixels differ >10.
        # bad_pct is high because every spray stamp lands in a different
        # location than CSP's saved placement (LCG/seed/pattern divergence)
        # — the splatter cloud's spatial extent matches but per-pixel does
        # not. Tighten only after a perceptual / SSIM-style metric replaces
        # raw L1.
        assert diff.mean() < 50, f"mean pixel diff {diff.mean():.2f} exceeds 50"
        bad_pct = 100 * (diff.max(-1) > 10).sum() / (diff.shape[0] * diff.shape[1])
        assert bad_pct < 50, f"{bad_pct:.2f}% of pixels differ > 10 (baseline ~38%)"

    def test_psd_has_matching_layer_structure(
        self, test001_clip: ClipImage, test001_psd: PSDImage
    ):
        # The .psd was exported from CLIP Studio with layer structure preserved
        # and should have the same three top-level layers as the .clip file.
        psd_names = [l.name for l in test001_psd.descendants() if l.kind == "pixel"]
        clip_names = [l.name for l in test001_clip]
        assert psd_names == clip_names
