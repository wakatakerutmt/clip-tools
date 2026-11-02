"""Round 5 Phase 1 — Option B (touched_layer_ids) regression tests.

Verifies the two 2-pass-open API additions from ADR 0016:
- ``ClipImage.open_metadata(path)`` builds the layer tree without loading
  any per-layer raster buffers.
- ``ClipImage.open(path, touched_layer_ids=set)`` loads buffers only for
  layers whose ``MainId`` is in that set; others are absent from
  ``_raster`` and composite as transparent.

The default-``None`` path must remain byte-identical to pre-ADR-0016.
"""

from __future__ import annotations

import numpy as np

from clip_tools import ClipImage


def _collect_descendant_ids(layer) -> set[int]:
    """All descendant layer ids (not including ``layer`` itself)."""
    ids: set[int] = set()
    for child in layer:
        ids.add(child.layer_id)
        if child.is_group():
            ids |= _collect_descendant_ids(child)
    return ids


def test_open_metadata_builds_tree_without_rasters(test000_clip_path):
    """open_metadata must produce a walkable tree with no raster buffers.

    Baseline: ``_raster`` is an empty dict, iteration still works, and
    layer names / ids match a normal open.
    """
    meta = ClipImage.open_metadata(str(test000_clip_path))
    full = ClipImage.open(str(test000_clip_path))

    # Metadata tree has no rasters — _raster is empty but tree is walkable.
    assert meta._raster == {}, "open_metadata must leave _raster empty"
    assert [l.name for l in meta] == [l.name for l in full], (
        "metadata-only open must expose the same layer names as a full open"
    )
    assert [l.layer_id for l in meta] == [l.layer_id for l in full], (
        "metadata-only open must expose the same layer ids as a full open"
    )
    # Composite would be transparent on a metadata-only tree (expected).
    # The real contract (byte-identical output) is verified in
    # ``test_full_touched_set_matches_default_open``.


def test_full_touched_set_matches_default_open(test000_clip, test000_clip_path):
    """touched_layer_ids containing every layer id must be byte-identical
    to the default (``touched_layer_ids=None``) open.

    This is the bit-exact anchor from ADR 0016 D5: when the touched set
    is NOT under-approximated, output is unchanged.
    """
    all_ids: set[int] = {test000_clip.layer_id}
    all_ids |= _collect_descendant_ids(test000_clip)

    filtered = ClipImage.open(str(test000_clip_path), touched_layer_ids=all_ids)

    default_out = np.array(test000_clip.composite())
    filtered_out = np.array(filtered.composite())

    assert np.array_equal(default_out, filtered_out), (
        "Passing all layer ids in touched_layer_ids must preserve the "
        "default output byte-for-byte"
    )


def test_touched_filter_skips_untouched_layer_buffers(test000_clip, test000_clip_path):
    """A touched_layer_ids set that excludes some layers must leave those
    layers absent from ``_raster`` entirely — this is where the memory
    saving comes from (no per-layer RGBA buffer allocation).
    """
    all_ids: set[int] = {test000_clip.layer_id}
    all_ids |= _collect_descendant_ids(test000_clip)

    full_default = ClipImage.open(str(test000_clip_path))
    default_ids = set(full_default._raster.keys())
    # test000 has at least some raster entries we can thin out.
    assert default_ids, "fixture has no raster entries to test against"

    # Keep roughly half the layers; the specific choice doesn't matter —
    # we just need to see untouched ones drop out of _raster.
    keep = set(sorted(default_ids)[: max(1, len(default_ids) // 2)])
    dropped = default_ids - keep

    filtered = ClipImage.open(
        str(test000_clip_path), touched_layer_ids=keep | {full_default.layer_id}
    )
    kept_entries = set(filtered._raster.keys())

    assert kept_entries <= keep | {full_default.layer_id}, (
        "touched_layer_ids skip must not keep any id outside the set"
    )
    for dropped_id in dropped:
        assert dropped_id not in filtered._raster, (
            f"layer id {dropped_id} was excluded but still has a raster entry"
        )


def test_empty_touched_set_leaves_all_rasters_empty(test000_clip_path):
    """An empty touched set (``touched_layer_ids=set()``) must produce an
    empty ``_raster`` dict — every layer's buffer allocation is skipped.
    """
    filtered = ClipImage.open(str(test000_clip_path), touched_layer_ids=set())
    assert filtered._raster == {}, (
        f"empty touched set must drop every raster; got {len(filtered._raster)} entries"
    )
