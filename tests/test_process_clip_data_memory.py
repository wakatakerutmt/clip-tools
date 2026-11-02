"""Verify that ``process_clip_data`` releases each layer's chunk bytes
as it iterates — the behaviour added for the PAIR-MEM Phase 2 memory
work. Without the pop, every layer's decompressed block data stays live
in ``clip_data`` through the whole loop, peaking in the GB range for
large fixtures. The caller does not read ``clip_data`` after the call
returns, so dropping entries during iteration is observable only through
this test.
"""

from __future__ import annotations

from clip_tools import ClipImage
from clip_tools import processing as processing_module


def test_process_clip_data_empties_clip_data_as_it_goes(test000_clip_path, monkeypatch):
    """When ``ClipImage.open`` finishes loading, the internal ``clip_data``
    dict that was handed to ``process_clip_data`` must be empty — the
    streaming pop added for Phase 2 releases each layer's bytes as it
    goes. If someone reverts to ``for key, value in clip_data.items()``
    without popping, the dict will still hold every layer's bytes and
    this test fires.

    We observe this by wrapping ``process_clip_data`` to record a
    reference to the ``clip_data`` dict it was called with, then
    checking that dict's length after the call.
    """
    seen_dicts = []

    real_process_clip_data = processing_module.process_clip_data

    def wrapped(name, clip_data, dfs, record, external_id_map):
        seen_dicts.append(clip_data)
        return real_process_clip_data(name, clip_data, dfs, record, external_id_map)

    # Patch at BOTH the module where it's defined AND the api module that
    # imported it by name — ClipImage.open uses the latter reference.
    monkeypatch.setattr(processing_module, "process_clip_data", wrapped)
    from clip_tools.api import clip_image as clip_image_api

    monkeypatch.setattr(clip_image_api, "process_clip_data", wrapped)

    ClipImage.open(str(test000_clip_path))

    assert seen_dicts, "process_clip_data was never called via ClipImage.open"
    for d in seen_dicts:
        assert len(d) == 0, (
            f"process_clip_data must drain its clip_data arg; left {len(d)} entries. "
            "If this fails after a refactor, verify the loop still pops as it iterates."
        )


def test_process_clip_data_output_unchanged_from_open(test000_clip):
    """Belt-and-suspenders: ClipImage.open (internally calls
    process_clip_data) must still produce the test000 layer count. If
    the pop/del refactor accidentally lost a layer, this would catch it
    even without a golden image."""
    layers = list(test000_clip)
    assert len(layers) == 3, (
        f"test000 should have 3 top-level layers, got {len(layers)}"
    )
    assert [l.name for l in layers] == ["用紙", "layer1", "layer2"]
