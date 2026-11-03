import logging
import os
from typing import Iterable, Optional

from clip_tools.api.clip_layer import ClipLayer
from clip_tools.constants import DEBUG
from clip_tools.io import load_sqlite, split_clip_binary
from clip_tools.processing import (
    augment_layer_df,
    build_external_id_map,
    dump_dfs_csv,
    process_clip_data,
    rasterize_vectors,
)
from clip_tools.sqlite_records import CanvasRecord
from clip_tools.structs import process_chunk_binary

logger = logging.getLogger(__name__)


class ClipImage(ClipLayer):
    @classmethod
    def open(
        cls,
        path: str,
        touched_layer_ids: Optional[Iterable[int]] = None,
    ) -> ClipLayer:
        """Open a .clip file and return the root :class:`ClipImage`.

        :param touched_layer_ids: Optional set of layer MainIds that will
            actually participate in composite. When given, layers whose id
            is not in the set skip the per-layer raster buffer allocation
            inside :func:`process_clip_data`: ``_leaf_raster`` returns
            ``None`` for them during composite, which renders transparent
            (the compositor's existing safe fallback).

            ``None`` (the default) preserves pre-ADR-0016 behaviour: every
            layer gets its raster built, byte-identical to prior output.

            Consumers that want this saving typically do a cheap first
            :meth:`open_metadata` pass to walk the layer tree and compute
            the touched set, then call ``open(path, touched_layer_ids=...)``
            for the real load.
        """
        base_name = os.path.splitext(os.path.basename(path))[0]

        with open(path, "rb") as f:
            clip_str = f.read()

        clip_binary, sqlite_binary = split_clip_binary(clip_str)
        dfs = load_sqlite(sqlite_binary)
        logger.debug("Loaded CLIP SQLite database")

        if DEBUG:
            dump_dfs_csv(dfs, base_name)

        external_id_map = build_external_id_map(dfs)
        canvas = CanvasRecord.from_row(dfs["Canvas"].iloc[0])
        canvas_size = (int(canvas.canvas_height), int(canvas.canvas_width))
        brush_style = dfs.get("BrushStyle")

        clip_data, _num_skipped_chunks, _file_header, _block_metadata = (
            process_chunk_binary(clip_binary, canvas_size, external_id_map, brush_style)
        )

        for key, value in external_id_map.items():
            if not value.found:
                logger.debug(
                    f"External ID: {key} not found in {value.table_name} {value.column_name}"
                )

        # Vectors are now stored as raw bytes in clip_data; rasterize them now
        # that brush patterns (also in clip_data) are available.
        rasterize_vectors(clip_data, dfs, canvas_size)

        logger.debug("Processed CLIP binary data")

        record = augment_layer_df(dfs["Layer"])
        raster, _auxillary_list = process_clip_data(
            base_name,
            clip_data,
            dfs,
            record,
            external_id_map,
            touched_layer_ids=touched_layer_ids,
        )

        return cls(record, 0, raster, canvas_size)

    @classmethod
    def open_metadata(cls, path: str) -> ClipLayer:
        """Open a .clip file and return the layer tree WITHOUT loading rasters.

        Fast first-pass for discovering layer structure (``find`` /
        ``descendants`` / ``find_deepest_same_named`` / etc.) when the
        caller plans to compute a touched layer id set and re-open with
        :meth:`open(path, touched_layer_ids=...)`.

        Internally this skips :func:`process_chunk_binary`,
        :func:`rasterize_vectors` and :func:`process_clip_data`: only the
        SQLite side (``dfs`` + ``augment_layer_df``) is parsed. ``_raster``
        is an empty dict, so ``composite()`` on the returned tree would
        yield fully-transparent output — the metadata-only tree is for
        traversal, not rendering.

        See ADR 0016 for the 2-pass open contract.
        """
        base_name = os.path.splitext(os.path.basename(path))[0]

        with open(path, "rb") as f:
            clip_str = f.read()

        _clip_binary, sqlite_binary = split_clip_binary(clip_str)
        # Drop the raw bytes right after splitting — the metadata-only path
        # never touches the clip_binary half, so there's no reason to keep
        # it alive for the lifetime of this method.
        del clip_str
        dfs = load_sqlite(sqlite_binary)
        logger.debug("Loaded CLIP SQLite database (metadata-only open)")

        if DEBUG:
            dump_dfs_csv(dfs, base_name)

        canvas = CanvasRecord.from_row(dfs["Canvas"].iloc[0])
        canvas_size = (int(canvas.canvas_height), int(canvas.canvas_width))

        record = augment_layer_df(dfs["Layer"])
        return cls(record, 0, {}, canvas_size)
