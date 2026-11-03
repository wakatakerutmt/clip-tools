import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from clip_tools.constants import DEBUG, LayerFolderBit
from clip_tools.rendering import (  # re-exported for backwards compat
    extract_brush_pattern_images,
    rasterize_vectors,
)
from clip_tools.structs import process_layer_blocks
from clip_tools.types import ExternalIdEntry, LayerEntry, LayerRecord
from clip_tools.utils import arr_to_pil


__all__ = [
    "augment_layer_df",
    "build_external_id_map",
    "dump_dfs_csv",
    "extract_brush_pattern_images",
    "process_clip_data",
    "rasterize_vectors",
]


logger = logging.getLogger(__name__)


def build_external_id_map(
    dfs: Dict[str, pd.DataFrame],
) -> Dict[str, ExternalIdEntry]:
    """Map each external chunk id to its owning table and column."""
    external_id_map: Dict[str, ExternalIdEntry] = {}
    for _, row in dfs["ExternalTableAndColumnName"].iterrows():
        if row["TableName"] not in dfs:
            continue
        if row["ColumnName"] not in dfs[row["TableName"]].columns:
            logger.debug(
                f"Skipping missing column {row['TableName']}.{row['ColumnName']}"
            )
            continue
        external_ids = dfs[row["TableName"]][row["ColumnName"]]
        for external_id in external_ids:
            external_id_str = external_id.decode("UTF-8")
            external_id_map[external_id_str] = ExternalIdEntry(
                table_name=row["TableName"],
                column_name=row["ColumnName"],
            )
    return external_id_map


def augment_layer_df(layer_df: pd.DataFrame) -> pd.DataFrame:
    """Add ParentLayer and Prefix columns to layer_df for easier traversal."""
    layer_map = {row["MainId"]: index for index, row in layer_df.iterrows()}

    layer_df["ParentLayer"] = 0

    for _, row in layer_df.iterrows():
        if not (row["LayerFolder"] & LayerFolderBit.IS_FOLDER):
            continue
        if row["LayerFirstChildIndex"] == 0:
            continue
        layer_df.loc[layer_map[row["LayerFirstChildIndex"]], "ParentLayer"] = row[
            "MainId"
        ]
        child_layer = layer_df.loc[layer_map[row["LayerFirstChildIndex"]]]
        while child_layer["LayerNextIndex"] != 0:
            layer_df.loc[layer_map[child_layer["LayerNextIndex"]], "ParentLayer"] = row[
                "MainId"
            ]
            child_layer = layer_df.loc[layer_map[child_layer["LayerNextIndex"]]]

    layer_df["Prefix"] = [[] for _ in range(len(layer_df))]

    for idx, row in layer_df.iterrows():
        if row["ParentLayer"] == 0 or row["ParentLayer"] == layer_df.loc[0]["MainId"]:
            continue

        parent_layer = layer_df.loc[layer_map[row["ParentLayer"]]]
        layer_df.loc[idx, "Prefix"].insert(0, parent_layer["LayerName"])

        while (
            parent_layer["ParentLayer"] != 0
            and parent_layer["ParentLayer"] != layer_df.loc[0]["MainId"]
        ):
            parent_layer = layer_df.loc[layer_map[parent_layer["ParentLayer"]]]
            layer_df.loc[idx, "Prefix"].insert(0, parent_layer["LayerName"])

    return layer_df


def save_debug_layer_image(arr: np.ndarray, name: str, key: str, mode: str) -> None:
    temp_folder = f"temp/{name}/external"
    os.makedirs(temp_folder, exist_ok=True)
    if mode == "raster":
        arr_to_pil(arr).save(os.path.join(temp_folder, f"{key}.png"))
    else:
        Image.fromarray(arr, "RGBA").save(os.path.join(temp_folder, f"{key}.png"))


def process_clip_data(
    name: str,
    clip_data: Dict[str, Union[Dict[int, bytes], np.ndarray]],
    dfs: Dict[str, pd.DataFrame],
    layer_df: pd.DataFrame,
    external_id_map: Dict[str, ExternalIdEntry],
    touched_layer_ids: Optional[Iterable[int]] = None,
) -> Tuple[Dict[int, LayerEntry], List[LayerEntry]]:
    """Classify processed chunks into raster/vector layers and an auxiliary bucket.

    When ``touched_layer_ids`` is given, layers whose ``MainId`` is not in that
    set skip the per-layer raster buffer allocation: ``process_layer_blocks``
    is not called for them and no entry is added to ``raster_dict``. During
    composite, ``_leaf_raster`` returns ``None`` for those layers so they
    render transparent, which is the correct behaviour when the consumer has
    pre-determined that the layer will not participate (ADR 0016).

    ``touched_layer_ids=None`` (the default) preserves the pre-ADR-0016
    behaviour: every layer gets its raster built, byte-identical to prior
    output.
    """
    raster_dict: Dict[int, LayerEntry] = {}
    auxillary_list: List[LayerEntry] = []
    processed: set = set()

    touched_set: Optional[Set[int]] = (
        set(touched_layer_ids) if touched_layer_ids is not None else None
    )

    # Iterate by popping so each layer's decompressed block bytes are freed
    # as soon as its raster buffer has been built. Without this, every
    # layer's chunks stay live in ``clip_data`` through the whole loop,
    # peaking in the GB range for medium/large fixtures. See the
    # downstream Phase 1 memray findings for the full picture.
    for key in list(clip_data.keys()):
        value = clip_data.pop(key)
        table_name = external_id_map[key].table_name
        column_name = external_id_map[key].column_name

        logger.debug(f"Processing blocks: {key} in {table_name} {column_name}")

        if isinstance(value, dict) and value:
            offscreen = dfs["Offscreen"][
                dfs["Offscreen"]["BlockData"] == key.encode("ascii")
            ].iloc[0]

            layer_id = offscreen["LayerId"]

            # Fast-path skip (ADR 0016): if the consumer told us this layer
            # won't participate in composite, drop the chunks straight away
            # — never call process_layer_blocks for it. This is where the
            # peak RSS saving comes from: a non-touched layer never
            # allocates its full-canvas RGBA buffer.
            if (
                touched_set is not None
                and layer_id != 0
                and layer_id not in touched_set
            ):
                del value
                continue

            blocks = sorted(value.items(), key=lambda x: x[0])
            # Drop the only remaining reference to the per-layer bytes dict so
            # it can be freed; ``blocks`` already holds the bytes it needs.
            del value
            try:
                processed_layer_arr = process_layer_blocks(blocks, offscreen)
            except Exception:
                logger.error(
                    f"Error processing layer: {key} in {table_name} {column_name}"
                )
            # Raw block tuples no longer needed once the raster is built.
            del blocks

            if DEBUG:
                save_debug_layer_image(processed_layer_arr, name, key, "raster")

            if layer_id == 0:
                continue

            if layer_id not in layer_df["MainId"].values:
                logger.debug(f"Skipping invalid layer: {layer_id}")
                auxillary_list.append(
                    LayerEntry(type="invalid", image=processed_layer_arr)
                )
                continue

            layer_metadata = LayerRecord.from_row(
                layer_df[layer_df["MainId"] == layer_id].iloc[0]
            )
            layer_name = layer_metadata.layer_name
            layer_prefix = layer_metadata.prefix
            layer_folder = layer_metadata.layer_folder

            if layer_metadata.layer_clip != 0:
                logger.warning(
                    f"WARNING: Skipping clipped layer: {layer_name} in folder: {layer_prefix}"
                )
                auxillary_list.append(
                    LayerEntry(type="clipped", image=processed_layer_arr)
                )
                continue

            if layer_folder != 0:
                # Groups cache their composited raster; keep it in raster_dict as
                # a fallback for when leaf children have no stored pixel data
                # (ClipStudio sometimes flattens strokes into the group's cache).
                raster_dict[layer_id] = LayerEntry(
                    type="group", image=processed_layer_arr
                )
                auxillary_list.append(
                    LayerEntry(type="group", image=processed_layer_arr)
                )
                continue

            if offscreen["MainId"] in dfs["MipmapInfo"]["Offscreen"].values:
                mipmapinfo = dfs["MipmapInfo"][
                    dfs["MipmapInfo"]["Offscreen"] == offscreen["MainId"]
                ].iloc[0]

                if mipmapinfo["ThisScale"] != 100.0:
                    logger.debug(
                        f"Skipping mipmap: {layer_name} in folder: {layer_prefix}"
                    )
                    auxillary_list.append(
                        LayerEntry(type="mipmap", image=processed_layer_arr)
                    )
                    continue
            else:
                auxillary_list.append(
                    LayerEntry(type="other", image=processed_layer_arr)
                )
                continue

            if layer_id in processed:
                raise Exception(f"Layer {layer_id} already processed")
            processed.add(layer_id)

            logger.debug(
                f"Processing layer: {layer_name} in folder: {layer_prefix} with ID: {layer_id}"
            )
            raster_dict[layer_id] = LayerEntry(type="raster", image=processed_layer_arr)
            # (``blocks`` already deleted above right after process_layer_blocks)

        elif isinstance(value, np.ndarray):
            vector_object = dfs["VectorObjectList"][
                dfs["VectorObjectList"]["VectorData"] == key.encode("ascii")
            ].iloc[0]
            layer_id = vector_object["LayerId"]

            # Vector-layer skip path (ADR 0016): drop the already-rasterized
            # vector buffer when the consumer said this layer won't
            # participate. Mirrors the raster branch's touched check above.
            if touched_set is not None and layer_id not in touched_set:
                del value
                continue

            if DEBUG:
                save_debug_layer_image(value, name, key, "vector")

            if layer_id not in layer_df["MainId"].values:
                logger.debug(f"Skipping invalid layer: {layer_id}")
                auxillary_list.append(LayerEntry(type="invalid", image=value))
                continue

            raster_dict[layer_id] = LayerEntry(type="vector", image=value)

    return raster_dict, auxillary_list


def dump_dfs_csv(dfs: Dict[str, pd.DataFrame], base_name: str) -> None:
    """Debug helper: dump all SQLite tables to CSV under temp/<base_name>/csvs/."""
    csv_dir = f"temp/{base_name}/csvs"
    os.makedirs(csv_dir, exist_ok=True)
    for key, df in dfs.items():
        df.to_csv(f"{csv_dir}/{key}.csv", index=False)
