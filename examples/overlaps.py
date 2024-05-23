"""
Retrives overlap regions from config file generated
from dispim or exaspim scheduler.
"""

import glob
import os
import time
import uuid
from pathlib import Path

import yaml

import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.io as io


# (Run on the primary 405 channel)
def get_tile_overlaps(xml_path: str, s3_path: str):
    # Initialize Application Objects
    # required for fusion initalization.

    # Application Object: DATASET
    xml_path = str(Path(xml_path))
    s3_path = input_path
    DATASET = io.BigStitcherDataset(xml_path, s3_path, datastore=0)

    # Application Object: OUTPUT_PARAMS
    OUTPUT_PARAMS = io.OutputParameters(
        path="",
        chunksize=(1, 1, 128, 128, 128),
        resolution_zyx=DATASET.tile_resolution_zyx,
        datastore=0,
    )

    # Application Parameter: POST_REG_TFMS
    POST_REG_TFMS: list[geometry.Affine] = []

    # Get Tile AABB's
    _, _, _, tile_aabbs, _, _ = fusion.initialize_fusion(
        DATASET, POST_REG_TFMS, OUTPUT_PARAMS
    )

    # Get Tile Layout
    tile_layout = blend.parse_yx_tile_layout(xml_path)

    # Finally, get overlap regions
    blend.get_overlap_regions(tile_layout, tile_aabbs)

    return tile_to_overlap_ids, overlaps


if __name__ == "__main__":
    xml_path = "Your xml path here"
    s3_path = "Your s3 path here"
    get_tile_overlaps(xml_path, s3_path)
