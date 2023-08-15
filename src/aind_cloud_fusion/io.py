"""
Defines Standard Input to Fusion Algorithm.
"""

from dataclasses import dataclass

from aind_cloud_fusion import geometry

@dataclass
class Dataset:
    """
    Parameters
        tile_volumes: list of tile references
        affine_transforms: corresponding list of affine transforms
        resolution: um/pixel
    """
    tile_volumes: list[str]   # FIXME
    affine_transforms: list[geometry.Matrix]
    resolution: float

def load_dataset_from_big_stitcher(cloud_location: str, 
                                   xml_file: str) -> Dataset:
    """
    Parameters
        cloud_location: cloud path
        xml_file: xml file name
    """
    return Dataset(...)