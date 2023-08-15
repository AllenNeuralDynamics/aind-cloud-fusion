"""
Core fusion algorithm
"""

import numpy as np

from aind_cloud_fusion import io 
from aind_cloud_fusion import geometry
from aind_cloud_fusion import interpolate

class ChunkIterator:
    def __init__(self, 
                 vol_dim: tuple[int, int, int], 
                 chunk_dim: tuple[int, int, int]
                 ) -> None:
        self.vol_dim = vol_dim
        self.chunk_dim = chunk_dim

    def __iter__(self) -> tuple[int, int, int]:
        """
        Returns a chunk slice.
        """
        pass


def fuse(dataset: io.Dataset,
         chunk_size: tuple[int, int, int], 
         output_location: str
         ) -> None:
    """
    Parameters
        dataset: input dataset
        chunk_size: size of chunks
        output_location: ???
    """

    # Psuedocode
    # Initialize Output Volume
    transformed_volumes_and_affines = [(geometry.transform_polygon(geometry.Polygon(vol), aff), aff)
						   			  for vol, aff in zip(dataset.tile_volumes, dataset.affine_transforms)]
    output_vol = geometry.axis_aligned_bounding_box(dataset.tile_volumes)

    for chunk_polygon in ChunkIterator(output_vol, chunk_size): 
        # Collision Detection
        overlapping_polygons = []
        overlapping_affines = []
        overlapping_tile_index = []
        for i, transformed_volume, affine in enumerate(transformed_volumes_and_affines):
            collision = geometry.detect_collisions(chunk_polygon, [transformed_volume])
            if collision: 
                overlapping_polygons.append(collision)
                overlapping_affines.append(affine)
                overlapping_tile_index.append(i)

        overlap_pieces = []
        for o_polygon, o_affine, o_index in zip(overlapping_polygons, overlapping_affines, overlapping_tile_index): 
            # Define Overlapping Region
            clip_polygon = geometry.clip(chunk_polygon, o_polygon)
            
            # Discretize Overlapping Region
            overlap_mask = geometry.create_mask(chunk_polygon, clip_polygon, dataset.resolution)

            # Color Overlap Region,  FIXME: NOT CORRECT, SPIRIT OF ALGORITHM
            src_bounds = geometry.transform_polygon(clip_polygon, o_affine.inverse)
            src_pts = geometry.transform_points(overlap_mask.where(overlap_mask==True), o_affine.inverse)
            src_region = dataset.tile_volumes[o_index][src_bounds]			

            output_piece = np.zeros(chunk_size)
            output_piece = interpolate.Interpolator(src_region, src_pts)
            overlap_pieces.append(output_piece)

        # Combine Overlap Pieces, Maximum Projection
        output_chunk = np.max(overlap_pieces)
        output_location.write(output_chunk)