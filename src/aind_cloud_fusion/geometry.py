"""
Core Geometry Processing
"""

from dataclasses import dataclass
from nptyping import NDArray, Shape, Float16
import numpy as np

Vertex = NDArray(Shape["3, 1"], Float16)  # XYZ axis order
Matrix = NDArray(Shape["3, 4"], Float16)  # XYZ axis order

@dataclass
class Polygon:
    """
    Bounds a 3D area with vertices. 
    Parameters
        vertex_coordinates: list of vertices in ***specified order***
    """
    vertex_coordinates: list[Vertex]


def axis_aligned_bounding_box(polygons: list[Polygon]
                              ) -> Polygon:
    """
    Parameters
        polygons: list of polygons to be bounded by axis-aligned bounding box.
        Polygons expected in absolute coordinates.  
    
    Returns
        AABB Polygon
    """
    return Polygon(...)

def detect_collisions(main_polygon: Polygon, 
                      polygon_list: list[Polygon]
                      ) -> list[Polygon]:
    """
    Parameters
        main_polygon: central polygon to check collisions against
        polygon_list: potential polygons that collide with main_polygon
    
    Returns
        list of polygons that collide with main_poygon from the input list    
    """
    return list(Polygon(...))

def clip(polygon_1: Polygon, 
         polygon_2: Polygon
         ) -> Polygon:
    """
    Parameters 
        polygon_1, polygon_2: polygons to clip

    Returns
        Polygon representing overlapping region
    """
    return Polygon(...)

def create_mask(box_polygon: Polygon, 
                sub_polygon: Polygon, 
                sampling_rate: float
                ) -> np.ndarray: 
    """
    Parameters
        box_polygon: Boundary polygon to be discretized
        sub_polygon: Area inside of box_polygon to be masked
        sampling_rate: Defines resolution of output array
    
    Returns
        3D boolean mask representing sub_polygon inside of box_polygon
    """
    return np.ndarray


def transform_points(points: list[Vertex], 
                     transform_matrix: Matrix
                     ) -> list[Vertex]: 
    """
    Parameters
        points: points to transform
        transform_matrix: homogeneous matrix
        
    Returns
        list of transformed points
    """
    return list(Vertex)


def transform_polygon(polygon: Polygon, 
                      transform_matrix: Matrix
                      ) -> Polygon:
    """
    Parameters
        polygon: Polygon to transform
        transform_matrix: homogeneous matrix
    
    Returns
        transformed Polygon
    """
    return Polygon(...)