"""
Core Interpolation Utilities
"""
import numpy as np

from aind_cloud_fusion import geometry

class Interpolator: 
    def __call__(self, 
                 grayscale_image_volume: np.ndarray, 
                 img_vol_coordinates: list[geometry.Vertex],
                 **kwargs
                 ) -> list[geometry.Vertex]:
        """
        Parameters
            grayscale_image_volume: 3D image volume to interpolate within
            img_vol_coordinates: 3D image coordinates to attach color values to
            kwargs: extra parameters specific to implementation
            
        Returns
            Interpolated coordinate values in corresponding order to img_vol_coordinates
        """
        pass

class ScipyLinearInterpolator(Interpolator):
    def __call__(self, 
                 grayscale_image_volume: np.ndarray, 
                 img_vol_coordinates: list[geometry.Vertex],
                 **kwargs
                 ) -> list[geometry.Vertex]:
        """
        Implementation of Interpolator in Scipy.
        """
        pass