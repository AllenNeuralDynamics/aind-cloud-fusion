"""Example test template."""

import numpy as np
import os
import torch
from pathlib import Path
from PIL import Image
import shutil
import zarr

import unittest

import aind_cloud_fusion.io as io
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.runtime as runtime

class TestDataset(io.Dataset):
    def __init__(self, tile_1_zyx: np.ndarray, 
                       tile_2_zyx: np.ndarray, 
                       known_transform_zyx: geometry.Matrix, 
                       input_resolution_zyx: tuple[float, float, float]):
        self.tile_1_zyx = tile_1_zyx
        self.tile_2_zyx = tile_2_zyx
        self.known_transform_zyx = known_transform_zyx
        self.input_resolution_zyx = input_resolution_zyx

    @property
    def tile_volumes_zyx(self) -> dict[int, io.LazyArray]:
        tile_volumes = {0: self.tile_1_zyx,
                        1: self.tile_2_zyx}
        return tile_volumes

    @property
    def tile_transforms_zyx(self) -> dict[int, list[geometry.Transform]]:
        tile_transforms = {0: [geometry.Affine(np.array([[1, 0, 0, 0], 
                                                        [0, 1, 0, 0], 
                                                        [0, 0, 1, 0]]))],
                           1: [geometry.Affine(self.known_transform_zyx)]}
        return tile_transforms

    @property
    def tile_shapes_zyx(self) -> dict[int, tuple[int, int, int]]:
        tile_shapes = {0: self.tile_1_zyx.shape,
                       1: self.tile_2_zyx.shape}
        return tile_shapes

    @property
    def tile_resolution_zyx(self) -> tuple[float, float, float]:
        return self.input_resolution_zyx

class TestSyntheticDataFusion(unittest.TestCase):
    def setUp(self):
        img = Image.open('tests/nueral_dynamics_logo.jpeg').convert('L')
        self.img = np.array(img)
        self.x, self.y = self.img.shape
        self.stack_size = 400

        self.config_yaml_path = 'tests/test_worker_config.yml'
        
        self.output_path = Path('/tmp')
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _generate_z_dataset(self):
        ground_truth = np.zeros((self.x, self.y, self.stack_size))
        for i in range(self.stack_size):
            ground_truth[:, :, i] = self.img
        tile_1_zyx = ground_truth[:3*(self.x // 4), :, :].copy()
        tile_2_zyx = ground_truth[(self.x // 4):, :, :].copy()

        # Erase some signal in the overlap region to test blending, 
        # Erasing alternating stripes. 
        s = tile_1_zyx.shape[0]   # Split axis. 
        tile_1_zyx[(s//2):, 0::2, :] = 0   # Stripes in bottom half
        tile_2_zyx[0:(s//2), 1::2, :] = 0  # Stripes in upper half

        # Registration matrix is (identity, translation = to tile cut). 
        registration_zyx = np.array([[1, 0, 0, (self.x // 4)],   # Split axis 
                                    [0, 1, 0, 0],  
                                    [0, 0, 1, 0]])

        input_resolution_zyx = (1.0, 1.0, 1.0)

        dataset = TestDataset(tile_1_zyx, 
                              tile_2_zyx, 
                              registration_zyx,
                              input_resolution_zyx)
        
        return ground_truth, dataset

    def _generate_y_dataset(self):
        ground_truth = np.zeros((self.stack_size, self.x, self.y))
        for i in range(self.stack_size):
            ground_truth[i, :, :] = self.img
        tile_1_zyx = ground_truth[:, :3*(self.x // 4), :].copy()
        tile_2_zyx = ground_truth[:, (self.x // 4):, :].copy()

        # Erase some signal in the overlap region to test blending, 
        # Erasing alternating stripes. 
        s = tile_1_zyx.shape[1]   # Split axis. 
        tile_1_zyx[:, (s//2):, 0::2] = 0   # Stripes in bottom half
        tile_2_zyx[:, 0:(s//2), 1::2] = 0  # Stripes in upper half

        # Registration matrix is (identity, translation = to tile cut). 
        registration_zyx = np.array([[1, 0, 0, 0], 
                                    [0, 1, 0, (self.x // 4)],  # Split axis
                                    [0, 0, 1, 0]])

        input_resolution_zyx = (1.0, 1.0, 1.0)

        dataset = TestDataset(tile_1_zyx, 
                              tile_2_zyx, 
                              registration_zyx,
                              input_resolution_zyx)
        
        return ground_truth, dataset

    def _generate_x_dataset(self):
        ground_truth = np.zeros((self.x, self.stack_size, self.y))

        for i in range(self.stack_size):
            ground_truth[:, i, :] = self.img
        tile_1_zyx = ground_truth[:, :, :3*(self.x // 4)].copy()
        tile_2_zyx = ground_truth[:, :, (self.x // 4):].copy()

        # Erase some signal in the overlap region to test blending, 
        # Erasing alternating stripes. 
        s = tile_1_zyx.shape[2]   # Split axis. 
        tile_1_zyx[0::2, :, (s//2):] = 0   # Stripes in right half
        tile_2_zyx[1::2, :, 0:(s//2)] = 0  # Stripes in left half

        # Registration matrix is (identity, translation = to tile cut). 
        registration_zyx = np.array([[1, 0, 0, 0], 
                                    [0, 1, 0, 0],
                                    [0, 0, 1, (self.x // 4)]])  # Split Axis

        input_resolution_zyx = (1.0, 1.0, 1.0)

        dataset = TestDataset(tile_1_zyx, 
                              tile_2_zyx, 
                              registration_zyx,
                              input_resolution_zyx)
        
        return ground_truth, dataset

    def _read_zarr_zyx_volume(self, zarr_path: str):
        output_path = self.node.OUTPUT_PARAMS.path
        output_path = zarr_path + '/0'
        arr = zarr.open(output_path, mode='r')
        fused_data = arr[0, 0, :, :, :]

        return fused_data

    def test_fusion_in_z_axis(self): 
        # Generate Dataset
        ground_truth, dataset = self._generate_z_dataset()

        # Initalize Runtime, Adjust output path
        self.node = runtime.Worker(self.config_yaml_path, dataset)
        self.node.OUTPUT_PARAMS.path = str(self.output_path / 'fused_in_z.zarr')

        # Run Fusion        
        self.node.run()

        # Read output and compare with ground truth
        fused_data = self._read_zarr_zyx_volume(self.node.OUTPUT_PARAMS.path)
        self.assertTrue(np.allclose(fused_data, ground_truth))

    def test_fusion_in_y_axis(self): 
        ground_truth, dataset = self._generate_y_dataset()

        # Initalize Runtime, Adjust output path
        self.node = runtime.Worker(self.config_yaml_path, dataset)
        self.node.OUTPUT_PARAMS.path = str(self.output_path / 'fused_in_y.zarr')

        # Run Fusion        
        self.node.run()

        # Read output and compare with ground truth
        fused_data = self._read_zarr_zyx_volume(self.node.OUTPUT_PARAMS.path)
        self.assertTrue(np.allclose(fused_data, ground_truth))

    def test_fusion_in_x_axis(self): 
        ground_truth, dataset = self._generate_x_dataset()
        
        # Initalize Runtime, Adjust output path
        self.node = runtime.Worker(self.config_yaml_path, dataset)
        self.node.OUTPUT_PARAMS.path = str(self.output_path / 'fused_in_x.zarr')

        # Run Fusion        
        self.node.run()

        # Read output and compare with ground truth
        fused_data = self._read_zarr_zyx_volume(self.node.OUTPUT_PARAMS.path)
        self.assertTrue(np.allclose(fused_data, ground_truth))

    # TEMPORARY COMMENT OUT TO SEE OUTPUTS
    # def tearDown(self):
    #     # Delete test volumes.
    #     shutil.rmtree(str(self.output_path))

if __name__ == "__main__":
    # unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSyntheticDataFusion))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)