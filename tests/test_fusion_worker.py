"""Example test template."""

import numpy as np
import os
import torch
from pathlib import Path
from PIL import Image
import zarr

import unittest

import aind_cloud_fusion.io as io
import aind_cloud_fusion.runtime as runtime

class TestWorkerNode(runtime.Worker):
    def __init__(self, config_yaml, test_dataset: TestDataset):
        super.__init__(config_yaml)
        self.DATASET = test_dataset


class TestDataset(io.Dataset):
    def __init__(self, tile_1_zyx: np.ndarray, 
                       tile_2_zyx: np.ndarray, 
                       known_transform_zyx: geometry.Matrix, 
                       input_resolution_zyx: tuple[float, float, float]):
        self.tile_1_zyx = tile_1_zyx
        self.tile_2_zyx = tile_2_zyx
        self.known_transform_zyx = known_transform_zyx
        self.input_resolution_zyx = input_resolution_zyx

    def tile_volumes_zyx(self) -> dict[int, io.LazyArray]:
        tile_volumes = {0: self.tile_1_zyx,
                        1: self.tile_2_zyx}
        return tile_volumes

    def tile_transforms_zyx(self) -> dict[int, geometry.Transform]:
        tile_transforms = {0: geometry.Affine(np.array([[1, 0, 0, 0], 
                                                        [0, 1, 0, 0], 
                                                        [0, 0, 1, 0]])),
                           1: geometry.Affine(self.known_transform_zyx)}
        return tile_transforms

    def tile_shapes_zyx(self) -> dict[int, tuple[int, int, int]]:
        tile_shapes = {0: self.tile_1_zyx.shape,
                       1: self.tile_2_zyx.shape}
        return tile_shapes

    def tile_resolution_zyx(self) -> tuple[float, float, float]:        
        return self.input_resolution_zyx


class TestSyntheticDataFusion(unittest.TestCase):
    def __init__(self):
        img = Image.open('tests/nueral_dynamics_logo.jpeg').convert('L')
        self.img = np.array(img)
        self.x, self.y = self.img.shape
        self.stack_size = 400

        config_yaml_path = 'tests/test_worker_config.yml'
        self.node = TestWorkerNode(config_yaml_path, dataset)

        self.output_path = Path('/tmp')
        self.output_path.mkdir(parents=True, exist_ok=True)


    def test_fusion_in_z_axis(self): 
        # Define mock TestDataset, read test config yaml, initalize worker runtime:
        # Define overlapping tiles from image stack
        ground_truth = np.zeros(self.stack_size, self.x, self.y)
        for i in range(self.stack_size):
            ground_truth[i, :, :] = self.img
        tile_1_zyx = ground_truth[0:3*(self.stack_size / 4), :, :]
        tile_2_zyx = ground_truth[(self.stack_size / 4):, :, :]
        
        # Erase some signal in the overlap region to test blending, 
        # Erasing alternating stripes. 
        s = tile_1_zyx.shape[0]
        tile_1_zyx[(s/2):, 0::2, :] = 0
        tile_2_zyx[0:(s/2), 1::2, :] = 0

        # Registration matrix is (identity, translation = to tile cut). 
        registration_zyx = np.array([[1, 0, 0, (self.stack_size / 4)], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 0, 0]])
        
        input_resolution_zyx = (1.0, 1.0, 1.0)
        output_resolution_zyx = (1.0, 1.0, 1.0)

        dataset = TestDataset(tile_1_zyx, 
                              tile_2_zyx, 
                              registration_zyx,
                              input_resolution_zyx)

        # Adjust output path
        self.node.OUTPUT_PARAMS.path = str(self.output_path / 'fused_in_z.zarr')

        # Run Fusion        
        self.node.run()

        # Read output and compare with ground truth
        output_path = self.node.OUTPUT_PARAMS.path
        arr = zarr.open(output_path, mode='r')
        fused_data = arr[0, 0, :, :, :]

        self.assertTrue(np.allclose(fused_data, ground_truth))

    def test_fusion_in_y_axis(self): 
        ground_truth = np.zeros(self.x, self.stack_size, self.y)
        for i in range(self.stack_size):
            ground_truth[:, i, :] = self.img

        tile_1 = ground_truth[:, 0:3*(self.stack_size / 4), :]
        tile_2 = ground_truth[:, (self.stack_size / 4):, :]
        registration_zyx = np.array([[1, 0, 0, 0], 
                                     [0, 1, 0, (self.stack_size / 4)], 
                                     [0, 0, 1, 0]])
        
        input_resolution_zyx = (1.0, 1.0, 1.0)
        output_resolution_zyx = (1.0, 1.0, 1.0)

        dataset = TestDataset(tile_1_zyx, 
                              tile_2_zyx, 
                              registration_zyx,
                              input_resolution_zyx)
        config_yaml = io.read_config_yaml('tests/test_worker_config.yml')
        node = TestWorkerNode(config_yaml, dataset)

        # Adjust output path
        self.node.OUTPUT_PARAMS.path = str(self.output_path / 'fused_in_y.zarr')

        # Run Fusion        
        self.node.run()

        # Read output and compare with ground truth
        output_path = self.node.OUTPUT_PARAMS.path
        arr = zarr.open(output_path, mode='r')
        fused_data = arr[0, 0, :, :, :]

        self.assertTrue(np.allclose(fused_data, ground_truth))

    def test_fusion_in_x_axis(self): 
        ground_truth = np.zeros(self.x, self.y, self.stack_size)
        for i in range(self.stack_size):
            ground_truth[:, :, i] = self.img

        tile_1 = ground_truth[:, :, 0:3*(self.stack_size / 4)]
        tile_2 = ground_truth[:, :, (self.stack_size / 4):]
        registration_zyx = np.array([[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 0, (self.stack_size / 4)]])

        input_resolution_zyx = (1.0, 1.0, 1.0)
        output_resolution_zyx = (1.0, 1.0, 1.0)

        dataset = TestDataset(tile_1_zyx, 
                              tile_2_zyx, 
                              registration_zyx,
                              input_resolution_zyx)
        
        # Adjust output path
        self.node.OUTPUT_PARAMS.path = str(self.output_path / 'fused_in_x.zarr')

        # Run Fusion        
        self.node.run()

        # Read output and compare with ground truth
        output_path = self.node.OUTPUT_PARAMS.path
        arr = zarr.open(output_path, mode='r')
        fused_data = arr[0, 0, :, :, :]

        self.assertTrue(np.allclose(fused_data, ground_truth))

    def tearDown(self):
        # Delete test volumes.
        shutil.rmtree(str(self.output_path))

if __name__ == "__main__":
    unittest.main()