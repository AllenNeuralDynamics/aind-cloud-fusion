"""Test suite for fusion worker."""
import shutil
import unittest
from pathlib import Path

import torch
import numpy as np
import zarr

import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.io as io
import aind_cloud_fusion.geometry as geometry
from test_dataset import (
    generate_x_dataset,
    generate_y_dataset,
    generate_z_dataset,
)

class TestFusion(unittest.TestCase):
    def setUp(self):
        # Initalize Application Objects
        # Application Object: Dataset
        # Generated in each test case
        
        # Application Object: OUTPUT_PARAMS
        self.output_path = "./tmp/"
        Path(self.output_path).mkdir()
        # Generate in each test case

        # Application Object: RUNTIME PARAMS
        # Will define worker_cells at the end
        self.RUNTIME_PARAMS = io.RuntimeParameters(
            use_gpus=False,
            devices=[torch.device("cpu")],
            pool_size=16
        )

        # Application Parameter: CELL_SIZE
        self.CELL_SIZE = [100, 100, 100]

        # Application Parameter: POST_REG_TFMS
        self.POST_REG_TFMS: list[geometry.Affine] = []

        # Application Object: BLENDING_MODULE
        self.BLENDING_MODULE = blend.MaxProjection()

    def _read_zarr_zyx_volume(self, zarr_path: str):
        output_path = zarr_path + "/0"
        arr = zarr.open(output_path, mode="r")
        fused_data = arr[0, 0, :, :, :]

        return fused_data

    def test_fusion_in_z_axis(self):
        # Generate Dataset
        ground_truth, DATASET = generate_z_dataset()

        # Generate Output Parameters
        zarr_path = str(
            Path(self.OUTPUT_PARAMS.path) / "fused_in_z.zarr"
        )
        OUTPUT_PARAMS = io.OutputParameters(
            path=zarr_path,
            chunksize=(1, 1, 100, 100, 100),
            resolution_zyx=(1.0, 1.0, 1.0),
        )

        # Run Fusion
        fusion.run_fusion(DATASET, 
                          OUTPUT_PARAMS,
                          self.RUNTIME_PARAMS,
                          self.CELL_SIZE,
                          self.POST_REG_TFMS,
                          self.BLENDING_MODULE)

        # Read output and compare with ground truth
        fused_data = self._read_zarr_zyx_volume(OUTPUT_PARAMS.path)
        self.assertTrue(np.allclose(fused_data, ground_truth))

    def test_fusion_in_y_axis(self):
        # Generate Dataset
        ground_truth, DATASET = generate_y_dataset()

        # Generate Output Parameters
        zarr_path = str(
            Path(self.OUTPUT_PARAMS.path) / "fused_in_y.zarr"
        )
        OUTPUT_PARAMS = io.OutputParameters(
            path=zarr_path,
            chunksize=(1, 1, 100, 100, 100),
            resolution_zyx=(1.0, 1.0, 1.0),
        )

        # Run Fusion
        fusion.run_fusion(DATASET, 
                          OUTPUT_PARAMS,
                          self.RUNTIME_PARAMS,
                          self.CELL_SIZE,
                          self.POST_REG_TFMS,
                          self.BLENDING_MODULE)

        # Read output and compare with ground truth
        fused_data = self._read_zarr_zyx_volume(OUTPUT_PARAMS.path)
        self.assertTrue(np.allclose(fused_data, ground_truth))

    def test_fusion_in_x_axis(self):
        # Generate Dataset
        ground_truth, DATASET = generate_x_dataset()

        # Generate Output Parameters
        zarr_path = str(
            Path(self.OUTPUT_PARAMS.path) / "fused_in_x.zarr"
        )
        OUTPUT_PARAMS = io.OutputParameters(
            path=zarr_path,
            chunksize=(1, 1, 100, 100, 100),
            resolution_zyx=(1.0, 1.0, 1.0),
        )

        # Run Fusion
        fusion.run_fusion(DATASET, 
                          OUTPUT_PARAMS,
                          self.RUNTIME_PARAMS,
                          self.CELL_SIZE,
                          self.POST_REG_TFMS,
                          self.BLENDING_MODULE)

        # Read output and compare with ground truth
        fused_data = self._read_zarr_zyx_volume(OUTPUT_PARAMS.path)
        self.assertTrue(np.allclose(fused_data, ground_truth))

    def tearDown(self):
        # Delete test volumes.
        shutil.rmtree(str(self.output_path))


if __name__ == "__main__":
    # unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFusion))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
