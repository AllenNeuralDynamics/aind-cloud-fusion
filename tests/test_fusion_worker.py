"""Test suite for fusion worker."""
import shutil
import unittest
from pathlib import Path

import numpy as np
import zarr

import aind_cloud_fusion.io as io
import aind_cloud_fusion.runtime as runtime
from tests.test_dataset import (
    generate_x_dataset,
    generate_y_dataset,
    generate_z_dataset,
)


# NOTE:
# For convenient automated testing,
# TestWorker operates on synthetic data
# and outputs test outputs locally.
class TestWorker(unittest.TestCase):
    def setUp(self):
        self.config_yaml_path = "tests/test_worker_config.yml"

        params = io.read_config_yaml(self.config_yaml_path)
        self.output_path = Path(params["output"]["path"])
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _read_zarr_zyx_volume(self, zarr_path: str):
        output_path = self.node.OUTPUT_PARAMS.path
        output_path = zarr_path + "/0"
        arr = zarr.open(output_path, mode="r")
        fused_data = arr[0, 0, :, :, :]

        return fused_data

    def test_fusion_in_z_axis(self):
        # Generate Dataset
        ground_truth, dataset = generate_z_dataset()

        # Initalize Runtime, Adjust output path
        self.node = runtime.Worker(self.config_yaml_path, dataset)
        self.node.OUTPUT_PARAMS.path = str(
            self.output_path / "fused_in_z.zarr"
        )

        # Run Fusion
        self.node.run()

        # Read output and compare with ground truth
        fused_data = self._read_zarr_zyx_volume(self.node.OUTPUT_PARAMS.path)
        self.assertTrue(np.allclose(fused_data, ground_truth))

    def test_fusion_in_y_axis(self):
        ground_truth, dataset = generate_y_dataset()

        # Initalize Runtime, Adjust output path
        self.node = runtime.Worker(self.config_yaml_path, dataset)
        self.node.OUTPUT_PARAMS.path = str(
            self.output_path / "fused_in_y.zarr"
        )

        # Run Fusion
        self.node.run()

        # Read output and compare with ground truth
        fused_data = self._read_zarr_zyx_volume(self.node.OUTPUT_PARAMS.path)
        self.assertTrue(np.allclose(fused_data, ground_truth))

    def test_fusion_in_x_axis(self):
        ground_truth, dataset = generate_x_dataset()

        # Initalize Runtime, Adjust output path
        self.node = runtime.Worker(self.config_yaml_path, dataset)
        self.node.OUTPUT_PARAMS.path = str(
            self.output_path / "fused_in_x.zarr"
        )

        # Run Fusion
        self.node.run()

        # Read output and compare with ground truth
        fused_data = self._read_zarr_zyx_volume(self.node.OUTPUT_PARAMS.path)
        self.assertTrue(np.allclose(fused_data, ground_truth))

    def tearDown(self):
        # Delete test volumes.
        shutil.rmtree(str(self.output_path))


if __name__ == "__main__":
    # unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestWorker))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
