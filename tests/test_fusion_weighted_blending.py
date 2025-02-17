"""Test suite for fusion worker."""

import shutil
import unittest
from pathlib import Path

import numpy as np
import zarr
from test_dataset import (
    generate_x_lin_blend_dataset,
    generate_y_lin_blend_dataset,
)
from test_dataset_real import generate_real_dataset

import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.io as io


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
            option=0, pool_size=16, worker_cells=[]
        )
        # ^^^Worker cells initalized later in factory methods
        # following fusion initalization.

        # Application Parameter: CELL_SIZE
        self.CELL_SIZE = [100, 100, 75]

        # Application Parameter: POST_REG_TFMS
        self.POST_REG_TFMS: list[geometry.Affine] = []

        # Application Object: BLENDING_MODULE
        # ^^^Initalized in factory methods
        # following fusion initalization.

    def _read_zarr_zyx_volume(self, zarr_path: str):
        output_path = zarr_path + "/0"
        arr = zarr.open(output_path, mode="r")
        fused_data = arr[0, 0, :, :, :]

        return fused_data

    def test_fusion_in_x_axis(self):
        # Generate Dataset
        ground_truth, DATASET = generate_x_lin_blend_dataset()

        # Generate Output Parameters
        zarr_path = str(Path(self.output_path) / "fused_in_x.zarr")
        OUTPUT_PARAMS = io.OutputParameters(
            path=zarr_path,
            chunksize=(1, 1, 100, 100, 100),
            resolution_zyx=(1.0, 1.0, 1.0),
            datastore=0,
        )

        # Init and Run Fusion
        worker_cells = []
        _, _, _, tile_aabbs, output_volume_size, _ = fusion.initialize_fusion(
            DATASET, self.POST_REG_TFMS, OUTPUT_PARAMS
        )
        z_cnt, y_cnt, x_cnt = fusion.get_cell_count_zyx(
            output_volume_size, self.CELL_SIZE
        )
        for z in range(z_cnt):
            for y in range(y_cnt):
                for x in range(x_cnt):
                    worker_cells.append((z, y, x))
        self.RUNTIME_PARAMS.worker_cells = worker_cells

        # Set Blending parameters here
        self.BLENDING_MODULE = blend.WeightedLinearBlending(
            tile_aabbs=tile_aabbs
        )

        fusion.run_fusion(
            DATASET,
            OUTPUT_PARAMS,
            self.RUNTIME_PARAMS,
            self.CELL_SIZE,
            self.POST_REG_TFMS,
            self.BLENDING_MODULE,
        )

        # Read output and compare with ground truth
        fused_data = self._read_zarr_zyx_volume(OUTPUT_PARAMS.path)
        self.assertTrue(np.all(np.abs(fused_data - ground_truth) < 2))

    def test_fusion_in_y_axis(self):
        # Generate Dataset
        ground_truth, DATASET = generate_y_lin_blend_dataset()

        # Generate Output Parameters
        zarr_path = str(Path(self.output_path) / "fused_in_y.zarr")
        OUTPUT_PARAMS = io.OutputParameters(
            path=zarr_path,
            chunksize=(1, 1, 100, 100, 100),
            resolution_zyx=(1.0, 1.0, 1.0),
            datastore=0,
        )

        # Init and Run Fusion
        worker_cells = []
        _, _, _, tile_aabbs, output_volume_size, _ = fusion.initialize_fusion(
            DATASET, self.POST_REG_TFMS, OUTPUT_PARAMS
        )
        z_cnt, y_cnt, x_cnt = fusion.get_cell_count_zyx(
            output_volume_size, self.CELL_SIZE
        )
        for z in range(z_cnt):
            for y in range(y_cnt):
                for x in range(x_cnt):
                    worker_cells.append((z, y, x))
        self.RUNTIME_PARAMS.worker_cells = worker_cells

        self.BLENDING_MODULE = blend.WeightedLinearBlending(
            tile_aabbs=tile_aabbs
        )

        fusion.run_fusion(
            DATASET,
            OUTPUT_PARAMS,
            self.RUNTIME_PARAMS,
            self.CELL_SIZE,
            self.POST_REG_TFMS,
            self.BLENDING_MODULE,
        )

        # Read output and compare with ground truth
        fused_data = self._read_zarr_zyx_volume(OUTPUT_PARAMS.path)
        self.assertTrue(np.all(np.abs(fused_data - ground_truth) < 2))

    def test_fusion_on_real_data(self):
        # Generate Dataset
        ground_truth, DATASET = generate_real_dataset()

        zarr_path = str(Path(self.output_path) / "fused_real_data.zarr")

        OUTPUT_PARAMS = io.OutputParameters(
            path=zarr_path,
            chunksize=(1, 1, 100, 100, 100),
            resolution_zyx=(
                1.0,
                0.256,
                0.256,
            ),  # Preserving original resolution.
            datastore=0,
        )
        RUNTIME_PARAMS = io.RuntimeParameters(
            option=0, pool_size=16, worker_cells=[]
        )
        POST_REG_TFMS = []
        CELL_SIZE = [100, 100, 100]

        # Init and Run Fusion
        worker_cells = []
        _, _, _, tile_aabbs, output_volume_size, _ = fusion.initialize_fusion(
            DATASET, POST_REG_TFMS, OUTPUT_PARAMS
        )
        z_cnt, y_cnt, x_cnt = fusion.get_cell_count_zyx(
            output_volume_size, CELL_SIZE
        )
        for z in range(z_cnt):
            for y in range(y_cnt):
                for x in range(x_cnt):
                    worker_cells.append((z, y, x))
        RUNTIME_PARAMS.worker_cells = worker_cells

        BLENDING_MODULE = blend.WeightedLinearBlending(tile_aabbs=tile_aabbs)

        fusion.run_fusion(
            DATASET,
            OUTPUT_PARAMS,
            RUNTIME_PARAMS,
            CELL_SIZE,
            POST_REG_TFMS,
            BLENDING_MODULE,
        )

        # Read output and compare with ground truth
        fused_data = self._read_zarr_zyx_volume(OUTPUT_PARAMS.path)
        self.assertTrue(np.all(np.abs(fused_data - ground_truth) < 2))

    def tearDown(self):
        # Delete test volumes.
        shutil.rmtree(str(self.output_path))


if __name__ == "__main__":
    # unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFusion))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
