"""Defines entrypoints/contexts for fusion runtimes."""
import os
from pathlib import Path

import numpy as np
import time
import torch
import uuid

import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.io as io


def initalize_compute_node(yml_file: str):
    """
    'Factory' method.
    Initalize corresponding compute node
    to what is specified in configuration file.
    """
    params = io.read_config_yaml(yml_file)

    node: ComputeNode = None
    if "scheduler" in params["runtime"]:
        node = Scheduler(yml_file)
    else:
        node = Worker(yml_file)

    return node


class ComputeNode:
    """
    Generic Runtime Primitive.
    """

    def __init__(self, config_yaml: str):
        """
        Parses configuration file into application-specific primitives.
        """

        params = io.read_config_yaml(config_yaml)

        self.DATASET = None
        dataset_type = params["input"]["dataset_type"]
        if dataset_type == "big_stitcher":
            xml_path = str(
                Path(params["dataset_parameters"]["big_stitcher"]["xml_path"])
            )
            s3_path = params["dataset_parameters"]["big_stitcher"]["s3_path"]

            self.DATASET = io.BigStitcherDataset(xml_path, s3_path)

        output_path = ""
        if params["output"]["path"].startswith('s3'):
            output_path = params["output"]["path"]
        else: 
            output_path = str(Path(params["output"]["path"]))
        self.OUTPUT_PARAMS = io.OutputParameters(
            path=output_path,
            chunksize=tuple(params["output"]["chunksize"]),
            resolution_zyx=tuple(params["output"]["resolution_zyx"]),
        )

        chunk_size = params["output"]["chunksize"]
        self.CELL_SIZE = params["algorithm_parameters"]["cell_size"]
        assert (
            (self.CELL_SIZE[0] % chunk_size[0] == 0)
            and (self.CELL_SIZE[1] % chunk_size[1] == 0)
            and (self.CELL_SIZE[2] % chunk_size[2] == 0)
        ), f"""Cell size {self.CELL_SIZE} is not a multiple of chunksize {chunk_size}.
                Please update configuration file."""

        self.BLENDING_MODULE = None
        if (
            params["algorithm_parameters"]["blending_module"]
            == "MaxProjection"
        ):
            self.BLENDING_MODULE = blend.MaxProjection()

        self.POST_REG_TFMS: list[geometry.Affine] = []
        if (
            "post_registration_transforms_zyx"
            in params["algorithm_parameters"]
        ):
            for tfm in params["algorithm_parameters"][
                "post_registration_transforms_zyx"
            ]:
                self.POST_REG_TFMS.append(geometry.Affine(np.array(tfm)))

        DEVICES = []
        pool_size = 0
        if params["runtime"]["use_gpus"]:
            DEVICES = [
                torch.device(f"cuda:{i}")
                for i in range(torch.cuda.device_count())
            ]
            assert pool_size <= len(
                DEVICES
            ), f"""For GPU runtime, pool size must be <= number of GPUs.
                Pool Size: {pool_size}, Num GPU: {len(DEVICES)}"""
            pool_size = params["runtime"]["pool_size"]

        else:
            DEVICES = [torch.device("cpu")]
            assert (
                pool_size <= os.cpu_count()
            ), f"""For CPU runtime, pool size must be <= number of CPUs.
                Pool Size: {pool_size}, Num CPU: {os.cpu_count()}"""
            pool_size = params["runtime"]["pool_size"]

        self.RUNTIME_PARAMS = io.RuntimeParameters(
            use_gpus=params["runtime"]["use_gpus"],
            devices=DEVICES,
            pool_size=pool_size,
        )

    def run(self):
        raise NotImplementedError(
            '"Please implement in DistributedNode subclass."'
        )


class Scheduler(ComputeNode):
    """
    Defines context for a scheduler run.
    Scheduler Reads:
    - Local/Cloud tiles, depending on Dataset definition.
    - Local Configuruation File containing all
    general run configs + scheduler specific params.

    Scheduler Outputs:
    - Local Worker Configuration Files.

    """

    def __init__(self, config_yaml: str, test_dataset: io.Dataset = None):
        """
        test_dataset: Input for mock datasets used in automated testing.
        Essentially, this exists because creating a synthetic dataset
        with known transformations, etc. in numpy is easier than creating a
        synthetic dataset with BigStitcher, etc.
        """
        super().__init__(config_yaml)
        params = io.read_config_yaml(config_yaml)

        if test_dataset:
            self.DATASET = test_dataset
        self.config_yaml = config_yaml
        self.worker_yml_path = str(
            Path(params["runtime"]["scheduler"]["worker_yml_path"])
        )
        self.num_workers = params["runtime"]["scheduler"]["num_workers"]

        # Create path if does not exist
        Path(self.worker_yml_path).mkdir(parents=True, exist_ok=True)

    def run(self):
        """
        Outputs worker configuration files into specifed path.
        """

        # Get Output Volume Size from fusion
        _, _, _, _, output_volume_size, _ = fusion.initialize_fusion(
            self.DATASET, self.POST_REG_TFMS, self.OUTPUT_PARAMS
        )

        # Define/Divide Work, Generate YAML files.
        z_cnt, y_cnt, x_cnt = fusion.get_cell_count_zyx(
            output_volume_size, self.CELL_SIZE
        )
        total_cells = z_cnt * y_cnt * x_cnt
        cell_per_worker = total_cells // self.num_workers

        # Prep base (copied) yml
        params = io.read_config_yaml(self.config_yaml)
        del params["runtime"]["scheduler"]
        params["runtime"]["worker"] = {}
        curr_worker_cells = []
        worker_num = 0

        for z in range(z_cnt):
            for y in range(y_cnt):
                for x in range(x_cnt):
                    if len(curr_worker_cells) == cell_per_worker:
                        # Publish Yaml File, Reset (Worker cell, num) State
                        params["runtime"]["worker"][
                            "worker_cells"
                        ] = curr_worker_cells
                        yaml_path = (
                            Path(self.worker_yml_path)
                            / f"worker_config_{worker_num}.yaml"
                        )
                        io.write_config_yaml(
                            yaml_path=yaml_path, yaml_data=params
                        )

                        curr_worker_cells = []
                        params["runtime"]["worker"] = {}
                        worker_num += 1

                    curr_worker_cells.append([z, y, x])

        # Publish remaining state into last YAML file
        params["runtime"]["worker"]["worker_cells"] = curr_worker_cells
        yaml_path = (
            Path(self.worker_yml_path) / f"worker_config_{worker_num}.yaml"
        )
        io.write_config_yaml(yaml_path=str(yaml_path), yaml_data=params)


class Worker(ComputeNode):
    """
    Defines context for worker run.
    Worker Reads:
    - Local/Cloud tiles, depending on Dataset definition.
    - Scheduler-generated Local Configuruation File containing all
    general run configs + worker specific params.

    Scheduler Outputs:
    - Local/Cloud Volume, depending on output path.
    """

    def __init__(self, config_yaml: str, test_dataset: io.Dataset = None):
        """
        test_dataset: Input for mock datasets used in automated testing.
        Essentially, this exists because creating a synthetic dataset
        with known transformations, etc. in numpy is easier than creating a
        synthetic dataset with BigStitcher, etc.
        """
        super().__init__(config_yaml)
        params = io.read_config_yaml(config_yaml)
        if test_dataset:
            self.DATASET = test_dataset

        assert 'worker' in params['runtime'], \
            f'Worker yaml must contain worker parameters.'
        assert 'log_path' in params['runtime']['worker'], \
            f'Worker yaml must contain a log_path.'

        worker_cells = []
        # Distributed Worker
        if "worker_cells" in params['runtime']['worker']:
            worker_cells = [
                tuple(cell) for cell in params['runtime']["worker"]["worker_cells"]
            ]

        # Solo Worker
        else:
            # Get Output Volume Size from fusion initalization
            _, _, _, _, output_volume_size, _ = fusion.initialize_fusion(
                self.DATASET, self.POST_REG_TFMS, self.OUTPUT_PARAMS
            )

            # Fill worker cells with all work
            z_cnt, y_cnt, x_cnt = fusion.get_cell_count_zyx(
                output_volume_size, self.CELL_SIZE
            )
            for z in range(z_cnt):
                for y in range(y_cnt):
                    for x in range(x_cnt):
                        worker_cells.append((z, y, x))

        self.log_path = params['runtime']['worker']['log_path']
        self.RUNTIME_PARAMS.worker_cells = worker_cells

    def run(self):
        fusion.run_fusion(
            self.DATASET,
            self.OUTPUT_PARAMS,
            self.RUNTIME_PARAMS,
            self.CELL_SIZE,
            self.POST_REG_TFMS,
            self.BLENDING_MODULE,
        )

        # Unique log filename
        unique_id = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)
        unique_file_name = str(Path(self.log_path) / f"file_{timestamp}_{unique_id}.txt")

        log_content = \
        f"""Run complete, wrote cells:
        {self.RUNTIME_PARAMS.worker_cells}
        """
        
        with open(unique_file_name, 'w') as file:
            file.write(log_content)