"""
Runs fusion from a exaspim scheduler
generated config.yml file
"""

import glob 
import os
import uuid
import time
from pathlib import Path
import yaml

import torch

import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.io as io

def read_config_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict

def execute_job(yml_path: str, 
                xml_path: str,
                output_path: str):
    """
    yml_path: Local yml path
    xml_path: Local xml path 
    output_path: Local results path
    """

    # Parse information from worker yaml
    # (See dispim_scheduler data contract)
    configs = read_config_yaml(yml_path)
    input_path = configs['input_path']
    output_path = configs['output_path']
    worker_cells = [tuple(cell) for cell in configs['worker_cells']] 

    # Application Object: DATASET
    xml_path = str(Path(xml_path))
    s3_path = input_path
    DATASET = io.BigStitcherDataset(xml_path, s3_path)

    # Application Object: OUTPUT_PARAMS
    OUTPUT_PARAMS = io.OutputParameters(
        path=output_path,
        chunksize=(1, 1, 128, 128, 128),
        resolution_zyx=(0.5, 0.5, 0.5),
    )

    # Application Object: RUNTIME PARAMS
    RUNTIME_PARAMS = io.RuntimeParameters(
        use_gpus=False,
        devices=[torch.device("cpu")],
        pool_size=16, 
        worker_cells=worker_cells
    )
    # Application Parameter: CELL_SIZE
    CELL_SIZE = [512, 512, 512]

    # Application Parameter: POST_REG_TFMS
    POST_REG_TFMS: list[geometry.Affine] = []
    
    # Application Object: BLENDING_MODULE
    BLENDING_MODULE = blend.MaxProjection()

    # Run fusion
    fusion.run_fusion(
            DATASET,
            OUTPUT_PARAMS,
            RUNTIME_PARAMS,
            CELL_SIZE,
            POST_REG_TFMS,
            BLENDING_MODULE,
    )

    # Log 'done' file for next capsule in pipeline. 
    # Unique log filename
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)
    unique_file_name = str(Path(output_path) / f"file_{timestamp}_{unique_id}.yaml")

    log_content = {}
    log_content['output_path'] = OUTPUT_PARAMS.path
    log_content['resolution_zyx'] = list(OUTPUT_PARAMS.resolution_zyx)

    with open(unique_file_name, "w") as file:
        yaml.dump(log_content, file)


if __name__ == '__main__':
    # Special Initalization for CO: 
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.util.abstract_sockets_supported = False

    num_cpus = int(os.environ.get("CO_CPUS", 16))  # retrieve env var, defaults to 16.
    print(f'{num_cpus=}')
    torch.set_num_threads(num_cpus)

    yml_path = glob.glob('../data/*.yml')[0]
    xml_path = glob.glob('../data/**/*.xml')[0]
    output_path = os.path.abspath('../results')

    execute_job(yml_path, 
                xml_path, 
                output_path)