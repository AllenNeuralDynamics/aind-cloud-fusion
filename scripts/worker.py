"""
Runs fusion from config file generated 
from dispim or exaspim scheduler. 
"""

import glob
import os
import uuid
import time
from pathlib import Path
import yaml

import torch
import multiprocessing as mp
import numpy as np

import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.io as io

def read_config_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict

def run_exaspim_worker(yml_path: str, 
                       xml_path: str,
                       output_path: str): 
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

def run_dispim_worker(yml_path: str, 
                      xml_path: str,
                      output_path: str):
    # Parse information from worker yaml
    # (See dispim_scheduler data contract)
    configs = read_config_yaml(yml_path)
    dataset_type = configs['dataset_type']
    if 'channel' in configs: 
        channel = int(configs['channel'])
    input_path = configs['input_path']
    output_path = configs['output_path']

    # Initialize Application Objects
    # Application Object: DATASET
    if dataset_type == 'BigStitcherDataset':
        xml_path = str(Path(xml_path))
        s3_path = input_path
        dataset = io.BigStitcherDataset(xml_path, s3_path)
    elif dataset_type == 'BigStitcherDatasetChannel': 
        xml_path = str(Path(xml_path))
        s3_path = input_path
        channel_num = channel
        dataset = io.BigStitcherDatasetChannel(xml_path, s3_path, channel_num)
    DATASET = dataset

    # Application Object: OUTPUT_PARAMS
    OUTPUT_PARAMS = io.OutputParameters(
        path=output_path,
        chunksize=(1, 1, 128, 128, 128),
        resolution_zyx=(0.5, 0.5, 0.5),
    )

    # Application Object: RUNTIME PARAMS
    # Will define worker_cells at the end
    RUNTIME_PARAMS = io.RuntimeParameters(
        use_gpus=False,
        devices=[torch.device("cpu")],
        pool_size=16
    )

    # Application Parameter: CELL_SIZE
    CELL_SIZE = [512, 512, 512]

    # Application Parameter: POST_REG_TFMS
    POST_REG_TFMS: list[geometry.Affine] = []
    matrix = [[1, 0, -1, 0], 
              [0, 1, 0, 0], 
              [0, 0, 1, 0]]
    deskewing = geometry.Affine(np.array(matrix))
    POST_REG_TFMS.append(deskewing)

    # Application Object: BLENDING_MODULE
    BLENDING_MODULE = blend.MaxProjection()

    # Finally, initalize and run fusion. 
    # For the dispim pipeline, each worker will run on an entire channel. 
    worker_cells = []
    _, _, _, _, output_volume_size, _ = fusion.initialize_fusion(
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
    

def execute_job(yml_path: str, 
                xml_path: str,
                output_path: str):
    """
    yml_path: Local yml path
    xml_path: Local xml path 
    output_path: Local results path
    """

    # Parse Pipeline Type:
    configs = read_config_yaml(yml_path)
    pipeline = configs['pipeline']

    # EXASPIM WORKER
    if pipeline == 'exaspim': 
        run_exaspim_worker(yml_path,
                           xml_path, 
                           output_path)

    # DISPIM WORKER
    elif pipeline == 'dispim':
        run_dispim_worker(yml_path, 
                          xml_path, 
                          output_path)

if __name__ == '__main__':
    # Special Initalization for CO: 
    torch.multiprocessing.set_start_method('forkserver', force=True)
    mp.util.abstract_sockets_supported = False

    num_cpus = int(os.environ.get("CO_CPUS", 16))  # retrieve env var, defaults to 16.
    print(f'{num_cpus=}')
    torch.set_num_threads(num_cpus)

    yml_path = str(glob.glob('../data/*.yaml')[0])
    xml_path = str(glob.glob('../data/**/*.xml')[0])
    output_path = str(os.path.abspath('../results'))

    execute_job(yml_path, 
                xml_path, 
                output_path)