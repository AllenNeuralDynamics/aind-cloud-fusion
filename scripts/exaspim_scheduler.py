"""
Defines configuration file with unique fields per exaspim worker. 
"""

from collections import OrderedDict
import glob
import os
import xmltodict
import yaml

import numpy as np
from pathlib import Path
import torch

import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.io as io


"""
DATA CONTRACT: 
Exaspim scheduler outputs a configuration file 
with the following minimal fields. 

pipeline: 'exaspim'
input_path: 's3://<YOUR INPUT PATH>'
output_path: "s3://<YOUR OUTPUT PATH>"
worker_cells: [list of cell 3-ples]
"""

def write_config_yaml(yaml_path: str, yaml_data: dict) -> None:
    with open(yaml_path, "w") as file:
        yaml.dump(yaml_data, file)

def create_starter_ymls(xml_path: str, 
                        output_path: str, 
                        num_workers: int = 64):
    """
    xml_path: Local xml path 
    output_path: Local results path
    """

    # Construct I/O cloud paths from local paths in xml
    # This code converts:
    # This: /root/capsule/data/exaSPIM_674191_2023-09-12_12-37-37/exaSPIM.zarr
    # To This (input path): s3://aind-open-data/exaSPIM_674191_2023-09-12_12-37-37/exaSPIM.zarr/
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())
    parsed_path = data["SpimData"]["SequenceDescription"]["ImageLoader"][
        "zarr"
    ]["#text"]
    parts = parsed_path.split('/')
    if 'root' in parts:
        parts.remove('root')
    if 'capsule' in parts: 
        parts.remove('capsule')
    input_s3_path = 's3://aind-open-data/' + parts[2] + '/' + parts[3] + '/'
    output_s3_path_base = 's3://aind-open-data/' + parts[2] + '_full_res/'

    # Init application objects to init fusion scheduling. 
    # Application Object: DATASET
    xml_path = str(Path(xml_path))
    s3_path = input_s3_path
    DATASET = io.BigStitcherDataset(xml_path, s3_path)

    # Application Object: OUTPUT_PARAMS
    OUTPUT_PARAMS = io.OutputParameters(
        path=output_path,
        chunksize=(1, 1, 128, 128, 128),
        resolution_zyx=(1.0, 0.748, 0.748),
    )

    # Application Object: RUNTIME PARAMS
    # Not required here

    # Application Parameter: CELL_SIZE
    CELL_SIZE = [512, 512, 512]

    # Application Parameter: POST_REG_TFMS
    POST_REG_TFMS: list[geometry.Affine] = []
    
    # Application Object: BLENDING_MODULE
    # Not required here

    # Initalize fusion
    _, _, _, _, output_volume_size, _ = fusion.initialize_fusion(
        DATASET, POST_REG_TFMS, OUTPUT_PARAMS
    )

    # Divide work and export into yamls
    z_cnt, y_cnt, x_cnt = fusion.get_cell_count_zyx(
        output_volume_size, CELL_SIZE
    )
    zs = np.arange(0, z_cnt, 1)
    ys = np.arange(0, y_cnt, 1)
    xs = np.arange(0, x_cnt, 1)
    x, y, z = np.meshgrid(xs, ys, zs)
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    cell_coords = np.column_stack((z_flat, y_flat, x_flat))

    # Ex math: 
    # 50 cells
    # 6 workers
    # ceil(50 / 6) -> 9
    n = int(np.ceil(len(cell_coords) / num_workers))
    print(f'Each worker assigned {n} cells')
    for i in range(num_workers):
        print(f'Generating Worker {i} Yaml')
        start = i * n 
        end = (i + 1) * n
        worker_cells = cell_coords[start:end, :].tolist()

        configs = {}
        configs['pipeline'] = 'exaspim'
        configs['input_path'] = input_s3_path
        configs['output_path'] = output_s3_path_base + 'channel_561.zarr'
        configs['worker_cells'] = worker_cells

        yaml_path = (
            Path(output_path)
            / f"worker_config_{i}.yml"
        )
        io.write_config_yaml(
            yaml_path=yaml_path, yaml_data=configs
        )

if __name__ == '__main__':
    xml_path = str(glob.glob('../data/**/*.xml')[0])
    output_path = str(os.path.abspath('../results'))

    print(f'{xml_path=}')
    print(f'{output_path=}')
    
    create_starter_ymls(xml_path, 
                        output_path, 
                        num_workers = 500)