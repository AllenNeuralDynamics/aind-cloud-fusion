"""
Defines configuration file with unique fields per dispim worker. 
"""

from collections import OrderedDict
import glob
import os
import re

import boto3
from pathlib import Path
import xmltodict
import yaml
 
"""
DATA CONTRACT: 
Dispim scheduler outputs a configuration file 
with the following minimal fields. 

dataset_type: {BigStitcherDataset, BigStitcherDatasetChannel}
channel: num
input_path: 's3://<YOUR INPUT PATH>'
output_path: "s3://<YOUR OUTPUT PATH>"
"""


def write_config_yaml(yaml_path: str, yaml_data: dict) -> None:
    with open(yaml_path, "w") as file:
        yaml.dump(yaml_data, file)

def list_all_tiles_in_bucket_path(
    bucket_SPIM_folder: str, bucket_name="aind-open-data"
) -> list:
    """
    Lists all tiles in a given bucket path

    Parameters
    ------------------------
    bucket_SPIM_folder: str
        Path to SPIM folder in bucket.
    bucket_name: str
        Name of bucket.

    Returns
    ------------------------
    list:
        List of all tiles in SPIM folder.
    """
    # s3 = boto3.resource('s3')
    bucket_name, prefix = bucket_SPIM_folder.replace("s3://", "").split("/", 1)

    client = boto3.client("s3")
    result = client.list_objects(
        Bucket=bucket_name, Prefix=prefix, Delimiter="/"
    )

    tiles = []
    for o in result.get("CommonPrefixes"):
        tiles.append(o.get("Prefix"))
    return tiles

def extract_channel_from_tile_path(t_path: str) -> int:
    """
    Extracts channel from tile path naming convention:
    tile_X_####_Y_####_Z_####_ch_####.filetype

    Parameters
    ------------------------
    t_path: str
        Tile path to run regex on.

    Returns
    ------------------------
    int:
        Channel value.

    """

    pattern = r"(ch|CH)_(\d+)"
    match = re.search(pattern, t_path)
    channel = int(match.group(2))
    return channel

def get_unique_channels_for_dataset(dataset_path: str) -> list:
    """
    Extracts a list of channels in a given dataset

    Parameters:
    -----------
    dataset_path: str
        Path to a dataset's zarr folder

    Returns:
    --------
    unique_list_of_channels: list(int)
        A list of int, containing the unique list of channel wavelengths

    """
    # Reference Path: s3://aind-open-data/HCR_677594_2023-10-13_13-55-48/SPIM.ome.zarr/
    # path_parts = dataset_path.split('/')    
    # tiles_in_path = list_bucket_directory(path_parts[2], path_parts[3] + '/' + path_parts[4])

    tiles_in_path = list_all_tiles_in_bucket_path(
        dataset_path, "aind-open-data"
    )

    unique_list_of_channels = []
    for tile in tiles_in_path:
        channel = extract_channel_from_tile_path(tile)

        if channel not in unique_list_of_channels:
            unique_list_of_channels.append(channel)

    return unique_list_of_channels

def create_starter_ymls(xml_path: str, 
                        output_path: str):
    """
    xml_path: Local xml path 
    output_path: Local results path
    """

    # Construct I/O cloud paths from local paths in xml
    # This code converts:
    # This: /data/HCR_677594_2023-10-13_13-55-48/SPIM.ome.zarr
    # To This (input path): s3://aind-open-data/HCR_677594_2023-10-13_13-55-48/SPIM.ome.zarr/
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())
    parsed_path = data["SpimData"]["SequenceDescription"]["ImageLoader"][
        "zarr"
    ]["#text"]
    parts = parsed_path.split('/')
    input_s3_path = 's3://aind-open-data/' + parts[2] + '/' + parts[3] + '/'
    output_s3_path_base = 's3://aind-open-data/' + parts[2] + '_full_res/'

    # Write output yamls
    # Channel 405: 
    output_path = Path(output_path)
    ch_405_configs = {}
    ch_405_configs['dataset_type'] = 'BigStitcherDataset' 
    ch_405_configs['input_path'] = input_s3_path
    ch_405_configs['output_path'] = output_s3_path_base + 'channel_405.zarr'
    write_config_yaml(str(output_path / 'config_405.yml'), ch_405_configs)

    # Other Channels: 
    channels = get_unique_channels_for_dataset(input_s3_path)
    channels.remove(405)
    for ch in channels:
        ch_X_configs = {}
        ch_X_configs['dataset_type'] = 'BigStitcherDatasetChannel'
        ch_X_configs['channel'] = ch
        ch_X_configs['input_path'] = input_s3_path
        ch_X_configs['output_path'] = output_s3_path_base + f'channel_{ch}.zarr'
        write_config_yaml(str(output_path / f'config_{ch}.yml'), ch_X_configs)


if __name__ == '__main__':
    xml_paths = glob.glob('../data/**/*.xml')
    assert len(xml_paths) > 0, "No xml found, please provide input xml."
    assert len(xml_paths) == 1, "Multiple xml's found, please provide single input xml."    
    xml_path = str(xml_paths[0])

    results_folder = str(os.path.abspath('../results'))
    
    create_starter_ymls(xml_path, results_folder)