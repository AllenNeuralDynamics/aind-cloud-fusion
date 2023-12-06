"""
Defines configuration file with unique fields per dispim worker. 
"""

from collections import OrderedDict
import glob
import os

from pathlib import Path
import xmltodict
import yaml

import script_utils

"""
DATA CONTRACT: 
Dispim scheduler outputs a configuration file 
with the following minimal fields. 

pipeline: 'dispim'
dataset_type: {BigStitcherDataset, BigStitcherDatasetChannel}
channel: num
input_path: 's3://<YOUR INPUT PATH>'
output_path: "s3://<YOUR OUTPUT PATH>"
"""

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
    ch_405_configs['pipeline'] = 'dispim'
    ch_405_configs['dataset_type'] = 'BigStitcherDataset' 
    ch_405_configs['input_path'] = input_s3_path
    ch_405_configs['output_path'] = output_s3_path_base + 'channel_405.zarr'
    script_utils.write_config_yaml(str(output_path / 'config_405.yml'), ch_405_configs)

    # Other Channels: 
    channels = script_utils.get_unique_channels_for_dataset(input_s3_path)
    channels.remove(405)
    for ch in channels:
        ch_X_configs = {}
        ch_X_configs['pipeline'] = 'dispim'
        ch_X_configs['dataset_type'] = 'BigStitcherDatasetChannel'
        ch_X_configs['channel'] = ch
        ch_X_configs['input_path'] = input_s3_path
        ch_X_configs['output_path'] = output_s3_path_base + f'channel_{ch}.zarr'
        script_utils.write_config_yaml(str(output_path / f'config_{ch}.yml'), ch_X_configs)


if __name__ == '__main__':
    # xml_path = str(glob.glob('../data/**/*.xml')[0])  
    xml_path = str(glob.glob('../data/*.xml')[0])  
    output_path = str(os.path.abspath('../results'))
    
    print(f'{xml_path=}')
    print(f'{output_path=}')

    create_starter_ymls(xml_path, output_path)