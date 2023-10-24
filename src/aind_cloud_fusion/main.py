import numpy as np
import torch
import multiprocessing as mp

import aind_cloud_fusion.io as io
import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.fusion as fusion

def main():
    # NOTE: 
    # Add Data Validation at greater maturity. 
    params = io.read_config_yaml(yaml_path='/root/capsule/code/aind-cloud-fusion/src/aind_cloud_fusion/config.yaml')

    dataset_type = params['input']['dataset_type']
    if dataset_type == 'big_stitcher':
        xml_path = params['dataset_parameters']['big_stitcher']['xml_path']
        DATASET = io.BigStitcherDataset(xml_path)

    OUTPUT_PARAMS = io.OutputParameters(path=params['output']['path'],
                                    chunksize=tuple(params['output']['chunksize']), 
                                    resolution_zyx=tuple(params['output']['resolution_zyx']))

    if params['algorithm_parameters']['use_gpus']:
        DEVICES = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    else: 
        DEVICES = [torch.device('cpu')]

    CELL_SIZE = params['algorithm_parameters']['cell_size']
    if params['algorithm_parameters']['blending_module'] == 'MaxProjection': 
        BLENDING_MODULE = blend.MaxProjection()
    
    POST_REG_TFMS: list[geometry.Affine] = [] 
    for tfm in params['algorithm_parameters']["post_registration_transforms_zyx"]: 
        POST_REG_TFMS.append(geometry.Affine(np.array(tfm)))

    fusion.run_fusion(DATASET, 
                      OUTPUT_PARAMS,
                      DEVICES, 
                      CELL_SIZE, 
                      POST_REG_TFMS,
                      BLENDING_MODULE)

    # NOTE: 
    # Multiscaling, which is CPU-bound, can be standalone capsule to save on GPU costs.

if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    mp.set_start_method('forkserver', force=True)
    torch.cuda.empty_cache()
    main()