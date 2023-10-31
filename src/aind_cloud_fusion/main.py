import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch

import aind_cloud_fusion.runtime as runtime
import aind_cloud_fusion.io as io
import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.fusion as fusion


def main(config_yaml: str): 
    compute_node = runtime.initalize_compute_node(config_yml)
    compute_node.run()

    # NOTE: 
    # Multiscaling, which is CPU-bound, can be standalone capsule to save on GPU costs.

if __name__ == "__main__":
    # import wandb
    # wandb.init(project='pytorch-fusion')
    torch.cuda.empty_cache()
    main()