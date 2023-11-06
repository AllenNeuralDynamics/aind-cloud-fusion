import os

import torch

import aind_cloud_fusion.runtime as runtime


def main(config_yaml: str):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    compute_node = runtime.initalize_compute_node(config_yaml)
    compute_node.run()

    # NOTE:
    # Multiscaling, which is CPU-bound, can be standalone capsule to save on GPU costs.


if __name__ == "__main__":
    # import wandb
    # wandb.init(project='pytorch-fusion')
    torch.cuda.empty_cache()
    main()
