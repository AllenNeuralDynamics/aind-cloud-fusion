import dask.array as da
import torch

import multi_gpu_fusion
import multiscale

torch.cuda.empty_cache()

xml_path = '/root/capsule/code/bigstitcher_2023-09-18.xml'

output_path = "/root/capsule/results/gpu_output_volume_2_tile.zarr"
cloud_path = "s3://aind-msma-morphology-data/test_data/exaSPIM_674191_fusion.zarr"

multi_gpu_fusion.main(xml_path, output_path=output_path)
arr = zarr.open(output_path, mode='r')
arr = da.from_zarr(arr)
out_group = zarr.open_group(cloud_path, mode='w')      
voxel_sizes_zyx = (0.176, 0.298, 0.298)
run_multiscale(arr, out_group, voxel_sizes_zyx)