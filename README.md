# aind-cloud-fusion

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-43.1%25-red)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.7-blue?logo=python)


### Features
- Lightweight, pure-python installation
- Support for generic image volume file formats (zarr, n5, tiff stack) and transforms (affine, flow field). 
Transform types compose and can be used in sequence with one another. 
- Modular image blending (max projection, linear averaging, deep blending)
- Parallel execution across multiple GPU's
- Deskewing and anisotropy correction

## Installation:
1) Run
```
pip install aind-cloud-fusion
```

2) Please visit the PyTorch website and install the version of PyTorch compatible with your local OS and GPU hardware. All other dependencies are installed within package. 

### High-Level Algorithm
(To be shown pictorially)
1) Transform all volume boundaries by registration transforms
2) Calculate AABB's of transformed boundaries and store for reference. Use AABB's to calculate size of output volume.  
3) Iterate through each output chunk and initialize a coordinate block. Determine chunk-AABB collision and send coordinate block through corresponding inverse registration transforms. 
4) Interpolate intensity values from source images.
5) Blend all source chunks. 
6) Write to output image. 

### Usage
See `config.yaml` for all algorithm inputs, hyperparameters, and outputs.

Additional notes on configurations: 
- output_resolution: 
Fusion algorithm operates entirely in continous absolute coordinates and rasterizes the output volume as a final step. By default, output resolution is set to (0.5, 0.5, 0.5) to produce uniformly sized output voxels. Other options for output_resolution may include the input resolution of the raw volumes or a resolution that upsamples/downsamples in specific dimensions to prevent aliasing caused by post-registration transforms. 

- cell_size:
cell_size represents the size of the output that is colored. Fusion algorithm has option to swap between CPU and GPU runtimes. If operating with a GPU runtime, a good rule of thumb is to set the total size of cell_size equivalent to 50-70% of your local GPU memory. If operating with a CPU runtime, cell_size is arbitrary and has no significant impact on runtime. 

Additional notes on dataset:
- Registration transforms are expected in 'voxel'/'volume' basis. Input resolution, which scales the voxels to its absolute size, is expected as a separate input. 
Volume boundaries, as described in high-level algorithm description, go through the following transformations in this order: `registration transforms` -> `input resolution scaling` -> `post-registration transforms` -> `output resolution scaling`.

### Runtime Benchmarks
- X CPU's: __ Mb/s
- 4 T4's: __ Mb/s
- 4 V100's: __ Mb/s

## Contributing
Fusion features serveral generic components that may be extended to fit your use case: 
- Dataset
- Transform
- BlendingModule 

Add additional parameters to `config.yaml` as necessary. 

## Known Issues
- Dask array does not load input zarr with default input chunk size. Expose input chunk size parameter. 

- Expose output zarr compression parameter.

## Suggested Features: 
- Flow Field/Deformation Transform Implementation

- Blend Masking 