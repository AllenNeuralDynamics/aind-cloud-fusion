# aind-cloud-fusion

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-79.3%25-red)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.7-blue?logo=python)


### Features
- Lightweight, pure-python installation
- Support for generic image volume file formats (zarr, n5, tiff stack) and transforms (affine, flow field)
- Modular image blending (max projection, linear averaging, deep blending)
- Parallel execution across multiple GPU's
- Deskewing and anisotropy correction

### Runtime Benchmarks
- X CPU's: __ Mb/s
- 4 T4's: __ Mb/s
- 4 V100's: __ Mb/s

### Usage
See `example_runtime.py` which reads all inputs, hyperparameters, and output from `config.yaml`.
Inputs: 
- Matrices must be in ZYX order.


Outputs: 

Hyperparameters:


### High-Level Algorithm
(To be shown pictorially)
1) Transform all volume boundaries by registration transforms
2) Calculate AABB's of transformed boundaries and store for reference. Use AABB's to calculate size of output volume.  
3) Iterate through each output chunk and initialize a coordinate block. Determine chunk-AABB collision and send coordinate block through corresponding inverse registration transforms. 
4) Interpolate intensity values from source images.
5) Blend all source chunks. 
6) Write to output image. 