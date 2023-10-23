import os
import multiprocessing
import numpy as np
import torch
import zarr

import logging
import time

from aind_cloud_fusion.io import Dataset, LazyArray, OutputParameters
from aind_cloud_fusion.geometry import Transform, Affine, AABB, aabb_3d
from aind_cloud_fusion.blend import BlendingModule

# Parallelized Function
def color_cell(tile_arrays: dict[int, LazyArray], 
               tile_transforms: dict[int, list[Transform]], 
               tile_sizes_zyx: dict[int, tuple[int, int, int]], 
               tile_aabbs: dict[int, AABB],
               output_volume: zarr.core.Array, 
               output_volume_origin: torch.Tensor, 
               cell_size: tuple[int, int, int], 
               blend_module: BlendingModule,  
               z: int, y: int, x: int, device: torch.device, 
               cell_num: int, est_total_cells: int, logger: logging.Logger): 
    
    LOGGER = logger
    start_cell = time.time()

    # Cell Boundaries, exclusive stop index
    output_volume_size = output_volume.shape
    cell_box = np.array([[z * cell_size[0], z * cell_size[0] + cell_size[0]], 
                        [y * cell_size[1], y * cell_size[1] + cell_size[1]], 
                        [x * cell_size[2], x * cell_size[2] + cell_size[2]]])
    cell_box[:, 1] = np.minimum(cell_box[:, 1], np.array(output_volume_size[2:]))
    cell_box = cell_box.flatten()

    # Collision Detection
    # Collision defined by overlapping intervals in all 3 dimensions.
    # Two intervals (A, B) collide if A_max is not <= B_min and A_min is not >= B_max. 
    overlapping_tiles: list[int] = []
    for tile_id, t_aabb in tile_aabbs.items(): 
        if (cell_box[1] > t_aabb[0] and cell_box[0] < t_aabb[1]) and \
            (cell_box[3] > t_aabb[2] and cell_box[2] < t_aabb[3]) and \
            (cell_box[5] > t_aabb[4] and cell_box[4] < t_aabb[5]):
            overlapping_tiles.append(tile_id)

    LOGGER.info(f'{cell_box=}')
    LOGGER.info(f'{overlapping_tiles=}')
    
    # Interpolation
    z_length = cell_box[1] - cell_box[0]
    y_length = cell_box[3] - cell_box[2]
    x_length = cell_box[5] - cell_box[4]

    fused_cell = torch.zeros((1, 1, z_length, y_length, x_length)).to(device)
    for tile_id in overlapping_tiles: 
        # Init tile coords, arange end-exclusive, +0.5 to represent voxel center
        z_indices = torch.arange(cell_box[0], cell_box[1], step=1) + 0.5
        y_indices = torch.arange(cell_box[2], cell_box[3], step=1) + 0.5 
        x_indices = torch.arange(cell_box[4], cell_box[5], step=1) + 0.5
        z_indices = z_indices.to(device)
        y_indices = y_indices.to(device)
        x_indices = x_indices.to(device)

        z_grid, y_grid, x_grid = torch.meshgrid(z_indices, y_indices, x_indices, indexing='ij')
        z_grid = torch.unsqueeze(z_grid, 0)
        y_grid = torch.unsqueeze(y_grid, 0)
        x_grid = torch.unsqueeze(x_grid, 0)

        tile_coords = torch.concatenate((z_grid, y_grid, x_grid), axis=0)
        # (3, z, y, x) -> (z, y, x, 3)
        tile_coords = torch.movedim(tile_coords, source=0, destination=3)

        # Define tile coords wrt output vol origin
        tile_coords = tile_coords + torch.Tensor(output_volume_origin).to(device)

        # Send tile_coords through inverse transforms
        # NOTE: tile_transforms list must be iterated thru in reverse
        # (z, y, x, 3) -> (z, y, x, 3)
        for tfm in reversed(tile_transforms[tile_id]):
            tile_coords = tfm.backward(tile_coords, device=device)

        # Calculate AABB of transformed coords
        z_min, z_max, y_min, y_max, x_min, x_max = aabb_3d(tile_coords)

        # Mini Optimization: Check true collision before executing interpolation/fusion
        # That is, aabb of transformed coordinates into imagespace actually overlap the image. 
        t_size_zyx = tile_sizes_zyx[tile_id]
        if not ((z_max > 0 and z_min < t_size_zyx[0]) and
                (y_max > 0 and y_min < t_size_zyx[1]) and 
                (x_max > 0 and x_min < t_size_zyx[2])): 
            continue

        # Calculate overlapping region between transformed coords and image boundary
        # For intervals (A, B): 
        # The lower bound of overlapping region = max(A_min, B_min)
        # The upper bound of overlapping region = min(A_max, B_max)
        crop_min_z = torch.max(torch.Tensor([0, z_min]))
        crop_max_z = torch.min(torch.Tensor([t_size_zyx[0], z_max]))
        
        crop_min_y = torch.max(torch.Tensor([0, y_min]))
        crop_max_y = torch.min(torch.Tensor([t_size_zyx[1], y_max]))
        
        crop_min_x = torch.max(torch.Tensor([0, x_min]))
        crop_max_x = torch.min(torch.Tensor([t_size_zyx[2], x_max]))

        # Make sure crop_{min, max}_{z, y, x} are integers to be used as indices. 
        # Minimum values are rounded down to nearest integer. 
        # Maximum values are rounded up to nearest integer.
        crop_min_z = int(torch.floor(crop_min_z))
        crop_min_y = int(torch.floor(crop_min_y))
        crop_min_x = int(torch.floor(crop_min_x))

        crop_max_z = int(torch.ceil(crop_max_z))
        crop_max_y = int(torch.ceil(crop_max_y))
        crop_max_x = int(torch.ceil(crop_max_x))

        LOGGER.info(f'AABB of Crop belonging to image {tile_id}: {crop_min_z}, {crop_max_z}, {crop_min_y}, {crop_max_y}, {crop_min_x}, {crop_max_x}')
        
        # Define tile coords wrt base image crop coordinates
        image_crop_offset = torch.Tensor([crop_min_z, crop_min_y, crop_min_x]).to(device)
        tile_coords = tile_coords - image_crop_offset

        # Prep inputs to interpolation
        image_crop_slice = (slice(crop_min_z, crop_max_z), 
                            slice(crop_min_y, crop_max_y), 
                            slice(crop_min_x, crop_max_x))
        image_crop = tile_arrays[tile_id][image_crop_slice]
        image_crop = image_crop.astype(np.int32)  # Promote uint16 -> Pytorch compatible int32
        image_crop = torch.Tensor(image_crop).to(device)

        # Pytorch flow field follows a different basis than the image numpy basis.
        # Change of basis to interpolation basis, which preserves relative distances/angles/positions.  
        # (z, y, x, 3) -> (z, y, x, 3)
        interp_cob_matrix = torch.Tensor([[0, 0, 1, 0], 
                                          [0, 1, 0, 0], 
                                          [1, 0, 0, 0]])
        interp_cob = Affine(interp_cob_matrix)
        tile_coords = interp_cob.forward(tile_coords, device=device)

        # Interpolation expects 'grid' parameter/sample locations to be normalized [-1, 1].
        # Very specific per-dimension normalization according to CoB
        crop_z_length = crop_max_z - crop_min_z
        crop_y_length = crop_max_y - crop_min_y
        crop_x_length = crop_max_x - crop_min_x
        tile_coords[:, :, :, 0] = (tile_coords[:, :, :, 0] - (crop_x_length / 2)) / (crop_x_length / 2)
        tile_coords[:, :, :, 1] = (tile_coords[:, :, :, 1] - (crop_y_length / 2)) / (crop_y_length / 2)
        tile_coords[:, :, :, 2] = (tile_coords[:, :, :, 2] - (crop_z_length / 2)) / (crop_z_length / 2) 

        # Final reshaping
        # image_crop: (z_in, y_in, x_in) -> (1, 1, z_in, y_in, x_in)
        # tile_coords: (z_out, y_out, x_out, 3) -> (1, z_out, y_out, x_out, 3)
        # => tile_contribution: (1, 1, z_out, y_out, x_out)
        image_crop = image_crop[(None,)*2]
        tile_coords = torch.unsqueeze(tile_coords, 0)
        
        # Interpolate and Fuse
        tile_contribution = torch.nn.functional.grid_sample(image_crop, tile_coords, 
                                                            padding_mode='zeros', mode='nearest')
        fused_cell = blend_module.blend([fused_cell, tile_contribution], device=device)

        del tile_coords
        del tile_contribution

    # Write
    output_slice = (slice(0, 1), 
                    slice(0, 1), 
                    slice(cell_box[0], cell_box[1]),
                    slice(cell_box[2], cell_box[3]),
                    slice(cell_box[4], cell_box[5]))
    # Convert from float32 -> canonical uint16
    output_chunk = np.array(fused_cell.cpu()).astype(np.uint16)
    output_volume[output_slice] = output_chunk
    
    del fused_cell

    LOGGER.info(f'Cell {cell_num}/{est_total_cells}: {time.time() - start_cell}')


def run_fusion(dataset: Dataset,
               output_params: OutputParameters, 
               devices: list[torch.device], 
               cell_size: tuple[int, int, int],
               post_reg_tfms: list[Affine],
               blend_module: BlendingModule):
    
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)

    # Initalize Core Data Structures
    tile_arrays: dict[int, LazyArray] = dataset.tile_volumes_zyx
    LOGGER.info(f'Number of Tiles: {len(tile_arrays)}')

    tile_transforms: dict[int, list[Transform]] = dataset.tile_transforms_zyx
    input_resolution_zyx: tuple[float, float, float] = dataset.tile_resolution_zyx
    iz, iy, ix = input_resolution_zyx
    scale_input_zyx = Affine(np.array([[iz, 0, 0, 0], 
                                       [0, iy, 0, 0], 
                                       [0, 0, ix, 0]]))
    # Let me first try applying to the affine of the net transform. 

    # for tile_id, tfm_list in tile_transforms.items():
    #     net_transform_3x3 = tfm_list[0].matrix_3x3
    #     nz, ny, nx = net_translation = tfm_list[0].translation
    #     updated_translation = np.array([[iz * nz], 
    #                                     [iy * ny], 
    #                                     [ix * nx]])
    #     updated_matrix = np.hstack((net_transform_3x3, updated_translation))
    #     tile_transforms[tile_id][0] = Affine(updated_matrix)

    output_resolution_zyx: tuple[float, float, float] = output_params.resolution_zyx
    oz, oy, ox = output_resolution_zyx
    sample_output_zyx = Affine(np.array([[1/oz, 0, 0, 0], 
                                         [0, 1/oy, 0, 0], 
                                         [0, 0, 1/ox, 0]]))
    for tile_id, tfm_list in tile_transforms.items():
        tile_transforms[tile_id] = [*tfm_list, 
                                    scale_input_zyx,
                                    *post_reg_tfms,
                                    sample_output_zyx]

    tile_sizes_zyx: dict[int, tuple[int, int, int]] = {}
    tile_aabbs: dict[int, AABB] = {}
    tile_boundary_point_cloud_zyx = [] 

    # tmp_buffer = []
    
    for tile_id, tile_arr in tile_arrays.items():
        tile_sizes_zyx[tile_id] = zyx = tile_arr.shape
        tile_boundaries = torch.Tensor([[0., 0., 0.], 
                        [zyx[0], 0., 0.],
                        [0., zyx[1], 0.],
                        [0., 0., zyx[2]],
                        [zyx[0], zyx[1], 0.],
                        [zyx[0], 0., zyx[2]],
                        [0., zyx[1], zyx[2]],
                        [zyx[0], zyx[1], zyx[2]]])  
        
        # Appears like affine part is not defined wrt to underlying resolution basis, but voxel basis. 
        # Okay fine. How to resolve?
        # First, check if this is correct. 
        # That is, first transform's affine component must be sent through forward input resolution. 

        tfm_list = tile_transforms[tile_id]
        for i, tfm in enumerate(tfm_list): 
            tile_boundaries = tfm.forward(tile_boundaries, device=torch.device('cpu'))

            # if i == 1: 
            #     tmp_buffer.append(tile_boundaries.to('cpu'))
            # # (Following registration)
        
        # Checking to see if removing the initial rescaling changes things: 
        # tmp_buffer.append(tile_boundaries.to('cpu'))

        tile_aabbs[tile_id] = aabb_3d(tile_boundaries)
        tile_boundary_point_cloud_zyx.extend(tile_boundaries)
    tile_boundary_point_cloud_zyx = torch.stack(tile_boundary_point_cloud_zyx, dim=0)

    # Bug must be related to initalization. 
    # You can probably solve this tomorrow (Monday).
    # import plotly.graph_objects as go
    # import random
    # def get_random_rgb_string():
    #     r = random.randint(0, 255)
    #     g = random.randint(0, 255)
    #     b = random.randint(0, 255)
    #     return f'rgb({r}, {g}, {b})'

    # scene = []
    # for verts in tmp_buffer:
    #     color = get_random_rgb_string()
    #     vert_plot = go.Scatter3d(x=verts[:, 0],
    #                             y=verts[:, 1],
    #                             z=verts[:, 2],
    #                             mode='markers', 
    #                             marker=dict(color=color, size=10))
    #     scene.append(vert_plot)
    # fig = go.Figure(data=scene)
    # fig.write_html('/scratch/volume_init_new.html')

    # Resolve Output Volume Dimensions and Absolute Position
    global_tile_boundaries = aabb_3d(tile_boundary_point_cloud_zyx)
    OUTPUT_VOLUME_SIZE = (int(global_tile_boundaries[1] - global_tile_boundaries[0]), 
                          int(global_tile_boundaries[3] - global_tile_boundaries[2]),
                          int(global_tile_boundaries[5] - global_tile_boundaries[4]))
    LOGGER.info(f'{OUTPUT_VOLUME_SIZE=}')

    OUTPUT_VOLUME_ORIGIN = torch.Tensor([
        torch.min(tile_boundary_point_cloud_zyx[:, 0]).item(),
        torch.min(tile_boundary_point_cloud_zyx[:, 1]).item(),
        torch.min(tile_boundary_point_cloud_zyx[:, 2]).item()
    ])
    
    # Shift AABB's into Output Volume where
    # absolute position of output volume is moved to (0, 0, 0)
    for tile_id, t_aabb in tile_aabbs.items():
        updated_aabb = (t_aabb[0] - OUTPUT_VOLUME_ORIGIN[0], t_aabb[1] - OUTPUT_VOLUME_ORIGIN[0],
                        t_aabb[2] - OUTPUT_VOLUME_ORIGIN[1], t_aabb[3] - OUTPUT_VOLUME_ORIGIN[1], 
                        t_aabb[4] - OUTPUT_VOLUME_ORIGIN[2], t_aabb[5] - OUTPUT_VOLUME_ORIGIN[2])
        tile_aabbs[tile_id] = updated_aabb 

    # Initalize Output Volume
    out_group = zarr.open_group(output_params.path, mode='w')
    path = "0"
    chunksize = output_params.chunksize
    datatype = output_params.dtype
    dimension_separator = "/"
    compressor = output_params.compressor    
    global output_volume
    output_volume = out_group.create_dataset(
        path,
        shape=(1, 1, OUTPUT_VOLUME_SIZE[0], OUTPUT_VOLUME_SIZE[1], OUTPUT_VOLUME_SIZE[2]),
        chunks=chunksize,
        dtype=datatype,
        compressor=compressor,
        dimension_separator=dimension_separator,
        overwrite=True,
        fill_value=0
    )

    # Run Fusion: Define all work
    z_stride = int(np.ceil(OUTPUT_VOLUME_SIZE[0] / cell_size[0]))
    y_stride = int(np.ceil(OUTPUT_VOLUME_SIZE[1] / cell_size[1]))
    x_stride = int(np.ceil(OUTPUT_VOLUME_SIZE[2] / cell_size[2]))
    est_total_cells = z_stride * y_stride * x_stride
    LOGGER.info(f'Estimated Total Cells: {est_total_cells}')

    process_args: list[dict] = []
    cell_num = 0
    for z in range(z_stride):
        for y in range(y_stride): 
            for x in range(x_stride):
                process_args.append({'cell_num': cell_num, 
                                      'z': z, 
                                      'y': y, 
                                      'x': x})
                cell_num += 1
    
    # Single threaded execution
    for p_args in process_args:
        color_cell(tile_arrays, 
                   tile_transforms, 
                   tile_sizes_zyx, 
                   tile_aabbs, 
                   output_volume, 
                   OUTPUT_VOLUME_ORIGIN,
                   cell_size, 
                   blend_module,
                   p_args['z'], p_args['y'], p_args['x'], devices[0],
                   p_args['cell_num'], est_total_cells, LOGGER)
    

    # Multithreaded execution
    """
    # Run Fusion: Fill work queue with inital tasks
    # Task-specific info includes process_args and device. 
    if devices[0] == torch.device('cpu'): 
        pool_size = os.cpu_count() // 2
        LOGGER.info(f'CPU Runtime: Using {pool_size} CPUs')
    else: 
        pool_size = len(devices)
        LOGGER.info(f'GPU Runtime: Using {pool_size} GPUs')
    
    start_run = time.time()
    active_processes: list[tuple] = []
    for i in range(pool_size):
        p_args = process_args.pop(0)
        p = multiprocessing.Process(target=color_cell,
                                    args=(tile_arrays, 
                                          tile_transforms, 
                                          tile_sizes_zyx, 
                                          tile_aabbs, 
                                          output_volume, 
                                          OUTPUT_VOLUME_ORIGIN,
                                          cell_size, 
                                          blend_module,
                                          p_args['z'], p_args['y'], p_args['x'], devices[i % len(devices)],
                                          p_args['cell_num'], est_total_cells, LOGGER))
        active_processes.append((devices[i % len(devices)], p))
        p.start()
    
    # Run Fusion: Exhaust all the tasks 
    # Tasks all implictly defined in process_args buffer
    while len(active_processes) != 0: 
        tmp = []
        for (device, p) in active_processes:
            p.join(timeout=0)   # timeout indicates do not wait until the process is explicitly finished

            if p.is_alive():
                tmp.append((device, p))
            else:
                p.close()

                if len(process_args) != 0:
                    p_args = process_args.pop(0)
                    new_p = multiprocessing.Process(target=color_cell,
                                        args=(tile_arrays, 
                                              tile_transforms, 
                                              tile_sizes_zyx, 
                                              tile_aabbs, 
                                              output_volume, 
                                              OUTPUT_VOLUME_ORIGIN,
                                              cell_size, 
                                              blend_module,
                                              p_args['z'], p_args['y'], p_args['x'], device,
                                              p_args['cell_num'], est_total_cells, LOGGER))
                    tmp.append((device, new_p))
                    new_p.start()

            active_processes = tmp

    LOGGER.info(f'Runtime: {time.time() - start_run}')
    """