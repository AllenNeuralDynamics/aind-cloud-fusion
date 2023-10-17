import multiprocessing
import json
import numpy as np
import torch
import zarr

import logging
import time

import dataloading

# Parallelized function
def color_cell(tile_arrays, tile_transforms, tile_sizes_zyx, tile_aabbs,
               output_volume, output_volume_size, output_volume_origin, 
               cell_size, z, y, x, device, 
               cell_num, est_total_cells, logger): 
    """
    Parameters: 
    ----------
    tile_arrays: dict of tile_id -> zarr array
    tile_transforms: dict of tile_id -> numpy matrix
    tile_sizes_zyx: dict of tile_id -> tile size
    tile_aabbs: dict of tile_id -> tile aabb
    output_volume: global output zarr store
    output_volume_size: size of output store
    output_volume_origin: location of output volume in absolute coordinates
    cell_size: size of output chunk
    z, y, x: location of output chunk
    device: cuda device to execute work on
    cell_num, est_total_cells: helpful logging information
    logger: global logger instance

    Returns
    ----------
        None, colors output volume buffer
    """


    LOGGER = logger
    start_cell = time.time()

    # Cell Boundaries, exclusive stop index
    cell_box = np.array([[z * cell_size[0], z * cell_size[0] + cell_size[0]], 
                        [y * cell_size[1], y * cell_size[1] + cell_size[1]], 
                        [x * cell_size[2], x * cell_size[2] + cell_size[2]]])
    cell_box[:, 1] = np.minimum(cell_box[:, 1], np.array(output_volume_size))
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

        # Send tile coords through inverse affine
        forward_matrix = torch.Tensor(tile_transforms[tile_id])
        forward_matrix_3x3 = forward_matrix[:, :3]
        forward_translation = forward_matrix[:, 3]
        backward_matrix_3x3 = torch.linalg.inv(forward_matrix_3x3)
        backward_translation = -forward_translation
        backward_matrix_3x3 = backward_matrix_3x3.to(device)
        backward_translation = backward_translation.to(device)
        
        # (3, 3) -> (1, 1, 1, 3, 3)
        # (z, y, x, 3) -> (z, y, x, 3, 1)
        backward_matrix_3x3 = backward_matrix_3x3[(None,)*3] 
        tile_coords = torch.unsqueeze(tile_coords, 4)

        # Pytorch '@' Ref: https://discuss.pytorch.org/t/how-does-the-sign-work-in-this-instance/11232
        # (1, 1, 1, 3, 3) @ (z, y, x, 3, 1) -> (z, y, x, 3, 1)
        tile_coords = backward_matrix_3x3 @ tile_coords
        # (z, y, x, 3, 1) -> (1, z, y, x, 3)
        tile_coords = torch.movedim(tile_coords, source=4, destination=0)
        tile_coords = tile_coords + backward_translation

        # Calculate AABB of transformed coords
        z_min = torch.min(tile_coords[0, :, :, :, 0]).item()
        z_max = torch.max(tile_coords[0, :, :, :, 0]).item()
        
        y_min = torch.min(tile_coords[0, :, :, :, 1]).item()
        y_max = torch.max(tile_coords[0, :, :, :, 1]).item()
        
        x_min = torch.min(tile_coords[0, :, :, :, 2]).item()
        x_max = torch.max(tile_coords[0, :, :, :, 2]).item()
        LOGGER.info(f'AABB of transformed coords: {z_min}, {z_max}, {y_min}, {y_max}, {x_min}, {x_max}')

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
        crop_min_z = np.max(np.array([0, z_min]))
        crop_max_z = np.min(np.array([t_size_zyx[0], z_max]))
        
        crop_min_y = np.max(np.array([0, y_min]))
        crop_max_y = np.min(np.array([t_size_zyx[1], y_max]))
        
        crop_min_x = np.max(np.array([0, x_min]))
        crop_max_x = np.min(np.array([t_size_zyx[2], x_max]))

        # Make sure crop_{min, max}_{z, y, x} are integers to be used as indices. 
        # Minimum values are rounded down to nearest integer. 
        # Maximum values are rounded up to nearest integer.
        crop_min_z = int(np.floor(crop_min_z))
        crop_min_y = int(np.floor(crop_min_y))
        crop_min_x = int(np.floor(crop_min_x))

        crop_max_z = int(np.ceil(crop_max_z))
        crop_max_y = int(np.ceil(crop_max_y))
        crop_max_x = int(np.ceil(crop_max_x))

        LOGGER.info(f'AABB of Crop belonging to image {tile_id}: {crop_min_z}, {crop_max_z}, {crop_min_y}, {crop_max_y}, {crop_min_x}, {crop_max_x}')
        
        # Define tile coords wrt base image crop coordinates
        image_crop_offset = torch.Tensor([crop_min_z, crop_min_y, crop_min_x]).to(device)
        tile_coords = tile_coords - image_crop_offset

        # Prep inputs to interpolation
        image_crop_slice = (slice(0, 1), 
                            slice(0, 1), 
                            slice(crop_min_z, crop_max_z), 
                            slice(crop_min_y, crop_max_y), 
                            slice(crop_min_x, crop_max_x))
        image_crop = tile_arrays[tile_id][image_crop_slice]
        image_crop = image_crop.astype(np.int32)  # Promote uint16 -> Pytorch compatible int32
        log_image_crop = image_crop
        image_crop = torch.Tensor(image_crop).to(device)

        # Pytorch flow field follows a different basis than the image numpy basis.
        # Change of basis to interpolation basis, which preserves relative distances/angles/positions.  
        # tile_coords: (1, z, y, x, 3) 
        # (1, z, y, x, 3) -> (z, y, x, 3, 1)
        # (1, 1, 1, 3, 3) @ (z, y, x, 3, 1) -> (z, y, x, 3, 1) 
        # (z, y, x, 3, 1) -> (1, z, y, x, 3)
        tile_coords = torch.movedim(tile_coords, source=0, destination=-1)
        interp_CoB = torch.Tensor([[0, 0, 1], 
                                    [0, 1, 0], 
                                    [1, 0, 0]]).to(device)
        interp_CoB = interp_CoB[(None,)*3]
        tile_coords = interp_CoB @ tile_coords
        tile_coords = torch.movedim(tile_coords, source=-1, destination=0)

        # Interpolation expects 'grid' parameter/sample locations to be normalized [-1, 1].
        # Very specific per-dimension normalization according to CoB
        crop_z_length = crop_max_z - crop_min_z
        crop_y_length = crop_max_y - crop_min_y
        crop_x_length = crop_max_x - crop_min_x
        tile_coords[0, :, :, :, 0] = (tile_coords[0, :, :, :, 0] - (crop_x_length / 2)) / (crop_x_length / 2)
        tile_coords[0, :, :, :, 1] = (tile_coords[0, :, :, :, 1] - (crop_y_length / 2)) / (crop_y_length / 2)
        tile_coords[0, :, :, :, 2] = (tile_coords[0, :, :, :, 2] - (crop_z_length / 2)) / (crop_z_length / 2) 

        # Interpolate and Fuse
        # image_crop: (1, 1, z_in, y_in, x_in)
        # tile_coords: (1, z_out, y_out, x_out, 3)
        # => tile_contribution: (1, 1, z_out, y_out, x_out)
        # (1, z_out, y_out, x_out) -> (1, 1, z_out, y_out, x_out)
        tile_contribution = torch.nn.functional.grid_sample(image_crop, tile_coords, 
                                                            padding_mode='zeros', mode='nearest')
        fused_cell = torch.maximum(fused_cell, tile_contribution)

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


def main(xml_path: str, output_path: str): 
    # Geometry Utility
    def aabb(points: np.ndarray):
        """
        Parameters: 
        ----------
            points: Buffer of vec3's
        Returns
        ----------
            aabb: Ranges ordered in same order as components in input buffer.
        """
        dim_0_min = np.min(points[:, 0])
        dim_0_max = np.max(points[:, 0])

        dim_1_min = np.min(points[:, 1])
        dim_1_max = np.max(points[:, 1])

        dim_2_min = np.min(points[:, 2])
        dim_2_max = np.max(points[:, 2])

        return (dim_0_min, dim_0_max, dim_1_min, dim_1_max, dim_2_min, dim_2_max)

    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)

    # Parse XML
    tile_paths: dict[int, str] = dataloading.extract_tile_paths(xml_path)
    tile_transforms: dict[int, list[dict]] = dataloading.extract_tile_transforms(xml_path)
    net_transforms: dict[int, np.ndarray] = dataloading.calculate_net_transforms(tile_transforms)
    for t_id, t_path in tile_paths.items(): 
        tile_paths[t_id] = tile_paths[t_id] + '/0'

    # HARDCODED TRUNCATED RUN
    chosen_tiles = [4, 5]   # Meaty middle tiles
    c_tile_paths = {}
    c_net_transforms = {}
    for t_id in chosen_tiles:
        c_tile_paths[t_id] = tile_paths[t_id]
        c_net_transforms[t_id] = tile_transforms[t_id]
    tile_paths = c_tile_paths
    tile_transforms = c_net_transforms

    # Initalize Core Data Structures
    tile_arrays: dict[int, zarr.core.Array] = {}
    for tile_id, t_path in tile_paths.items():
        tile_arrays[tile_id] = zarr.open(t_path)

    AABB = tuple[int, int, int, int, int, int]
    tile_sizes_zyx: dict[int, tuple[int, int, int]] = {}
    tile_aabbs: dict[int, AABB] = {}

    tile_boundary_point_cloud_zyx = []
    for tile_id, t_path in tile_paths.items():
        zarray_path = t_path + '/.zarray'
        with open(zarray_path) as f:
            zarray_json = json.load(f)
        shape_tczyx = zarray_json["shape"]
        zyx = shape_tczyx[2:]
        tile_sizes_zyx[tile_id] = zyx

        tile_transform = net_transforms[tile_id]
        tile_boundaries = np.array([[0., 0., 0.], 
                        [zyx[0], 0., 0.],
                        [0., zyx[1], 0.],
                        [0., 0., zyx[2]],
                        [zyx[0], zyx[1], 0.],
                        [zyx[0], 0., zyx[2]],
                        [0., zyx[1], zyx[2]],
                        [zyx[0], zyx[1], zyx[2]]])
        tile_boundaries = np.concatenate((tile_boundaries, np.ones((tile_boundaries.shape[0], 1))), axis=1)
        tile_boundaries = tile_boundaries @ tile_transform.T
        
        tile_boundary_point_cloud_zyx.extend(tile_boundaries)
        tile_aabbs[tile_id] = aabb(tile_boundaries)
    tile_boundary_point_cloud_zyx = np.array(tile_boundary_point_cloud_zyx)

    # Shift AABB's into output volume and save this shift for later collision detection
    output_volume_origin = np.array([
        np.min(tile_boundary_point_cloud_zyx[:, 0]),
        np.min(tile_boundary_point_cloud_zyx[:, 1]),
        np.min(tile_boundary_point_cloud_zyx[:, 2])
    ])
    for tile_id, t_aabb in tile_aabbs.items():
        updated_aabb = (t_aabb[0] - output_volume_origin[0], t_aabb[1] - output_volume_origin[0],
                        t_aabb[2] - output_volume_origin[1], t_aabb[3] - output_volume_origin[1], 
                        t_aabb[4] - output_volume_origin[2], t_aabb[5] - output_volume_origin[2])
        tile_aabbs[tile_id] = updated_aabb 

    # Initalize Output Zarr Volume
    global_tile_boundaries = aabb(tile_boundary_point_cloud_zyx)
    OUTPUT_VOLUME_SIZE = (int(global_tile_boundaries[1] - global_tile_boundaries[0]), 
                          int(global_tile_boundaries[3] - global_tile_boundaries[2]),
                          int(global_tile_boundaries[5] - global_tile_boundaries[4]))
    LOGGER.info(f'{OUTPUT_VOLUME_SIZE=}')

    out_group = zarr.open_group(output_path, mode='w')
    path = "0"
    chunksize = (1, 1, 128, 256, 256)
    datatype = np.uint16
    dimension_separator = "/"
    compressor = None   # NOTE: May want to compress for a working script    
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

    # Initalize GPU conn
    available_gpus = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

    # Define cell dimensions and coordinates
    CELL_SIZE = (1024, 512, 512)
    z_stride = int(np.ceil(OUTPUT_VOLUME_SIZE[0] / CELL_SIZE[0]))
    y_stride = int(np.ceil(OUTPUT_VOLUME_SIZE[1] / CELL_SIZE[1]))
    x_stride = int(np.ceil(OUTPUT_VOLUME_SIZE[2] / CELL_SIZE[2]))
    est_total_cells = z_stride * y_stride * x_stride
    LOGGER.info(f'Estimated Total Cells: {est_total_cells}')

    # Define all work
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

    # Fill work buffer with inital tasks
    start_run = time.time()
    active_processes: list[tuple] = []
    for i in range(len(available_gpus)):
        p_args = process_args.pop(0)
        p = multiprocessing.Process(target=color_cell,
                                    args=(tile_arrays, net_transforms, tile_sizes_zyx, tile_aabbs, 
                                        output_volume, OUTPUT_VOLUME_SIZE, output_volume_origin,
                                        CELL_SIZE, p_args['z'], p_args['y'], p_args['x'], available_gpus[i],
                                        p_args['cell_num'], est_total_cells, LOGGER))
        active_processes.append((available_gpus[i], p))
        p.start()

    # Exhaust all the tasks
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
                                        args=(tile_arrays, net_transforms, tile_sizes_zyx, tile_aabbs, 
                                            output_volume, OUTPUT_VOLUME_SIZE, output_volume_origin,
                                            CELL_SIZE, p_args['z'], p_args['y'], p_args['x'], device,
                                            p_args['cell_num'], est_total_cells, LOGGER))
                    tmp.append((device, new_p))
                    new_p.start()

            active_processes = tmp

    LOGGER.info(f'Runtime: {time.time() - start_run}')