import torch

import aind_cloud_fusion.geometry as geometry

def check_collision(
    cell_box: geometry.AABB,
    t_aabb: geometry.AABB
) -> bool:
    """
    Check collision between two boxes.
    """
    return ((cell_box[1] > t_aabb[0] and cell_box[0] < t_aabb[1])
            and (cell_box[3] > t_aabb[2] and cell_box[2] < t_aabb[3])
            and (cell_box[5] > t_aabb[4] and cell_box[4] < t_aabb[5]))

def calculate_image_crop(cell_box: geometry.AABB,
                         output_volume_origin: tuple[float, float, float],
                         transform_list: list[geometry.Transform],
                         src_vol_shape_zyx: tuple[int, int, int],
                         device='cpu') -> tuple[slice]:
    """
    Inverse transforms cell_box boundary into source_vol space
    and returns an index-aligned, boundary-clipped crop in source_vol coordinates.
    Outputs a 5d-tuple of slices.

    IMPORTANT NOTE ON INPUTS:
    - cell_box is a AABB of index ranges, not absolute coordinates.

    - The expected order of transform_list is feedforward.
    We will apply these matrices in reverse!
    """

    # Derive cell_box extrema
    z_min, z_max, y_min, y_max, x_min, x_max = cell_box
    cell_box_pts = torch.Tensor(
        [
            [z_min, y_min, x_min],
            [z_min, y_min, x_max],
            [z_min, y_max, x_min],
            [z_min, y_max, x_max],
            [z_max, y_min, x_min],
            [z_max, y_min, x_max],
            [z_max, y_max, x_min],
            [z_max, y_max, x_max],
        ]
    )
    cell_box_pts = cell_box_pts.to(device)


    # Apply inverse transform
    cell_box_pts = cell_box_pts + torch.Tensor(output_volume_origin).to(
        device
    )
    for tfm in reversed(transform_list):
        cell_box_pts = tfm.backward(cell_box_pts, device=device)


    # Derive axis-aligned, boundary-clipped image
    cell_box_src: geometry.AABB = geometry.aabb_3d(cell_box_pts)
    z_min, z_max, y_min, y_max, x_min, x_max = cell_box_src

    sv_z, sv_y, sv_x = src_vol_shape_zyx
    aabb_src: geometry.AABB = (0, sv_z, 0, sv_y, 0, sv_x)


    # Validation
    if check_collision(cell_box_src, aabb_src) is False:
        raise ValueError("""Provided cell_box does not transform
                         into the provided source_volume.
                         Please check all inputs.""")

    # Calculate overlapping region between transformed coords and image boundary
    # For intervals (A, B):
    # The lower bound of overlapping region = max(A_min, B_min)
    # The upper bound of overlapping region = min(A_max, B_max)
    crop_min_z = torch.max(torch.Tensor([0, z_min]))
    crop_max_z = torch.min(torch.Tensor([sv_z, z_max]))

    crop_min_y = torch.max(torch.Tensor([0, y_min]))
    crop_max_y = torch.min(torch.Tensor([sv_y, y_max]))

    crop_min_x = torch.max(torch.Tensor([0, x_min]))
    crop_max_x = torch.min(torch.Tensor([sv_x, x_max]))

    image_crop_slice = (
        0,
        0,
        slice(crop_min_z, crop_max_z),
        slice(crop_min_y, crop_max_y),
        slice(crop_min_x, crop_max_x),
    )

    return image_crop_slice

def calculate_sample_field(cell_box: geometry.AABB,
                           output_volume_origin: tuple[float, float, float],
                           transform_list: list[geometry.Transform],
                           src_vol_shape_zyx: tuple[int, int, int],
                           device='cuda:0') -> torch.Tensor:
    """
    Inverse transforms indices in cell_box boundary into source_vol space
    and returns index-aligned, boundary-clipped samples in source_vol coordinates.

    Returned samples are properly normalized and in the pytorch basis,
    ready to input into grid_sample. Shape: (z, y, x, 3)

    IMPORTANT NOTE ON INPUTS:
    - cell_box is a AABB of index ranges, not absolute coordinates.

    - The expected order of transform_list is feedforward.
    We will apply these matrices in reverse!
    """

    # Init tile coords, arange end-exclusive, +0.5 to represent voxel center
    z_indices = torch.arange(cell_box[0], cell_box[1], step=1) + 0.5
    y_indices = torch.arange(cell_box[2], cell_box[3], step=1) + 0.5
    x_indices = torch.arange(cell_box[4], cell_box[5], step=1) + 0.5
    z_indices = z_indices.to(device)
    y_indices = y_indices.to(device)
    x_indices = x_indices.to(device)

    z_grid, y_grid, x_grid = torch.meshgrid(
        z_indices, y_indices, x_indices, indexing="ij"
    )
    z_grid = torch.unsqueeze(z_grid, 0)
    y_grid = torch.unsqueeze(y_grid, 0)
    x_grid = torch.unsqueeze(x_grid, 0)

    tile_coords = torch.concatenate((z_grid, y_grid, x_grid), axis=0)
    # (3, z, y, x) -> (z, y, x, 3)
    tile_coords = torch.movedim(tile_coords, source=0, destination=3)

    # Define tile coords wrt output vol origin
    tile_coords = tile_coords + torch.Tensor(output_volume_origin).to(
        device
    )

    # Send tile_coords through inverse transforms
    # NOTE: tile_transforms list must be iterated thru in reverse
    # (z, y, x, 3) -> (z, y, x, 3)
    for tfm in reversed(transform_list):
        tile_coords = tfm.backward(tile_coords, device=device)

    # Calculate AABB of transformed coords
    transformed_cell_box: geometry.AABB = geometry.aabb_3d(
        tile_coords
    )
    z_min, z_max, y_min, y_max, x_min, x_max = transformed_cell_box

    sv_z, sv_y, sv_x = src_vol_shape_zyx
    source_aabb: geometry.AABB = (0,
                                  sv_z,
                                  0,
                                  sv_y,
                                  0,
                                  sv_x)

    if check_collision(transformed_cell_box, source_aabb) is False:
        raise ValueError("""Provided cell_box does not transform
                         into the provided source_volume.
                         Please check all inputs.""")

    # Calculate overlapping region between transformed coords and image boundary
    # For intervals (A, B):
    # The lower bound of overlapping region = max(A_min, B_min)
    # The upper bound of overlapping region = min(A_max, B_max)
    crop_min_z = torch.max(torch.Tensor([0, z_min]))
    crop_max_z = torch.min(torch.Tensor([sv_z, z_max]))

    crop_min_y = torch.max(torch.Tensor([0, y_min]))
    crop_max_y = torch.min(torch.Tensor([sv_y, y_max]))

    crop_min_x = torch.max(torch.Tensor([0, x_min]))
    crop_max_x = torch.min(torch.Tensor([sv_x, x_max]))

    # Make sure crop_{min, max}_{z, y, x} are integers to be used as indices.
    # Minimum values are rounded down to nearest integer.
    # Maximum values are rounded up to nearest integer.
    crop_min_z = int(torch.floor(crop_min_z))
    crop_min_y = int(torch.floor(crop_min_y))
    crop_min_x = int(torch.floor(crop_min_x))

    crop_max_z = int(torch.ceil(crop_max_z))
    crop_max_y = int(torch.ceil(crop_max_y))
    crop_max_x = int(torch.ceil(crop_max_x))

    # Define tile coords wrt base image crop coordinates
    image_crop_offset = torch.Tensor(
        [crop_min_z, crop_min_y, crop_min_x]
    ).to(device)
    tile_coords = tile_coords - image_crop_offset

    # Pytorch flow field follows a different basis than the image numpy basis.
    # Change of basis to interpolation basis, which preserves relative distances/angles/positions.
    # (z, y, x, 3) -> (z, y, x, 3)
    interp_cob_matrix = torch.Tensor(
        [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    )
    interp_cob = geometry.Affine(interp_cob_matrix)
    tile_coords = interp_cob.forward(tile_coords, device=device)

    # Interpolation expects 'grid' parameter/sample locations to be normalized [-1, 1].
    # Very specific per-dimension normalization according to CoB
    crop_z_length = crop_max_z - crop_min_z
    crop_y_length = crop_max_y - crop_min_y
    crop_x_length = crop_max_x - crop_min_x
    tile_coords[:, :, :, 0] = (
        tile_coords[:, :, :, 0] - (crop_x_length / 2)
    ) / (crop_x_length / 2)
    tile_coords[:, :, :, 1] = (
        tile_coords[:, :, :, 1] - (crop_y_length / 2)
    ) / (crop_y_length / 2)
    tile_coords[:, :, :, 2] = (
        tile_coords[:, :, :, 2] - (crop_z_length / 2)
    ) / (crop_z_length / 2)

    return tile_coords

def interpolate(image_crop: torch.Tensor,
                sample_field: torch.Tensor,
                device="cuda:0") -> torch.Tensor:
    """
    Execute grid_sample and returns 5d tile_sample.

    Notes:
    - image_crop shape is (z_img, y_img, x_img)
    - sample_field shape is (z, y, x, 3)
    - tile_sample shape is (1, 1, z, y, x)
    """

    # Final reshaping
    # image_crop: (z_in, y_in, x_in) -> (1, 1, z_in, y_in, x_in)
    # tile_coords: (z_out, y_out, x_out, 3) -> (1, z_out, y_out, x_out, 3)
    # => tile_contribution: (1, 1, z_out, y_out, x_out)
    image_crop = image_crop[(None,) * 2]
    sample_field = torch.unsqueeze(sample_field, 0)

    # Interpolate
    image_crop = image_crop.to(device)
    sample_field = sample_field.to(device)
    tile_sample = torch.nn.functional.grid_sample(
        image_crop,
        sample_field,
        padding_mode="zeros",
        mode="nearest",
        align_corners=False,
    )

    return tile_sample
