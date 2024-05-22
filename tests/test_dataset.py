"""Mock Dataset Generation."""
import numpy as np
from PIL import Image

import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.io as io

class TestDataset(io.Dataset):
    """
    Formats synthetic data as Dataset application input.
    """

    def __init__(
        self,
        tile_1_zyx: np.ndarray,
        tile_2_zyx: np.ndarray,
        known_transform_zyx: geometry.Matrix,
        input_resolution_zyx: tuple[float, float, float],
    ):
        self.tile_1_zyx = tile_1_zyx
        self.tile_2_zyx = tile_2_zyx
        self.known_transform_zyx = known_transform_zyx
        self.input_resolution_zyx = input_resolution_zyx

    @property
    def tile_volumes_tczyx(self) -> dict[int, io.InputArray]:
        tile_volumes = {0: self.tile_1_zyx, 1: self.tile_2_zyx}
        return tile_volumes

    @property
    def tile_transforms_zyx(self) -> dict[int, list[geometry.Transform]]:
        tile_transforms = {
            0: [
                geometry.Affine(
                    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
                )
            ],
            1: [geometry.Affine(self.known_transform_zyx)],
        }
        return tile_transforms

    @property
    def tile_shapes_zyx(self) -> dict[int, tuple[int, int, int]]:
        tile_shapes = {0: self.tile_1_zyx.shape, 1: self.tile_2_zyx.shape}
        return tile_shapes

    @property
    def tile_resolution_zyx(self) -> tuple[float, float, float]:
        return self.input_resolution_zyx


def generate_z_max_proj_dataset() -> tuple[np.ndarray, io.Dataset]:
    """
    200x200 image stacked to 400 in 3rd dimension.
    Each image is 150x200 with a 50 pixel overlap.
    """
    img = Image.open("tests/nueral_dynamics_logo.jpeg").convert("L")
    img = np.array(img)
    x, y = img.shape
    stack_size = 400

    ground_truth = np.zeros((x, y, stack_size))
    for i in range(stack_size):
        ground_truth[:, :, i] = img
    tile_1_zyx = ground_truth[: 3 * (x // 4), :, :].copy()
    tile_2_zyx = ground_truth[(x // 4) :, :, :].copy()

    # Erase some signal in the overlap region to test blending,
    # Erasing alternating stripes.
    s = tile_1_zyx.shape[0]  # Split axis.
    tile_1_zyx[(s // 2) :, 0::2, :] = 0  # Stripes in bottom half
    tile_2_zyx[0 : (s // 2), 1::2, :] = 0  # Stripes in upper half

    # Registration matrix is (identity, translation = to tile cut).
    registration_zyx = np.array(
        [[1, 0, 0, (x // 4)], [0, 1, 0, 0], [0, 0, 1, 0]]  # Split axis
    )

    input_resolution_zyx = (1.0, 1.0, 1.0)

    # Reshape from 3D -> 5D
    tile_1_tczyx = tile_1_zyx[np.newaxis, np.newaxis, ...]
    tile_2_tczyx = tile_2_zyx[np.newaxis, np.newaxis, ...]

    dataset = TestDataset(
        tile_1_tczyx, tile_2_tczyx, registration_zyx, input_resolution_zyx
    )

    return ground_truth, dataset

def generate_y_max_proj_dataset() -> tuple[np.ndarray, io.Dataset]:
    """
    200x200 image stacked to 400 in 3rd dimension.
    Each image is 150x200 with a 50 pixel overlap.
    """
    img = Image.open("tests/nueral_dynamics_logo.jpeg").convert("L")
    img = np.array(img)
    x, y = img.shape
    stack_size = 400

    ground_truth = np.zeros((stack_size, x, y))
    for i in range(stack_size):
        ground_truth[i, :, :] = img
    tile_1_zyx = ground_truth[:, : 3 * (x // 4), :].copy()
    tile_2_zyx = ground_truth[:, (x // 4) :, :].copy()

    # Erase some signal in the overlap region to test blending,
    # Erasing alternating stripes.
    s = tile_1_zyx.shape[1]  # Split axis.
    tile_1_zyx[:, (s // 2) :, 0::2] = 0  # Stripes in bottom half
    tile_2_zyx[:, 0 : (s // 2), 1::2] = 0  # Stripes in upper half

    # Registration matrix is (identity, translation = to tile cut).
    registration_zyx = np.array(
        [[1, 0, 0, 0], [0, 1, 0, (x // 4)], [0, 0, 1, 0]]  # Split axis
    )

    input_resolution_zyx = (1.0, 1.0, 1.0)

    # Reshape from 3D -> 5D
    tile_1_tczyx = tile_1_zyx[np.newaxis, np.newaxis, ...]
    tile_2_tczyx = tile_2_zyx[np.newaxis, np.newaxis, ...]

    dataset = TestDataset(
        tile_1_tczyx, tile_2_tczyx, registration_zyx, input_resolution_zyx
    )

    return ground_truth, dataset

def generate_x_max_proj_dataset() -> tuple[np.ndarray, io.Dataset]:
    """
    200x200 image stacked to 400 in 3rd dimension.
    Each image is 150x200 with a 50 pixel overlap.
    """
    img = Image.open("tests/nueral_dynamics_logo.jpeg").convert("L")
    img = np.array(img)
    x, y = img.shape
    stack_size = 400

    ground_truth = np.zeros((x, stack_size, y))

    for i in range(stack_size):
        ground_truth[:, i, :] = img
    tile_1_zyx = ground_truth[:, :, : 3 * (x // 4)].copy()
    tile_2_zyx = ground_truth[:, :, (x // 4) :].copy()

    # Erase some signal in the overlap region to test blending
    s = tile_1_zyx.shape[2]  # Split axis.
    tile_1_zyx[0::2, :, (s // 2) :] = 0  # Stripes in right half
    tile_2_zyx[1::2, :, 0 : (s // 2)] = 0  # Stripes in left half

    # Registration matrix is (identity, translation = to tile cut).
    registration_zyx = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, (x // 4)]]
    )  # Split Axis

    input_resolution_zyx = (1.0, 1.0, 1.0)

    # Reshape from 3D -> 5D
    tile_1_tczyx = tile_1_zyx[np.newaxis, np.newaxis, ...]
    tile_2_tczyx = tile_2_zyx[np.newaxis, np.newaxis, ...]

    dataset = TestDataset(
        tile_1_tczyx, tile_2_tczyx, registration_zyx, input_resolution_zyx
    )

    return ground_truth, dataset

def generate_y_lin_blend_dataset() -> tuple[np.ndarray, io.Dataset]:
    """This dataset is padded such that each tile is a square. """

    img = Image.open("tests/nueral_dynamics_logo.jpeg").convert("L")
    img = np.array(img)
    y, x = img.shape

    stack_size = 400
    ground_truth = np.ones((stack_size, y + 100, x)) * 255  # Add padding upfront
    for i in range(stack_size):
        ground_truth[i, 50:-50, :] = img
    tile_1_zyx = ground_truth[:, 0:200, :].copy()
    tile_2_zyx = ground_truth[:, 100:300, :].copy()

    # Registration matrix is (identity, translation = to tile cut).
    registration_zyx = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 100], [0, 0, 1, 0]]
    ) 
    input_resolution_zyx = (1.0, 1.0, 1.0)

    # Reshape from 3D -> 5D
    tile_1_tczyx = tile_1_zyx[np.newaxis, np.newaxis, ...]
    tile_2_tczyx = tile_2_zyx[np.newaxis, np.newaxis, ...]

    dataset = TestDataset(
        tile_1_tczyx, tile_2_tczyx, registration_zyx, input_resolution_zyx
    )

    return ground_truth, dataset


def generate_x_lin_blend_dataset() -> tuple[np.ndarray, io.Dataset]:
    """This dataset is padded such that each tile is a square. """

    img = Image.open("tests/nueral_dynamics_logo.jpeg").convert("L")
    img = np.array(img)
    y, x = img.shape

    stack_size = 400
    ground_truth = np.ones((stack_size, y, x + 100)) * 255  # Add padding upfront
    for i in range(stack_size):
        ground_truth[i, :, 50:-50] = img
    tile_1_zyx = ground_truth[:, :, 0:200].copy()
    tile_2_zyx = ground_truth[:, :, 100:300].copy()

    # Registration matrix is (identity, translation = to tile cut).
    registration_zyx = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 100]]
    ) 
    input_resolution_zyx = (1.0, 1.0, 1.0)

    # Reshape from 3D -> 5D
    tile_1_tczyx = tile_1_zyx[np.newaxis, np.newaxis, ...]
    tile_2_tczyx = tile_2_zyx[np.newaxis, np.newaxis, ...]

    dataset = TestDataset(
        tile_1_tczyx, tile_2_tczyx, registration_zyx, input_resolution_zyx
    )

    return ground_truth, dataset
