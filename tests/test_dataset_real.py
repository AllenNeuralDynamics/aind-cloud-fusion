"""Mock Dataset Generation."""

import dask.array as da
import numpy as np
import zarr

import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.io as io


class TestDatasetReal(io.Dataset):
    """
    Formats real data as Dataset application input.
    """

    def __init__(self):
        pass

    @property
    def tile_volumes_tczyx(self) -> dict[int, io.InputArray]:
        tile_5 = "tests/real_data/tile_5/2/"
        tile_6 = "tests/real_data/tile_6/2/"

        tile_volumes = {5: da.from_zarr(tile_5), 6: da.from_zarr(tile_6)}
        return tile_volumes

    @property
    def tile_transforms_zyx(self) -> dict[int, list[geometry.Transform]]:
        tile_5_transforms = [
            geometry.Affine(
                np.array(
                    [
                        [1.0, 0.0, 0.0, 93.0],
                        [0.0, 1.0, 0.0, 1851.0],
                        [0.0, 0.0, 1.0, 5554.0],
                    ]
                )
            ),
            geometry.Affine(
                np.array(
                    [
                        [1.0, 0.0, 0.0, 4.34549],
                        [0.0, 1.0, 0.0, 3.67476],
                        [0.0, 0.0, 1.0, -65.9757],
                    ]
                )
            ),
        ]
        tile_6_transforms = [
            geometry.Affine(
                np.array(
                    [
                        [1.0, 0.0, 0.0, 114.0],
                        [0.0, 1.0, 0.0, 1851.0],
                        [0.0, 0.0, 1.0, 3703.0],
                    ]
                )
            ),
            geometry.Affine(
                np.array(
                    [
                        [1.0, 0.0, 0.0, 1.78407],
                        [0.0, 1.0, 0.0, -16.18333],
                        [0.0, 0.0, 1.0, -46.52276],
                    ]
                )
            ),
        ]
        # ^^^Reference net transforms:
        # tile_5_net_zyx: [97, 1854, 5489]
        # tile_6_net_zyx: [116, 1853, 3657]

        P_inverse = geometry.Affine(
            np.array(
                [
                    [4.0, 0.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 0.0],
                ]
            )
        )
        P = geometry.Affine(
            np.array(
                [
                    [0.25, 0.0, 0.0, 0.0],
                    [0.0, 0.25, 0.0, 0.0],
                    [0.0, 0.0, 0.25, 0.0],
                ]
            )
        )
        tile_5_transforms.insert(0, P_inverse)
        tile_5_transforms.append(P)

        tile_6_transforms.insert(0, P_inverse)
        tile_6_transforms.append(P)

        tile_transforms = {5: tile_5_transforms, 6: tile_6_transforms}
        return tile_transforms

    @property
    def tile_resolution_zyx(self) -> tuple[float, float, float]:
        tile_resolution_zyx = [1.0, 0.256, 0.256]
        return tile_resolution_zyx


def generate_real_dataset():
    ground_truth = zarr.open(
        "tests/real_data/fused.zarr/0"
    )  # Full resolution was fused from downsampled inputs.
    dataset = TestDatasetReal()

    return ground_truth, dataset


if __name__ == "__main__":
    # Prep application parameters
    DATASET = TestDatasetReal()
    zarr_path = str("tests/real_data/fused.zarr")
    OUTPUT_PARAMS = io.OutputParameters(
        path=zarr_path,
        chunksize=(1, 1, 100, 100, 100),
        resolution_zyx=(1.0, 0.256, 0.256),  # Preserving original resolution.
    )
    RUNTIME_PARAMS = io.RuntimeParameters(
        option=0, pool_size=16, worker_cells=[]
    )
    POST_REG_TFMS = []
    CELL_SIZE = [100, 100, 100]

    # Init and Run Fusion
    worker_cells = []
    _, _, _, tile_aabbs, output_volume_size, _ = fusion.initialize_fusion(
        DATASET, POST_REG_TFMS, OUTPUT_PARAMS
    )
    z_cnt, y_cnt, x_cnt = fusion.get_cell_count_zyx(
        output_volume_size, CELL_SIZE
    )
    for z in range(z_cnt):
        for y in range(y_cnt):
            for x in range(x_cnt):
                worker_cells.append((z, y, x))
    RUNTIME_PARAMS.worker_cells = worker_cells

    tile_layout = [[5, 6]]  # Tile 5 and 6 are adjacent in x.
    BLENDING_MODULE = blend.WeightedLinearBlending(
        tile_layout=tile_layout, tile_aabbs=tile_aabbs
    )

    fusion.run_fusion(
        DATASET,
        OUTPUT_PARAMS,
        RUNTIME_PARAMS,
        CELL_SIZE,
        POST_REG_TFMS,
        BLENDING_MODULE,
    )
