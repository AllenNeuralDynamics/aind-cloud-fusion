"""
Defines all standard input to fusion algorithm.
"""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import boto3
import dask.array as da
import numpy as np
from numcodecs import Blosc
import re
import s3fs
import torch
import xmltodict
import yaml
import zarr

import aind_cloud_fusion.geometry as geometry


def read_config_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


def write_config_yaml(yaml_path: str, yaml_data: dict) -> None:
    with open(yaml_path, "w") as file:
        yaml.dump(yaml_data, file)


class LazyArray:
    def __getitem__(self, value):
        """
        Member function for slice syntax, ex: arr[0:10, 0:10]
        Value is a Python slice object.
        """
        raise NotImplementedError("Please implement in LazyArray subclass.")

    @property
    def shape(self):
        raise NotImplementedError("Please implement in LazyArray subclass.")


class ZarrArray(LazyArray):
    def __init__(self, arr: da.Array):
        self.arr = arr

    def __getitem__(self, slice):
        return self.arr[slice].compute()

    @property
    def shape(self):
        return self.arr.shape


class Dataset:
    """
    Data are 5d tczyx objects. 
    Transforms are 3d zyx objects.
    """

    class WriteError(Exception):
        pass

    @property
    def tile_volumes_tczyx(self) -> dict[int, LazyArray]:
        """
        Dict of tile_id -> tile references.
        """
        raise NotImplementedError("Please implement in Dataset subclass.")

    @tile_volumes_tczyx.setter
    def tile_volumes_tczyx(self, value):
        raise Dataset.WriteError("tile_volumes_tczyx is read-only.")

    @property
    def tile_transforms_zyx(self) -> dict[int, list[geometry.Transform]]:
        """
        Dict of tile_id -> tile transforms.
        """
        raise NotImplementedError("Please implement in Dataset subclass.")

    @tile_transforms_zyx.setter
    def tile_transforms_zyx(self, value):
        raise Dataset.WriteError("tile_transforms_zyx is read-only.")

    @property
    def tile_resolution_zyx(self) -> tuple[float, float, float]:
        """
        Specifies absolute size of each voxel in tile volume.
        Tile resolution is used to scale tile volume into absolute space prior to registration.
        """
        raise NotImplementedError("Please implement in Dataset subclass.")

    @tile_resolution_zyx.setter
    def tile_resolution_zyx(self, value):
        raise Dataset.WriteError("tile_resolution_zyx is read-only.")


class BigStitcherDataset(Dataset):
    def __init__(self, xml_path: str, s3_path: str):
        self.xml_path = xml_path
        self.s3_path = s3_path

    @property
    def tile_volumes_tczyx(self) -> dict[int, LazyArray]:
        tile_paths = self._extract_tile_paths(self.xml_path)
        for t_id, t_path in tile_paths.items(): 
            if not self.s3_path.endswith('/'):
                self.s3_path = self.s3_path + '/'
            tile_paths[t_id] = self.s3_path + Path(t_path).name + '/0'
            
        tile_arrays: dict[int, LazyArray] = {}
        for tile_id, t_path in tile_paths.items():            
            print(f'Loading Tile {tile_id} / {len(tile_paths)}')

            tile_zarr = da.from_zarr(t_path)
            
            # Replacing this with an s3fs with higher read concurrency. 
            s3 = s3fs.S3FileSystem(
            config_kwargs={
                    'max_pool_connections': 50,
                    's3': {
                      'max_concurrent_requests': 20  # Increased 10->20.
                    }, # Compute instances are in-network of s3 buckets. 
                    'retries': {
                      'total_max_attempts': 100,
                      'mode': 'adaptive',
                    }
                }
            )
            store = s3fs.S3Map(root=t_path, s3=s3)
            in_group = zarr.open(store=store, mode='r')       
            tile_zarr = da.from_zarr(in_group)
            
            # tile_zarr_zyx = tile_zarr[0, 0, :, :, :]
            # tile_arrays[int(tile_id)] = ZarrArray(tile_zarr_zyx)
            # ^Although not computed, causes a large task graph. 

            tile_arrays[int(tile_id)] = ZarrArray(tile_zarr)

        return tile_arrays

    @property
    def tile_transforms_zyx(self) -> dict[int, list[geometry.Transform]]:
        tile_tfms = self._extract_tile_transforms(self.xml_path)
        tile_net_tfms = self._calculate_net_transforms(tile_tfms)

        for tile_id, tfm in tile_net_tfms.items():
            # BigStitcher XYZ -> ZYX
            # Given Matrix 3x4:
            # Swap Rows 0 and 2; Swap Colums 0 and 2
            tmp = np.copy(tfm)
            tmp[[0, 2], :] = tmp[[2, 0], :]
            tmp[:, [0, 2]] = tmp[:, [2, 0]]
            tfm = tmp

            # Pack into list
            tile_net_tfms[int(tile_id)] = [geometry.Affine(tfm)]

        return tile_net_tfms

    @property
    def tile_resolution_zyx(self) -> tuple[float, float, float]:
        with open(self.xml_path, "r") as file:
            data: OrderedDict = xmltodict.parse(file.read())

        resolution_str = data["SpimData"]["SequenceDescription"]["ViewSetups"][
            "ViewSetup"
        ][0]["voxelSize"]["size"]
        resolution_xyz = [float(num) for num in resolution_str.split(" ")]
        return tuple(resolution_xyz[::-1])

    def _extract_tile_paths(self, xml_path: str) -> dict[int, str]:
        """
        Utility called in property.
        Parses BDV xml and outputs map of setup_id -> tile path.

        Parameters
        ------------------------
        xml_path: str
            Path of xml outputted from BigStitcher.

        Returns
        ------------------------
        dict[int, str]:
            Dictionary of tile ids to tile paths.
        """
        view_paths: dict[int, str] = {}
        with open(xml_path, "r") as file:
            data: OrderedDict = xmltodict.parse(file.read())

        parent = data["SpimData"]["SequenceDescription"]["ImageLoader"][
            "zarr"
        ]["#text"]

        for i, zgroup in enumerate(
            data["SpimData"]["SequenceDescription"]["ImageLoader"]["zgroups"][
                "zgroup"
            ]
        ):
            view_paths[i] = parent + "/" + zgroup["path"]

        return view_paths

    def _extract_tile_transforms(self, xml_path: str) -> dict[int, list[dict]]:
        """
        Utility called in property.
        Parses BDV xml and outputs map of setup_id -> list of transformations
        Output dictionary maps view number to list of {'@type', 'Name', 'affine'}
        where 'affine' contains the transform as string of 12 floats.

        Matrices are listed in the order of forward execution.

        Parameters
        ------------------------
        xml_path: str
            Path of xml outputted by BigStitcher.

        Returns
        ------------------------
        dict[int, list[dict]]
            Dictionary of tile ids to transform list. List entries described above.
        """

        view_transforms: dict[int, list[dict]] = {}
        with open(xml_path, "r") as file:
            data: OrderedDict = xmltodict.parse(file.read())

        for view_reg in data["SpimData"]["ViewRegistrations"][
            "ViewRegistration"
        ]:
            tfm_stack = view_reg["ViewTransform"]
            if type(tfm_stack) is not list:
                tfm_stack = [tfm_stack]
            view_transforms[int(view_reg["@setup"])] = tfm_stack

        view_transforms = {
            view: tfs[::-1] for view, tfs in view_transforms.items()
        }

        return view_transforms

    def _calculate_net_transforms(
        self, view_transforms: dict[int, list[dict]]
    ) -> dict[int, geometry.Matrix]:
        """
        Utility called in property.
        Accumulate net transform and net translation for each matrix stack.
        Net translation =
            Sum of translation vectors converted into original nominal basis
        Net transform =
            Product of 3x3 matrices
        NOTE: Translational component (last column) is defined
            wrt to the DOMAIN, not codomain.
            Implementation is informed by this given.

        Parameters
        ------------------------
        view_transforms: dict[int, list[dict]]
            Dictionary of tile ids to transforms associated with each tile.

        Returns
        ------------------------
        dict[int, np.ndarray]:
            Dictionary of tile ids to net_transform.
        """

        identity_transform = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        net_transforms: dict[int, np.ndarray] = {}
        for tile_id in view_transforms:
            net_transforms[tile_id] = np.copy(identity_transform)

        for view, tfs in view_transforms.items():
            net_translation = np.zeros(3)
            net_matrix_3x3 = np.eye(3)
            curr_inverse = np.eye(3)

            for (
                tf
            ) in (
                tfs
            ):  # Tfs is a list of dicts containing transform under 'affine' key
                nums = [float(val) for val in tf["affine"].split(" ")]
                matrix_3x3 = np.array([nums[0::4], nums[1::4], nums[2::4]])
                translation = np.array(nums[3::4])

                net_translation = net_translation + (
                    curr_inverse @ translation
                )
                net_matrix_3x3 = matrix_3x3 @ net_matrix_3x3
                curr_inverse = np.linalg.inv(
                    net_matrix_3x3
                )  # Update curr_inverse

            net_transforms[view] = np.hstack(
                (net_matrix_3x3, net_translation.reshape(3, 1))
            )

        return net_transforms


class BigStitcherDatasetChannel(BigStitcherDataset):
    """
    Convenience Dataset class that reuses tile registrations, 
    tile shapes, and tile resolution across channels. 
    Tile volumes is overloaded with channel-specific data. 
    """
    
    def __init__(self, xml_path: str, s3_path: str, channel_num: int):
        """
        Only new information required is channel number.
        """
        super().__init__(xml_path, s3_path)
        self.channel_num = channel_num

    @property
    def tile_volumes_tczyx(self) -> dict[int, LazyArray]:
        """
        Load in channel-specific tiles.
        """
        tile_arrays: dict[int, LazyArray] = {}

        with open(self.xml_path, "r") as file:
            data: OrderedDict = xmltodict.parse(file.read())
        tile_id_lut = {}
        for zgroup in data['SpimData']['SequenceDescription']['ImageLoader']['zgroups']['zgroup']:
            tile_id = zgroup['@setup']
            tile_name = zgroup['path']
            s_parts = tile_name.split('_')
            location = (int(s_parts[2]), 
                        int(s_parts[4]), 
                        int(s_parts[6]))
            tile_id_lut[location] = int(tile_id)

        # Reference path: s3://aind-open-data/HCR_677594_2023-10-20_15-10-36/SPIM.ome.zarr/
        slash_2 = self.s3_path.find('/', self.s3_path.find('/') + 1)
        slash_3 = self.s3_path.find('/', self.s3_path.find('/', self.s3_path.find('/') + 1) + 1)
        bucket_name = self.s3_path[slash_2 + 1:slash_3]
        directory_path = self.s3_path[slash_3 + 1:]

        pattern = r'^[^_]+\.W\d+(_X_\d{4}_Y_\d{4}_Z_\d{4}_ch_\d+)\.zarr$'
        for p in self._list_bucket_directory(bucket_name, directory_path):
            if p.endswith('.zgroup'):
                continue

            # Validation
            # result = re.match(pattern, p)
            # assert result, \
            #     f"""Data directory {p} does not follow file convention:
            #     <tile_name, no underscores>_X_####_Y_####_Z_####_ch_###.zarr
            #     """
            
            # Data loading
            match = None
            channel_num = 0
            pattern = r'(\d{3})\.zarr$'
            match = re.search(pattern, p) 

            if match: 
                channel_num = int(match.group(1))
                
                
            if channel_num == self.channel_num:
                full_resolution_p = self.s3_path + p + '/0'
            
                # Replacing this with an s3fs with higher read concurrency. 
                s3 = s3fs.S3FileSystem(
                config_kwargs={
                        'max_pool_connections': 50,
                        's3': {
                          'max_concurrent_requests': 20  # Increased 10->20.
                        }, # Compute instances are in-network of s3 buckets. 
                        'retries': {
                          'total_max_attempts': 100,
                          'mode': 'adaptive',
                        }
                    }
                )
                store = s3fs.S3Map(root=full_resolution_p, s3=s3)
                in_group = zarr.open(store=store, mode='r')       
                tile_zarr = da.from_zarr(in_group)

                s_parts = p.split('_')
                location = (int(s_parts[2]), 
                            int(s_parts[4]), 
                            int(s_parts[6]))
                tile_id = tile_id_lut[location]

                # tile_zarr_zyx = tile_zarr[0, 0, :, :, :]
                # tile_arrays[int(tile_id)] = ZarrArray(tile_zarr_zyx)
                # ^Although not computed, causes a large task graph.                 
                tile_arrays[int(tile_id)] = ZarrArray(tile_zarr)
                
        return tile_arrays
                
    def _list_bucket_directory(self, bucket_name: str, directory_path: str):
        client = boto3.client("s3")
        result = client.list_objects(
            Bucket=bucket_name, Prefix=directory_path, Delimiter="/"
        )

        paths = []   # These are paths
        for o in result.get("CommonPrefixes"):
            paths.append(o.get("Prefix"))

        # Parse the ending files from the paths
        files = []
        for p in paths: 
            if p.endswith('/'): 
                p = p.rstrip("/")  # Remove trailing slash from directories
            
            parts = p.split('/')
            files.append(parts[-1])

        return files


@dataclass
class OutputParameters:
    path: str
    chunksize: tuple[int, int, int, int, int]
    resolution_zyx: tuple[float, float, float]
    dtype: np.dtype = np.uint16
    dimension_separator: str = "/"
    compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)


class RuntimeParameters:
    def __init__(
        self,
        use_gpus: bool,
        devices: list[torch.cuda.device],
        pool_size: int,
        worker_cells: list[tuple[int, int, int]] = [],
    ):
        self.use_gpus = use_gpus
        self.devices = devices
        self.pool_size = pool_size
        self.worker_cells = worker_cells
