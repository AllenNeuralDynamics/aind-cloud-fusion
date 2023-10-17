"""
Defines all standard input to fusion algorithm.
"""
from collections import OrderedDict, defaultdict
import numpy as np
import xmltodict
import yaml
import zarr

import geometry

def read_config_yaml(yaml_path: str = './config.yaml'):
    with open(yaml_path, 'r') as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict

class LazyArray:
    class WriteError(Exception):
        pass

    def __getitem__(self, value):
        """
        Member function for slice syntax, ex: arr[0:10, 0:10]
        Value is a Python slice object.
        """
        raise NotImplementedError("Please implement in LazyArray subclass.")

    @property
    def shape(self):
        raise NotImplementedError("Please implement in LazyArray subclass.")

    @shape.setter
    def shape(self, value):
        raise LazyArray.WriteError("shape is read-only.")


class ZarrArray(LazyArray):
    def __init__(self, arr: zarr.core.Array): 
        self.arr = arr

    def __getitem__(self, slice):
        return self.arr[slice]

    @property
    def shape(self):
        return self.arr.shape
    

class Dataset:
    class WriteError(Exception):
        pass

    @property
    def tile_volumes(self) -> list[LazyArray]:
        """
        List of tile references.
        """
        raise NotImplementedError("Please implement in Dataset subclass.")

    @tile_volumes.setter
    def tile_volumes(self, value): 
        raise Dataset.WriteError("tile_volumes is read-only.")

    @property
    def tile_transforms(self) -> list[geometry.Transform]:
        """
        Corresponding list of transforms in matching order.
        """
        raise NotImplementedError("Please implement in Dataset subclass.")

    @tile_transforms.setter
    def tile_transforms(self, value): 
        raise Dataset.WriteError("tile_transforms is read-only.")


class BigStitcherDataset(Dataset): 
    def __init__(self, xml_file: str): 
        self.xml_file = xml_file

    @property
    def tile_volumes(self) -> dict[int, LazyArray]:
        tile_paths = self._extract_tile_paths(self.xml_file)
        for t_id, t_path in tile_paths.items(): 
            tile_paths[t_id] = tile_paths[t_id] + '/0'

        tile_arrays: dict[int, LazyArray] = {}
        for tile_id, t_path in tile_paths.items():
            tile_zarr = zarr.open(t_path)
            tile_arrays[tile_id] = LazyArray(tile_zarr)

        return tile_arrays

    @property
    def tile_transforms(self) -> dict[int, geometry.Transform]:
        tile_tfms = self._extract_tile_transforms(self.xml_file)
        tile_net_tfms = self._calculate_net_transforms(tile_tfms)
        for tile_id, tfm in tile_net_tfms.items():
            tile_net_tfms[tile_id] = geometry.Affine(tfm)
        return tile_net_tfms

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

        parent = data["SpimData"]["SequenceDescription"]["ImageLoader"]["zarr"]['#text']

        for i, zgroup in enumerate(
            data["SpimData"]["SequenceDescription"]["ImageLoader"]["zgroups"]["zgroup"]
        ):
            view_paths[i] = parent + '/' + zgroup["path"]

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

        for view_reg in data["SpimData"]["ViewRegistrations"]["ViewRegistration"]:
            tfm_stack = view_reg["ViewTransform"]
            if type(tfm_stack) is not list:
                tfm_stack = [tfm_stack]
            view_transforms[int(view_reg["@setup"])] = tfm_stack

        view_transforms = {
            view: tfs[::-1] for view, tfs in view_transforms.items()
        }

        return view_transforms

    def _calculate_net_transforms(self, view_transforms: dict[int, list[dict]]
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
        net_transforms: dict[int, np.ndarray] = defaultdict(
            lambda: np.copy(identity_transform)
        )

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

                net_translation = net_translation + (curr_inverse @ translation)
                net_matrix_3x3 = matrix_3x3 @ net_matrix_3x3
                curr_inverse = np.linalg.inv(net_matrix_3x3)  # Update curr_inverse

            net_transforms[view] = np.hstack(
                (net_matrix_3x3, net_translation.reshape(3, 1))
            )

        return net_transforms