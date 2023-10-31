import numpy as np
import torch

import aind_cloud_fusion.io as io
import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.fusion as fusion

def initalize_compute_node(yml_file: str): 
    """
    'Factory' method. 
    Initalize corresponding compute node
    to what is specified in configuration file. 
    """
    params = io.read_config_yaml(yml_file)
    
    node: ComputeNode = None
    if params['runtime']['distributed'] is True: 
        if 'scheduler' in params['runtime']:
            node = DistributedScheduler(yml_file)

        elif 'worker' in params['runtime']:
            node = DistributedWorker(yml_file)
    else: 
        node = SoloNode(yml_file)

    return node


class ComputeNode: 
    """
    Compute Primitive in Distributed Fusion Run. 
    """
    def __init__(self, config_yaml: str): 
        """
        Parses configuration file into application-specific primitives. 
        """

        params = io.read_config_yaml(config_yaml)

        self.DATASET = None
        dataset_type = params['input']['dataset_type']
        if dataset_type == 'big_stitcher':
            xml_path = params['dataset_parameters']['big_stitcher']['xml_path']
            self.DATASET = io.BigStitcherDataset(xml_path)

        self.OUTPUT_PARAMS = io.OutputParameters(path=params['output']['path'],
                                        chunksize=tuple(params['output']['chunksize']), 
                                        resolution_zyx=tuple(params['output']['resolution_zyx']))

        self.DEVICES = []
        if params['algorithm_parameters']['use_gpus']:
            DEVICES = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        else: 
            DEVICES = [torch.device('cpu')]

        chunk_size = params['output']['chunksize']
        self.CELL_SIZE = params['algorithm_parameters']['cell_size']
        assert (CELL_SIZE[0] % chunk_size[0] == 0) and \
            (CELL_SIZE[1] % chunk_size[1] == 0) and \
            (CELL_SIZE[2] % chunk_size[2] == 0), \
            f'Cell size {CELL_SIZE} is not a multiple of chunksize {chunk_size}. 
                Please update configuration file.'

        self.BLENDING_MODULE = None
        if params['algorithm_parameters']['blending_module'] == 'MaxProjection': 
            self.BLENDING_MODULE = blend.MaxProjection()
        
        self.POST_REG_TFMS: list[geometry.Affine] = [] 
        for tfm in params['algorithm_parameters']["post_registration_transforms_zyx"]: 
            self.POST_REG_TFMS.append(geometry.Affine(np.array(tfm)))

    def run(self): 
        raise NotImplementedError('"Please implement in DistributedNode subclass."')


class DistributedWorker(ComputeNode):
    """
    Defines context for worker in distributed run. 
    """

    def __init__(self, config_yaml: str):
        """
        Load distributed-worker-specific parameters.
        """

        super().__init(config_yaml)
        params = io.read_config_yaml(config_yaml)
        self.WORKER_CELLS = params['worker']['worker_cells']
    
    def run(self):
        """
        Run fusion on worker cells defined in configuration file. 
        """

        fusion.run_fusion(self.DATASET, 
                        self.OUTPUT_PARAMS,
                        self.DEVICES, 
                        self.CELL_SIZE, 
                        self.POST_REG_TFMS,
                        self.BLENDING_MODULE,
                        self.WORKER_CELLS)


class DistributedScheduler(ComputeNode): 
    """
    Defines context for scheduler in distributed run. 
    """

    def __init__(self, config_yaml: str):
        """
        Load distributed-scheduler-specific parameters.
        """

        super().__init__(config_yaml)
        params = io.read_config_yaml(config_yaml)
        self.worker_yml_path = params['runtime']['scheduler']['worker_yml_path']
        self.num_workers = params['runtime']['scheduler']['num_workers']
        

    def run(self):
        """
        Outputs worker configuration files into specifed path. 
        """

        # Scheduler-specific initalization. 
        # Run fusion initialization to find output volume dimensions. 
        tile_transforms: dict[int, list[Transform]] = self.DATASET.tile_transforms_zyx
        input_resolution_zyx: tuple[float, float, float] = self.DATASET.tile_resolution_zyx
        iz, iy, ix = input_resolution_zyx
        scale_input_zyx = geometry.Affine(np.array([[iz, 0, 0, 0], 
                                                    [0, iy, 0, 0], 
                                                    [0, 0, ix, 0]]))
        output_resolution_zyx: tuple[float, float, float] = output_params.resolution_zyx
        oz, oy, ox = output_resolution_zyx
        sample_output_zyx = geometry.Affine(np.array([[1/oz, 0, 0, 0], 
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
            
            tfm_list = tile_transforms[tile_id]
            for i, tfm in enumerate(tfm_list): 
                tile_boundaries = tfm.forward(tile_boundaries, device=torch.device('cpu'))

            tile_aabbs[tile_id] = aabb_3d(tile_boundaries)
            tile_boundary_point_cloud_zyx.extend(tile_boundaries)
        tile_boundary_point_cloud_zyx = torch.stack(tile_boundary_point_cloud_zyx, dim=0)

        global_tile_boundaries = aabb_3d(tile_boundary_point_cloud_zyx)
        output_volume_size = (int(global_tile_boundaries[1] - global_tile_boundaries[0]), 
                            int(global_tile_boundaries[3] - global_tile_boundaries[2]),
                            int(global_tile_boundaries[5] - global_tile_boundaries[4]))

        # Define/Divide Work. 
        # Generate YAML files.
        z_cnt, y_cnt, x_cnt = fusion.get_cell_count_zyx(output_volume_size, self.CELL_SIZE)
        total_cells = z_cnt * y_cnt * x_cnt
        cell_per_worker = total_cells // self.num_workers

        params = io.read_config_yaml(config_yaml)
        del params['runtime']['scheduler']
        params['runtime']['worker'] = {}

        worker_num = 0
        curr_worker_cells = []
        for z in range(z_cnt):
            for y in range(y_cnt):
                for x in range(x_cnt):
                    if len(curr_worker_cells) == cell_per_worker:
                        # Publish Yaml File, Reset State
                        params['runtime']['worker']['worker_cells'] = curr_worker_cells
                        io.write_config_yaml(yaml_path=f'worker_config_{worker_num}.yaml',
                                             yaml_data=params)

                        curr_worker_cells = []
                        params['runtime']['worker'] = {}
                        worker_num += 1

                    curr_worker_cells.append((z, y, x))
        
        # Publish remaining state into last YAML file
        params['runtime']['worker']['worker_cells'] = curr_worker_cells
        io.write_config_yaml(yaml_path=f'worker_config_{worker_num}.yaml',
                                yaml_data=params)


class SoloNode(ComputeNode):
    """
    Defines context for full-dataset fusion on a single machine.
    """

    def __init__(self, config_yaml: str): 
        """
        Same as parent constructor. 
        """
        super.__init__(config_yaml)

    def run(self):
        """
        Define all work across entire dataset and 
        pass to fusion. (implicit scheduler + worker combined)
        """

        # To fill with all cells
        self.WORKER_CELLS = []
        
        # Scheduler-specific initalization. 
        # Run fusion initialization to find output volume dimensions. 
        tile_transforms: dict[int, list[Transform]] = self.DATASET.tile_transforms_zyx
        input_resolution_zyx: tuple[float, float, float] = self.DATASET.tile_resolution_zyx
        iz, iy, ix = input_resolution_zyx
        scale_input_zyx = geometry.Affine(np.array([[iz, 0, 0, 0], 
                                                    [0, iy, 0, 0], 
                                                    [0, 0, ix, 0]]))
        output_resolution_zyx: tuple[float, float, float] = output_params.resolution_zyx
        oz, oy, ox = output_resolution_zyx
        sample_output_zyx = geometry.Affine(np.array([[1/oz, 0, 0, 0], 
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
            
            tfm_list = tile_transforms[tile_id]
            for i, tfm in enumerate(tfm_list): 
                tile_boundaries = tfm.forward(tile_boundaries, device=torch.device('cpu'))

            tile_aabbs[tile_id] = aabb_3d(tile_boundaries)
            tile_boundary_point_cloud_zyx.extend(tile_boundaries)
        tile_boundary_point_cloud_zyx = torch.stack(tile_boundary_point_cloud_zyx, dim=0)

        global_tile_boundaries = aabb_3d(tile_boundary_point_cloud_zyx)
        output_volume_size = (int(global_tile_boundaries[1] - global_tile_boundaries[0]), 
                            int(global_tile_boundaries[3] - global_tile_boundaries[2]),
                            int(global_tile_boundaries[5] - global_tile_boundaries[4]))

        z_cnt, y_cnt, x_cnt = fusion.get_cell_count_zyx(output_volume_size, self.CELL_SIZE)
        for z in range(z_cnt):
            for y in range(y_cnt):
                for x in range(x_cnt):
                    self.WORKER_CELLS.append((z, y, x))

        fusion.run_fusion(self.DATASET, 
                        self.OUTPUT_PARAMS,
                        self.DEVICES, 
                        self.CELL_SIZE, 
                        self.POST_REG_TFMS,
                        self.BLENDING_MODULE,
                        self.WORKER_CELLS)