"""
Interface for generic blending.
"""
import numpy as np
import torch

from math import ceil, floor
from collections import defaultdict

import aind_cloud_fusion.geometry as geometry


class BlendingModule:
    """
    Minimal interface for modular blending function.
    Subclass can define arbitrary constructors/attributes/members as necessary.
    """

    def blend(self, 
              snowball_chunk: torch.Tensor, 
              chunks: list[torch.Tensor], 
              device: torch.device,
              kwargs = {}
    ) -> torch.Tensor:
        """
        snowball_chunk: base chunk to modify.
            Defining a snowball_chunk helps to amortize computation
            and limit peak memory usage. 
        chunks: 
            Chunks to blend into snowball_chunk
        kwargs: 
            Extra keyword arguments  
        """
        
        raise NotImplementedError(
            "Please implement in BlendingModule subclass."
        )


class MaxProjection(BlendingModule):
    """
    Simplest blending implementation possible. No constructor needed.
    """

    # NOTE: To prevent CPU/GPU OOM, recommended to pass in 1 chunk at a time.
    def blend(self, 
              snowball_chunk, 
              chunks: list[torch.Tensor], 
              device: torch.device,
              kwargs = {}
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        chunks: list of 3D tensors to combine. Contains 2 or more elements.

        Returns
        -------
        fused_chunk: combined chunk
        """

        assert (
            len(chunks) >= 1
        ), f"Length of input list is {len(chunks)}, blending requires 2 or more chunks."

        fused_chunk = torch.maximum(snowball_chunk.to(device), chunks[0].to(device))
        for c in chunks[1:]:
            c = c.to(device)
            fused_chunk = torch.maximum(fused_chunk, c)

        return fused_chunk


class MaskedBlending(BlendingModule): 
    """
    Defines a mask within tile overlap regions in which tiles are not blended.
    IMPORTANT: 
    Assumes tiles are arranged in a regular grid-- no significant warping of the image volume. 
    """
    
    def __init__(self, tile_aabbs: dict[int, geometry.AABB],
                       cell_size: tuple[int, int, int],
                       mask_axes: list[int], 
                       mask_percent: float = 0.9):
        """
        tile_aabbs: Dict of tile_id -> AABB
        cell_size: zyx cell size, application input
        mask_axes: axes to mask in. 
            0 = z-axis
            1 = y-axis
            2 = x-axis
            Ex: [0], [0, 1, 2]
        mask_percent: Percent of overlap region that is masked. 
            Within overlap region, mask starts with the minimum overlap value
            and progresses in the positive axis direction.  
            Ex: 
            x ------------>
              |OOOOOOO|--|
        """
        
        self.tile_aabbs = tile_aabbs
        self.cell_size = cell_size
        self.mask_percent = mask_percent
        
        # Create tile layout
        # Tile layout is a higher-level discrete representation 
        # of continuous AABB's. Defines relative position of tiles within a 3D array. 
        # Number of tiles in each axis determined by simple 1-D binning:
        t_ids = []
        points = []
        z_buffer = []
        y_buffer = []
        x_buffer = []
        for tile_id, t_aabb in tile_aabbs.items():
            min_z, _, min_y, _, min_x, _ = t_aabb
            min_z, min_y, min_x = int(min_z), int(min_y), int(min_x)
            t_ids.append(tile_id)
            points.append((min_z, min_y, min_x))
            z_buffer.append(min_z)
            y_buffer.append(min_y)
            x_buffer.append(min_x)

        z_clusters = self._cluster_1d(z_buffer)
        y_clusters = self._cluster_1d(y_buffer)
        x_clusters = self._cluster_1d(x_buffer)

        # Indices of tile layout = (tile_point - origin) / tile_stride
        origin = np.minimum(np.array(points))
        tile_stride = [1, 1, 1]
        if len(z_clusters) != 1:
            tile_stride[0] = z_clusters[0][0] - z_clusters[1][0]
        if len(y_clusters) != 1:
            tile_stride[1] = y_clusters[0][0] - y_clusters[1][0]
        if len(x_clusters) != 1:
            tile_stride[2] = x_clusters[0][0] - x_clusters[1][0]
        
        z_dim, y_dim, x_dim = len(z_clusters), len(y_clusters), len(x_clusters)
        TILE_LAYOUT = np.zeros((z_dim, y_dim, x_dim))
        print('f{TILE_LAYOUT.shape=}')
        for t_id, pt in zip(t_ids, points):
            z = round((pt[0] - origin[0]) / tile_stride[0])
            y = round((pt[1] - origin[1]) / tile_stride[1])
            x = round((pt[2] - origin[2]) / tile_stride[2])       
            TILE_LAYOUT[z, y, x] = t_id

        # Create mask data structures 
        # tile_to_mask_ids: Maps tile_id to associated mask_ids 
        # masks: Maps mask_id to mask AABB
        #        Mask AABB boundaries are rounded to a multiple cell_size.
        # block_list: Maps mask_id to blocked tile_ids
        # Access pattern: 
        # tile_id -> mask -> block_list
        self.tile_to_mask_ids: dict[int, list[int]] = defaultdict(list)
        self.masks: dict[int, geometry.AABB] = {}
        self.block_list: dict[int, list[int]] = defaultdict(list)

        mask_id = 0
        for mask_axis in mask_axes:
            # Create masks for z axis
            if mask_axis == 0:
                assert z_dim != 1, 'Number of tiles in Z axis is 1. Cannot mask in Z.'
                for i in range(z_dim - 1):
                    for j in range(y_dim):
                        for k in range(x_dim):
                            tid_1 = TILE_LAYOUT[i, j, k]
                            tid_2 = TILE_LAYOUT[i + 1, j, k]
                            tile_aabb_1 = self.tile_aabbs[tid_1]
                            tile_aabb_2 = self.tile_aabbs[tid_2]
                            mask_aabb = self._create_mask(tile_aabb_1, tile_aabb_2, 0)

                            # Update mask data structures
                            self.tile_to_mask_ids[tid_1].append(mask_id)
                            self.tile_to_mask_ids[tid_2].append(mask_id)
                            self.masks[mask_id] = mask_aabb
                            self.block_list[mask_id].append(tid_2)  # Only blocking tile2
                            mask_id += 1

            # Create masks for y axis
            if mask_axis == 1:
                assert y_dim != 1, 'Number of tiles in Y axis is 1. Cannot mask in Y.' 
                for i in range(z_dim):
                    for j in range(y_dim - 1):
                        for k in range(x_dim):
                            tid_1 = TILE_LAYOUT[i, j, k]
                            tid_2 = TILE_LAYOUT[i, j + 1, k]
                            tile_aabb_1 = self.tile_aabbs[tid_1]
                            tile_aabb_2 = self.tile_aabbs[tid_2]
                            mask_aabb = self._create_mask(tile_aabb_1, tile_aabb_2, 1)

                            # Update mask data structures
                            self.tile_to_mask_ids[tid_1].append(mask_id)
                            self.tile_to_mask_ids[tid_2].append(mask_id)
                            self.masks[mask_id] = mask_aabb
                            self.block_list[mask_id].append(tid_2)  # Only blocking tile2
                            mask_id += 1

            # Create masks for x axis
            if mask_axis == 2: 
                assert x_dim != 1, 'Number of tiles in X axis is 1. Cannot mask in X.'
                for i in range(z_dim):
                    for j in range(y_dim):
                        for k in range(x_dim - 1):
                            tid_1 = TILE_LAYOUT[i, j, k]
                            tid_2 = TILE_LAYOUT[i, j, k + 1]
                            tile_aabb_1 = self.tile_aabbs[tid_1]
                            tile_aabb_2 = self.tile_aabbs[tid_2]
                            mask_aabb = self._create_mask(tile_aabb_1, tile_aabb_2, 2)

                            # Update mask data structures
                            self.masks[mask_id] = mask_aabb
                            self.tile_to_mask_ids[tid_1].append(mask_id)
                            self.tile_to_mask_ids[tid_2].append(mask_id)
                            self.block_list[mask_id].append(tid_2)  # Only blocking tile2
                            mask_id += 1

    def _cluster_1d(self, buffer: list[int]) -> list[list[int]]:
        # Like a psuedo k-means
        # Source: https://stackoverflow.com/questions/11513484/1d-number-array-clustering
        # I: buffer: list of ints to cluster
        # O: clusters: list of binned ints
        # Ex: 
        # [1, 2, 3, 10, 50] w/ eps=5
        # [[1, 2, 3], [10], [50]]

        clusters = []
        EPS = 50   # Bin based on 50-pixel interval
        buffer_sorted = sorted(buffer)
        curr_point = buffer_sorted[0]
        curr_cluster = [curr_point]
        for point in buffer_sorted[1:]:
            if point <= curr_point + EPS:
                curr_cluster.append(point)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [point]
            curr_point = point
        clusters.append(curr_cluster)
        
        return clusters

    def _create_mask(self, 
                     aabb_1: geometry.AABB, 
                     aabb_2: geometry.AABB, 
                     mask_axis: int):
        # Creates masks between two tile aabb's
        # I: tile1, tile2: Tile AABB's
        # I: mask_axis: {0, 1, 2} = {z, y, x}
        # O: mask_aabb

        # Find overlap region between adjacent tiles
        # Check AABB's are colliding, meaning they colllide in all 3 axes
        assert (aabb_1[0] <= aabb_2[0] and aabb_1[1] >= aabb_2[0]) and \
               (aabb_1[2] <= aabb_2[2] and aabb_1[3] >= aabb_2[2]) and \
               (aabb_1[4] <= aabb_2[4] and aabb_1[5] >= aabb_2[4]), \
               'Input AABBs are not colliding.'
        
        # Between two colliding intervals A and B, 
        # the overlap interval is the maximum of (A_min, B_min)
        # and the minimum of (A_max, B_max).
        overlap_aabb = (np.max([aabb_1[0], aabb_2[0]]),
                    np.min([aabb_1[1], aabb_2[1]]),
                    np.max([aabb_1[2], aabb_2[2]]),
                    np.min([aabb_1[3], aabb_2[3]]),
                    np.max([aabb_1[4], aabb_2[4]]),
                    np.min([aabb_1[5], aabb_2[5]]))
    
        # Convert overlap region into mask 
        # (Masks will not be tight to output volume boundary)
        min_z, max_z, min_y, max_y, min_x, max_x = overlap_aabb
        cz, cy, cx = self.cell_size

        if mask_axis == 0: 
            max_z = min_z + ((max_z - min_z) * self.mask_percent)
        if mask_axis == 1: 
            max_y = min_y + ((max_y - min_y) * self.mask_percent)
        if mask_axis == 2: 
            max_x = min_x + ((max_x - min_x) * self.mask_percent)

        mask_aabb = (floor(min_z / cz) * cz, 
                    ceil(max_z / cz) * cz,
                    floor(min_y / cy) * cy,
                    ceil(max_y / cy) * cy,
                    floor(min_x / cx) * cx,
                    ceil(max_x / cx) * cx)
        
        return mask_aabb

    def blend(self, 
              snowball_chunk: torch.Tensor, 
              chunks: list[torch.Tensor], 
              device: torch.device, 
              kwargs = {}
    ) -> torch.Tensor:
        
        # Keyword arguments:
        # chunk_tile_ids: list of tile ids corresponding to each chunk
        # cell_box: cell AABB in output volume/absolute coordinates
        chunk_tile_ids = kwargs['chunk_tile_ids']
        cell_box = kwargs['cell_box']

        # For every chunk, add chunk to blend list 
        # if the chunk does not reside in a tile mask and inside 
        # the mask's block list.  
        to_blend: list[torch.Tensor] = []
        for t_id, chunk in zip(chunk_tile_ids, chunks):
            for m_id in self.tile_to_mask_ids[t_id]:
                m_aabb = self.masks[m_id]
                blocked_tids = self.block_list[m_id]

                # Add to blending if chunk does not fulfill masking condition. 
                # Masking condition: 
                # - Inside mask
                #   Collision defined by overlapping intervals in all 3 dimensions.
                #   Two intervals (A, B) collide if A_max is not <= B_min 
                #   and A_min is not >= B_max.
                # - Inside mask's block list
                if not ((cell_box[1] > m_aabb[0] and cell_box[0] > m_aabb[1])
                        and (cell_box[3] > m_aabb[2] and cell_box[2] < m_aabb[3])
                        and (cell_box[5] > m_aabb[4] and cell_box[4] < m_aabb[5])
                        and t_id in blocked_tids):
                    to_blend.append(chunk)

        assert (
            len(to_blend) > 0
        ), f"Length of pass list is {len(to_blend)}, blending requires 1 or more chunks."

        fused_chunk = torch.maximum(snowball_chunk.to(device), to_blend[0].to(device))
        for c in to_blend[1:]:
            c = c.to(device)
            fused_chunk = torch.maximum(fused_chunk, c)

        return fused_chunk
