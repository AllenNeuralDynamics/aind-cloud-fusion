"""
Interface for generic blending.
"""
import numpy as np
import torch

from collections import defaultdict

import aind_cloud_fusion.geometry as geometry


class BlendingModule:
    """
    Minimal interface for modular blending function.
    Subclass can define arbitrary constructors/attributes/members as necessary.
    """

    def blend(self,
              chunks: list[torch.Tensor],
              device: torch.device,
              kwargs = {}
    ) -> torch.Tensor:
        """
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

    def blend(self,
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

        fused_chunk = chunks[0].to(device)
        for c in chunks[1:]:
            c = c.to(device)
            fused_chunk = torch.maximum(fused_chunk, c)

        return fused_chunk


class SimpleAveraging(BlendingModule):
    """
    Simple average of the overlaping regions.
    """

    def __init__(self,
                 tile_layout: list[list[int]],
                 tile_aabbs: dict[int, geometry.AABB]
                 ) -> None:
        super().__init__()
        """
        tile_layout: array of tile ids arranged corresponding to stage coordinates
        tile_aabbs: dict of tile_id -> AABB, defined in fusion initalization.

        Important NOTE:
        tile_layout follows axis convention:
        +--- +x
        |
        |
        +y
        Inconsistent tile_layout with absolute tile coordinates
        will result in error.

        """
        self.tile_layout = tile_layout
        self.tile_aabbs = tile_aabbs

        # Create mask data structures
        # tile_to_overlap_ids: Maps tile_id to associated overlap region id
        # overlaps: Maps overlap_id to actual overlap region AABB
        # Access pattern:
        # tile_id -> overlap_id -> overlaps
        self.tile_to_overlap_ids: dict[int, list[int]] = defaultdict(list)
        self.overlaps: dict[int, geometry.AABB] = {}

        # directions basis: (i, j)
        x_length = len(self.tile_layout)
        y_length = len(self.tile_layout[0])
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),         (0, 1),
                      (1, -1), (1, 0), (1, 1)]
        overlap_id = 0
        for x in range(x_length):
            for y in range(y_length):
                for (dx, dy) in directions:
                    nx = x + dx
                    ny = y + dy
                    if (0 <= nx and nx < x_length and
                       0 <= ny and ny < y_length):

                        c_id = self.tile_layout[x][y]
                        c_aabb = self.tile_aabbs[c_id]
                        n_id = self.tile_layout[nx][ny]
                        n_aabb = self.tile_aabbs[n_id]
                        o_aabb = self._get_overlap_aabb(c_aabb, n_aabb)
                        self.tile_to_overlap_ids[c_id].append(overlap_id)
                        self.overlaps[overlap_id] = o_aabb

                        overlap_id += 1

    def _get_overlap_aabb(self,
                          aabb_1: geometry.AABB,
                          aabb_2: geometry.AABB):
        """
        Utility for finding overlapping regions between tiles and chunks.
        """

        # Check AABB's are colliding, meaning they colllide in all 3 axes
        assert (aabb_1[1] > aabb_2[0] and aabb_1[0] < aabb_2[1]) and \
               (aabb_1[3] > aabb_2[2] and aabb_1[2] < aabb_2[3]) and \
               (aabb_1[5] > aabb_2[4] and aabb_1[4] < aabb_2[5]), \
               f'Input AABBs are not colliding: {aabb_1=}, {aabb_2=}'

        # Between two colliding intervals A and B,
        # the overlap interval is the maximum of (A_min, B_min)
        # and the minimum of (A_max, B_max).
        overlap_aabb = (np.max([aabb_1[0], aabb_2[0]]),
                    np.min([aabb_1[1], aabb_2[1]]),
                    np.max([aabb_1[2], aabb_2[2]]),
                    np.min([aabb_1[3], aabb_2[3]]),
                    np.max([aabb_1[4], aabb_2[4]]),
                    np.min([aabb_1[5], aabb_2[5]]))

        return overlap_aabb

    def blend(self,
              chunks: list[torch.Tensor],
              device: torch.device,
              kwargs = {}
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        snowball chunk: 5d tensor in 11zyx order
        chunks: 5d tensor(s) in 11zyx order
        kwargs:
            chunk_tile_ids:
                list of tile ids corresponding to each chunk
            cell_box:
                cell AABB in output volume/absolute coordinates

        Returns
        -------
        fused_chunk: combined chunk
        """

        # Trivial no blending case -- non-overlaping region.
        if len(chunks) == 1:
            return chunks[0]

        # For 2+ chunks, within an overlapping region:
        chunk_tile_ids = kwargs['chunk_tile_ids']
        cell_box = kwargs['cell_box']

        # 1) Derive occupancy maps
        occupancy_maps: list[torch.Tensor] = []
        for t_id in chunk_tile_ids:
            # Define occupancy map.
            # We will filter this to only pass occupancy within overlap region.
            occupancy_map = torch.ones(chunks[0].shape)

            for o_id in self.tile_to_overlap_ids[t_id]:
                # Find the overlapping region(s) that
                # this chunk belongs to.
                o_aabb = self.overlaps[o_id]
                chunk_is_within_overlap = \
                ((cell_box[1] > o_aabb[0] and cell_box[0] < o_aabb[1]) and \
                (cell_box[3] > o_aabb[2] and cell_box[2] < o_aabb[3]) and \
                (cell_box[5] > o_aabb[4] and cell_box[4] < o_aabb[5]))
                if not chunk_is_within_overlap:
                    continue

                # Next, color count_map based on
                # partial or full occulsion
                cell_aabb = np.array(cell_box).flatten()
                full_z_overlap = (cell_aabb[0] >= o_aabb[0] and
                                    cell_aabb[1] <= o_aabb[1])
                full_y_overlap = (cell_aabb[2] >= o_aabb[2] and
                                    cell_aabb[3] <= o_aabb[3])
                full_x_overlap = (cell_aabb[4] >= o_aabb[4] and
                                    cell_aabb[5] <= o_aabb[5])

                # Full occlusion-- tile signal contributes to entire overlap region
                if full_z_overlap and full_y_overlap and full_x_overlap:
                    occupancy_map += 1
                    break   # Full occupancy, move onto next tile.

                # Partial occulision-- tile signal contributes to portion of overlap region
                else:
                    partial_min_z_overlap = cell_aabb[0] < o_aabb[0]
                    partial_min_y_overlap = cell_aabb[2] < o_aabb[2]
                    partial_min_x_overlap = cell_aabb[4] < o_aabb[4]

                    partial_max_z_overlap = o_aabb[1] < cell_aabb[1]
                    partial_max_y_overlap = o_aabb[3] < cell_aabb[3]
                    partial_max_x_overlap = o_aabb[5] < cell_aabb[5]

                    signal_map = torch.zeros(chunks[0].shape)
                    if not full_z_overlap:
                        if partial_min_z_overlap:
                            z_index = round(o_aabb[0] - cell_aabb[0])
                            signal_map[0, 0, z_index:, :, :] += 1

                        if partial_max_z_overlap:
                            z_index = round(o_aabb[1] - cell_aabb[0])
                            signal_map[0, 0, :z_index, :, :] += 1

                    if not full_y_overlap:
                        if partial_min_y_overlap:
                            y_index = round(o_aabb[2] - cell_aabb[2])
                            signal_map[0, 0, :, y_index:, :] += 1

                        if partial_max_y_overlap:
                            y_index = round(o_aabb[3] - cell_aabb[2])
                            signal_map[0, 0, :, :y_index, :] += 1

                    if not full_x_overlap:
                        if partial_min_x_overlap:
                            x_index = round(o_aabb[4] - cell_aabb[4])
                            signal_map[0, 0, :, :, x_index:] += 1

                        if partial_max_x_overlap:
                            x_index = round(o_aabb[5] - cell_aabb[4])
                            signal_map[0, 0, :, :, :x_index] += 1

                    occupancy_map *= signal_map

            # End of loop through overlap regions
            occupancy_maps.append(occupancy_map)

        # 2) Post-process overlap map into weight map.
        count_map = occupancy_maps[0]
        for o in occupancy_maps[1:]:
            count_map += o
        count_map[count_map == 0] = 1
        weight_map = 1. / count_map

        # 3) Finally, blend all the chunks together.
        fused_chunk = torch.zeros(chunks[0].shape)
        weight_map = weight_map.to(device)
        for c in chunks:
            fused_chunk += (weight_map * c)

        return fused_chunk


class WeightedLinearBlending(BlendingModule):
    """
    Linear Blending with distance-based weights.
    """

    def __init__(self,
                 tile_layout: list[list[int]],
                 tile_aabbs: dict[int, geometry.AABB]
                 ) -> None:
        super().__init__()
        """
        tile_layout: array of tile ids arranged corresponding to stage coordinates
        tile_aabbs: dict of tile_id -> AABB, defined in fusion initalization.

        Important NOTE:
        tile_layout follows axis convention:
        +--- +x
        |
        |
        +y
        Inconsistent tile_layout with absolute tile coordinates
        will result in error.

        """
        self.tile_layout = tile_layout
        self.tile_aabbs = tile_aabbs

        # Create mask data structures
        # tile_to_overlap_ids: Maps tile_id to associated overlap region id
        # overlaps: Maps overlap_id to actual overlap region AABB
        # Access pattern:
        # tile_id -> overlap_id -> overlaps
        self.tile_to_overlap_ids: dict[int, list[int]] = defaultdict(list)
        self.overlaps: dict[int, geometry.AABB] = {}

        # directions basis: (i, j)
        x_length = len(self.tile_layout)
        y_length = len(self.tile_layout[0])
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),         (0, 1),
                      (1, -1), (1, 0), (1, 1)]
        overlap_id = 0
        for x in range(x_length):
            for y in range(y_length):
                for (dx, dy) in directions:
                    nx = x + dx
                    ny = y + dy
                    if (0 <= nx and nx < x_length and
                       0 <= ny and ny < y_length):

                        c_id = self.tile_layout[x][y]
                        c_aabb = self.tile_aabbs[c_id]
                        n_id = self.tile_layout[nx][ny]
                        n_aabb = self.tile_aabbs[n_id]
                        o_aabb = self._get_overlap_aabb(c_aabb, n_aabb)
                        self.tile_to_overlap_ids[c_id].append(overlap_id)
                        self.overlaps[overlap_id] = o_aabb

                        overlap_id += 1

        # Define tile_centers for reference in conic weight function
        self.tile_centers: dict[int, tuple[float, float, float]] = {}
        for t_id, t_aabb in tile_aabbs.items():
            mz, my, mx = (t_aabb[1] - t_aabb[0],
                          t_aabb[3] - t_aabb[2],
                          t_aabb[5] - t_aabb[4])
            self.tile_centers[t_id] = (mz, my, mx)

    def _get_overlap_aabb(self,
                          aabb_1: geometry.AABB,
                          aabb_2: geometry.AABB):
        """
        Utility for finding overlapping regions between tiles and chunks.
        """

        # Check AABB's are colliding, meaning they colllide in all 3 axes
        assert (aabb_1[1] > aabb_2[0] and aabb_1[0] < aabb_2[1]) and \
               (aabb_1[3] > aabb_2[2] and aabb_1[2] < aabb_2[3]) and \
               (aabb_1[5] > aabb_2[4] and aabb_1[4] < aabb_2[5]), \
               f'Input AABBs are not colliding: {aabb_1=}, {aabb_2=}'

        # Between two colliding intervals A and B,
        # the overlap interval is the maximum of (A_min, B_min)
        # and the minimum of (A_max, B_max).
        overlap_aabb = (np.max([aabb_1[0], aabb_2[0]]),
                    np.min([aabb_1[1], aabb_2[1]]),
                    np.max([aabb_1[2], aabb_2[2]]),
                    np.min([aabb_1[3], aabb_2[3]]),
                    np.max([aabb_1[4], aabb_2[4]]),
                    np.min([aabb_1[5], aabb_2[5]]))

        return overlap_aabb

    def blend(self,
              chunks: list[torch.Tensor],
              device: torch.device,
              kwargs = {}
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        snowball chunk: 5d tensor in 11zyx order
        chunks: 5d tensor(s) in 11zyx order
        kwargs:
            chunk_tile_ids:
                list of tile ids corresponding to each chunk
            cell_box:
                cell AABB in output volume/absolute coordinates

        Returns
        -------
        fused_chunk: combined chunk
        """

        # Trivial no blending case -- non-overlaping region.
        if len(chunks) == 1:
            return chunks[0]

        # For 2+ chunks, within an overlapping region:
        chunk_tile_ids = kwargs['chunk_tile_ids']
        cell_box = kwargs['cell_box']

        # 1) Derive distance maps
        total_distance_map = torch.zeros(chunks[0].shape)
        distance_maps: list[torch.Tensor] = []  # Corresponding order as chunks
        for t_id in chunk_tile_ids:
            # Define distance map.
            # Calculate distance of absolute coordinates wrt to current tile.
            # We will filter this to only pass distances within the overlap region.
            z_indices = torch.arange(cell_box[0], cell_box[1], step=1) + 0.5
            y_indices = torch.arange(cell_box[2], cell_box[3], step=1) + 0.5
            x_indices = torch.arange(cell_box[4], cell_box[5], step=1) + 0.5
            z_grid, y_grid, x_grid = torch.meshgrid(
                z_indices, y_indices, x_indices, indexing="ij"  # {z_grid, y_grid, x_grid} are 3D Tensors
            )
            cz, cy, cx = self.tile_centers[t_id]
            z_dist = torch.abs(z_grid - cz)
            y_dist = torch.abs(y_grid - cy)
            x_dist = torch.abs(x_grid - cx)
            distance_map = 1 / (z_dist + y_dist + x_dist)  # Notion of distance decreases with distance from center
            distance_map = distance_map.unsqueeze(0).unsqueeze(0)  # Now a 5D Tensor

            for o_id in self.tile_to_overlap_ids[t_id]:
                # Find the overlapping region(s) that
                # this chunk belongs to.
                o_aabb = self.overlaps[o_id]
                chunk_is_within_overlap = \
                ((cell_box[1] > o_aabb[0] and cell_box[0] < o_aabb[1]) and \
                (cell_box[3] > o_aabb[2] and cell_box[2] < o_aabb[3]) and \
                (cell_box[5] > o_aabb[4] and cell_box[4] < o_aabb[5]))
                if not chunk_is_within_overlap:
                    continue

                # Next, mask the distance map according to
                # actual signal in the overlap region.
                cell_aabb = np.array(cell_box).flatten()
                full_z_overlap = (cell_aabb[0] >= o_aabb[0] and
                                    cell_aabb[1] <= o_aabb[1])
                full_y_overlap = (cell_aabb[2] >= o_aabb[2] and
                                    cell_aabb[3] <= o_aabb[3])
                full_x_overlap = (cell_aabb[4] >= o_aabb[4] and
                                    cell_aabb[5] <= o_aabb[5])

                # Full occlusion-- tile signal contributes to entire overlap region
                if full_z_overlap and full_y_overlap and full_x_overlap:
                    break   # All distances lie in overlap region, move onto next tile.

                # Partial occulision-- tile signal contributes to portion of overlap region
                else:
                    partial_min_z_overlap = cell_aabb[0] < o_aabb[0]
                    partial_min_y_overlap = cell_aabb[2] < o_aabb[2]
                    partial_min_x_overlap = cell_aabb[4] < o_aabb[4]

                    partial_max_z_overlap = o_aabb[1] < cell_aabb[1]
                    partial_max_y_overlap = o_aabb[3] < cell_aabb[3]
                    partial_max_x_overlap = o_aabb[5] < cell_aabb[5]

                    signal_map = torch.zeros(chunks[0].shape)
                    if not full_z_overlap:
                        if partial_min_z_overlap:
                            z_index = round(o_aabb[0] - cell_aabb[0])
                            signal_map[0, 0, z_index:, :, :] += 1

                        if partial_max_z_overlap:
                            z_index = round(o_aabb[1] - cell_aabb[0])
                            signal_map[0, 0, :z_index, :, :] += 1

                    if not full_y_overlap:
                        if partial_min_y_overlap:
                            y_index = round(o_aabb[2] - cell_aabb[2])
                            signal_map[0, 0, :, y_index:, :] += 1

                        if partial_max_y_overlap:
                            y_index = round(o_aabb[3] - cell_aabb[2])
                            signal_map[0, 0, :, :y_index, :] += 1

                    if not full_x_overlap:
                        if partial_min_x_overlap:
                            x_index = round(o_aabb[4] - cell_aabb[4])
                            signal_map[0, 0, :, :, x_index:] += 1

                        if partial_max_x_overlap:
                            x_index = round(o_aabb[5] - cell_aabb[4])
                            signal_map[0, 0, :, :, :x_index] += 1

                    distance_map *= signal_map  # Pass distance map through signal map

            # End of loop through overlap regions
            distance_maps.append(distance_map)
            total_distance_map += distance_map

        # 2) Post-process distance maps into weight maps
        weight_maps = []
        for d_map, c in zip(distance_maps, chunks):
            weight_map = d_map / total_distance_map
            weight_map[d_map == 0] = 1
            weight_maps.append(weight_map)

        # 3) Finally, blend all the chunks together.
        fused_chunk = torch.zeros(chunks[0].shape)
        for w, c in zip(weight_maps, chunks):
            w = w.to(device)
            c = c.to(device)
            fused_chunk += (w * c)

        return fused_chunk

# Immediate problem is why the weight map suprreses all signal outside overlap.
# Appears like line 484 (or steps leading up to that) are not doing what is intended.
# Dividing by 0! Let's fix that.

class MaskedBlending(BlendingModule):
    """
    Defines a mask within tile overlap regions in which tiles are not blended.
    IMPORTANT:
    Assumes tiles are arranged in a regular grid-- no significant warping of the image volume.
    """
    CLUSTER_EPS = 500  # Size of bucket to bin tile_aabb corners to form tile_layout.

    def __init__(self, tile_aabbs: dict[int, geometry.AABB],
                       cell_size: tuple[int, int, int],
                       mask_axes: list[int],
                       mask_percent: float = 0.9,
                       cluster_eps: int = 500):
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
        MaskedBlending.CLUSTER_EPS = cluster_eps

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
        # Ref: [[1, 2, 3], [10], [50]]

        # Indices of tile layout = (tile_point - origin) / tile_stride
        points = np.array(points)
        origin = np.zeros(3)
        origin[0] = np.min(points[:, 0])
        origin[1] = np.min(points[:, 1])
        origin[2] = np.min(points[:, 2])

        tile_stride = [1, 1, 1]
        if len(z_clusters) != 1:
            tile_stride[0] = z_clusters[1][0] - z_clusters[0][0]
        if len(y_clusters) != 1:
            tile_stride[1] = y_clusters[1][0] - y_clusters[0][0]
        if len(x_clusters) != 1:
            tile_stride[2] = x_clusters[1][0] - x_clusters[0][0]

        z_dim, y_dim, x_dim = len(z_clusters), len(y_clusters), len(x_clusters)
        TILE_LAYOUT = np.zeros((z_dim, y_dim, x_dim)).astype('uint8')
        print('f{TILE_LAYOUT.shape=}')
        for t_id, pt in zip(t_ids, points):
            z = round((pt[0] - origin[0]) / tile_stride[0])
            y = round((pt[1] - origin[1]) / tile_stride[1])
            x = round((pt[2] - origin[2]) / tile_stride[2])
            TILE_LAYOUT[z, y, x] = int(t_id)

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
        buffer_sorted = sorted(buffer)
        curr_point = buffer_sorted[0]
        curr_cluster = [curr_point]
        for point in buffer_sorted[1:]:
            if point <= curr_point + MaskedBlending.CLUSTER_EPS:
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
        overlap_aabb = self._get_overlap_aabb(aabb_1, aabb_2)

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

        mask_aabb = (min_z, max_z, min_y, max_y, min_x, max_x)

        return mask_aabb

    def _get_overlap_aabb(self,
                          aabb_1: geometry.AABB,
                          aabb_2: geometry.AABB):
        # Check AABB's are colliding, meaning they colllide in all 3 axes
        assert (aabb_1[1] > aabb_2[0] and aabb_1[0] < aabb_2[1]) and \
               (aabb_1[3] > aabb_2[2] and aabb_1[2] < aabb_2[3]) and \
               (aabb_1[5] > aabb_2[4] and aabb_1[4] < aabb_2[5]), \
               f'Input AABBs are not colliding: {aabb_1=}, {aabb_2=}'

        # Between two colliding intervals A and B,
        # the overlap interval is the maximum of (A_min, B_min)
        # and the minimum of (A_max, B_max).
        overlap_aabb = (np.max([aabb_1[0], aabb_2[0]]),
                    np.min([aabb_1[1], aabb_2[1]]),
                    np.max([aabb_1[2], aabb_2[2]]),
                    np.min([aabb_1[3], aabb_2[3]]),
                    np.max([aabb_1[4], aabb_2[4]]),
                    np.min([aabb_1[5], aabb_2[5]]))

        return overlap_aabb

    def blend(self,
              snowball_chunk: torch.Tensor,
              chunks: list[torch.Tensor],
              device: torch.device,
              kwargs = {}
    ) -> torch.Tensor:

        # Keyword arguments:
        # snowball chunk/chunks are 5d tensors in 11zyx order.
        # chunk_tile_ids: list of tile ids corresponding to each chunk
        # cell_box: cell AABB in output volume/absolute coordinates
        chunk_tile_ids = kwargs['chunk_tile_ids']
        cell_box = kwargs['cell_box']

        # Iterate though chunks and each chunk and chunk
        to_blend: list[torch.Tensor] = []
        for t_id, chunk in zip(chunk_tile_ids, chunks):
            for m_id in self.tile_to_mask_ids[t_id]:
                m_aabb = self.masks[m_id]
                blocked_tids = self.block_list[m_id]

                # Reject conditions
                chunk_is_masked = \
                   ((cell_box[1] > m_aabb[0] and cell_box[0] < m_aabb[1]) and \
                   (cell_box[3] > m_aabb[2] and cell_box[2] < m_aabb[3]) and \
                   (cell_box[5] > m_aabb[4] and cell_box[4] < m_aabb[5]))
                chunk_is_blocked = t_id in blocked_tids

                # If chunk is masked and blocked,
                # figure out how much of the chunk is masked and blocked.
                if chunk_is_masked and chunk_is_blocked:
                    cell_aabb = np.array(cell_box).flatten()
                    cell_inside_mask_aabb = self._get_overlap_aabb(m_aabb, cell_aabb)
                    full_z_overlap = (cell_inside_mask_aabb[0] == cell_aabb[0] and
                                      cell_inside_mask_aabb[1] == cell_aabb[1])
                    full_y_overlap = (cell_inside_mask_aabb[2] == cell_aabb[2] and
                                      cell_inside_mask_aabb[3] == cell_aabb[3])
                    full_x_overlap = (cell_inside_mask_aabb[4] == cell_aabb[4] and
                                      cell_inside_mask_aabb[5] == cell_aabb[5])

                    # If chunk is partially masked,
                    # pass along region of chunk that is not masked for blending.
                    # Otherwise, this conditional branch does not contribute to to_blend.
                    if not (full_z_overlap and full_y_overlap and full_x_overlap):
                        chunk_mask = torch.ones(chunk.shape)
                        partial_min_z_overlap = (cell_inside_mask_aabb[0] <= m_aabb[0] and
                                                 m_aabb[0] <= cell_inside_mask_aabb[1])
                        partial_min_y_overlap = (cell_inside_mask_aabb[2] <= m_aabb[2] and
                                                 m_aabb[2] <= cell_inside_mask_aabb[3])
                        partial_min_x_overlap = (cell_inside_mask_aabb[4] <= m_aabb[4] and
                                                 m_aabb[4] <= cell_inside_mask_aabb[5])

                        partial_max_z_overlap = (cell_inside_mask_aabb[0] <= m_aabb[1] and
                                                 m_aabb[1] <= cell_inside_mask_aabb[1])
                        partial_max_y_overlap = (cell_inside_mask_aabb[2] <= m_aabb[3] and
                                                 m_aabb[3] <= cell_inside_mask_aabb[3])
                        partial_max_x_overlap = (cell_inside_mask_aabb[4] <= m_aabb[5] and
                                                 m_aabb[5] <= cell_inside_mask_aabb[5])

                        # NOTE: Mask is mirrored depending on +/- partial overlap.
                        if not full_z_overlap:
                            if partial_min_z_overlap:
                                z_index = round(m_aabb[0] - cell_inside_mask_aabb[0])
                                chunk_mask[:, :, z_index:, :, :] = 0

                            if partial_max_z_overlap:
                                z_index = round(m_aabb[1] - cell_inside_mask_aabb[0])
                                chunk_mask[:, :, :z_index, :, :] = 0

                        if not full_y_overlap:
                            if partial_min_y_overlap:
                                y_index = round(m_aabb[2] - cell_inside_mask_aabb[2])
                                chunk_mask[:, :, :, y_index:, :] = 0

                            if partial_max_y_overlap:
                                y_index = round(m_aabb[3] - cell_inside_mask_aabb[2])
                                chunk_mask[:, :, :, :y_index, :] = 0

                        if not full_x_overlap:
                            if partial_min_x_overlap:
                                x_index = round(m_aabb[4] - cell_inside_mask_aabb[4])
                                chunk_mask[:, :, :, :, x_index:] = 0

                            if partial_max_x_overlap:
                                x_index = round(m_aabb[5] - cell_inside_mask_aabb[4])
                                chunk_mask[:, :, :, :, :x_index] = 0

                        chunk_mask = chunk_mask.to(device)
                        updated_chunk = chunk_mask * chunk
                        to_blend.append(updated_chunk)

                else:
                    to_blend.append(chunk)

        # All input chunks are masked
        if len(to_blend) == 0:
            return snowball_chunk

        # Blend un-masked chunks
        else:
            fused_chunk = torch.maximum(snowball_chunk.to(device), to_blend[0].to(device))
            for c in to_blend[1:]:
                c = c.to(device)
                fused_chunk = torch.maximum(fused_chunk, c)

            return fused_chunk
