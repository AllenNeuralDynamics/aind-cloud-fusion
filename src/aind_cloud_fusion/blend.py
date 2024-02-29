"""
Interface for generic blending.
"""
import numpy as np
import torch
import xmltodict

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


def get_overlap_regions(tile_layout: list[list[int]],
                        tile_aabbs: dict[int, geometry.AABB]
                        ) -> tuple[dict[int, list[int]], 
                                   dict[int, geometry.AABB]]:
    """
    Input: 
    tile_layout: array of tile ids arranged corresponding to stage coordinates
    tile_aabbs: dict of tile_id -> AABB, defined in fusion initalization.
    
    Output: 
    tile_to_overlap_ids: Maps tile_id to associated overlap region id
    overlaps: Maps overlap_id to actual overlap region AABB
    
    Access pattern:
    tile_id -> overlap_id -> overlaps
    """

    def _get_overlap_aabb(aabb_1: geometry.AABB,
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

    # Output Data Structures
    tile_to_overlap_ids: dict[int, list[int]] = defaultdict(list)
    overlaps: dict[int, geometry.AABB] = {}

    # 1) Find all unique edges
    edges: list[tuple[int, int]] = []
    x_length = len(tile_layout)
    y_length = len(tile_layout[0])
    directions = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),         (0, 1),
                    (1, -1), (1, 0), (1, 1)]
    for x in range(x_length):
        for y in range(y_length):
            for (dx, dy) in directions:
                nx = x + dx
                ny = y + dy
                if (0 <= nx and nx < x_length and
                    0 <= ny and ny < y_length and   # Boundary conditions
                    tile_layout[x][y] != -1 and
                    tile_layout[nx][ny] != -1):  # Spacer conditions

                    id_1 = tile_layout[x][y]
                    id_2 = tile_layout[nx][ny]
                    e = tuple(sorted([id_1, id_2]))
                    edges.append(e)
    edges = sorted(list(set(edges)), key=lambda x: (x[0], x[1]))

    # 2) Find overlap regions
    overlap_id = 0
    for (id_1, id_2) in edges: 
        aabb_1 = tile_aabbs[id_1]
        aabb_2 = tile_aabbs[id_2]

        try:
            o_aabb = _get_overlap_aabb(aabb_1, aabb_2)
        except:
            continue

        overlaps[overlap_id] = o_aabb
        tile_to_overlap_ids[id_1].append(overlap_id)
        tile_to_overlap_ids[id_2].append(overlap_id)
        overlap_id += 1

    return tile_to_overlap_ids, overlaps


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
        """

        # Create mask data structures
        self.tile_to_overlap_ids: dict[int, list[int]] = defaultdict(list)
        self.overlaps: dict[int, geometry.AABB] = {}
        self.tile_to_overlap_ids, self.overlaps = get_overlap_regions(tile_layout, tile_aabbs)

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
        occupancy_maps: dict[list[torch.Tensor]] = defaultdict(list)
        for t_id in chunk_tile_ids:
            for o_id in self.tile_to_overlap_ids[t_id]:
                # Each tile has potentially many overlap regions.
                # Check if the current chunk is within the current overlap before proceeding.
                o_aabb = self.overlaps[o_id]
                chunk_outside_overlap_region = ((cell_box[1] <= o_aabb[0] or o_aabb[1] <= cell_box[0]) or
                                                (cell_box[3] <= o_aabb[2] or o_aabb[3] <= cell_box[2]) or
                                                (cell_box[5] <= o_aabb[4] or o_aabb[5] <= cell_box[4]))
                if chunk_outside_overlap_region:
                    continue

                # To be modified in the following colision checks.
                occupancy_map = torch.zeros(chunks[0].shape)
                MASK_SLICE = [slice(None), slice(None), slice(None), slice(None), slice(None)]

                # Derive Occupancy Map for full chunk coverings in each dimension
                # chunk: (---)
                # overlap: [---]
                # Ex: (---[-----]---)
                chunk_covers_overlap_z = (cell_box[0] < o_aabb[0] and o_aabb[1] < cell_box[1])
                chunk_covers_overlap_y = (cell_box[2] < o_aabb[2] and o_aabb[3] < cell_box[3])
                chunk_covers_overlap_x = (cell_box[4] < o_aabb[4] and o_aabb[5] < cell_box[5])
                if chunk_covers_overlap_z:
                    start_index = round(o_aabb[0] - cell_box[0])
                    end_index = round(o_aabb[1] - cell_box[0])
                    MASK_SLICE[2] = slice(start_index, end_index)

                if chunk_covers_overlap_y:
                    start_index = round(o_aabb[2] - cell_box[2])
                    end_index = round(o_aabb[3] - cell_box[2])
                    MASK_SLICE[3] = slice(start_index, end_index)

                if chunk_covers_overlap_x:
                    start_index = round(o_aabb[4] - cell_box[4])
                    end_index = round(o_aabb[5] - cell_box[4])
                    MASK_SLICE[4] = slice(start_index, end_index)

                # Derive Occupancy Map for full overlap coverings in each dimension
                # chunk: (---)
                # overlap: [---]
                # Ex: [---(-----)---]
                overlap_covers_chunk_z = (o_aabb[0] <= cell_box[0] and cell_box[1] <= o_aabb[1])
                overlap_covers_chunk_y = (o_aabb[2] <= cell_box[2] and cell_box[3] <= o_aabb[3])
                overlap_covers_chunk_x = (o_aabb[4] <= cell_box[4] and cell_box[5] <= o_aabb[5])
                if overlap_covers_chunk_z:
                    MASK_SLICE[2] = slice(None)
                if overlap_covers_chunk_y:
                    MASK_SLICE[3] = slice(None)
                if overlap_covers_chunk_x:
                    MASK_SLICE[4] = slice(None)

                # Derive Occupancy Map for partial chunk coverings in each dimension
                # chunk: (---)
                # overlap: [---]
                # Ex: (---[------)---] or [---(-----]--)
                chunk_covers_min_overlap_z = (cell_box[0] < o_aabb[0] and
                                              o_aabb[0] < cell_box[1] and cell_box[1] <= o_aabb[1])
                chunk_covers_min_overlap_y = (cell_box[2] < o_aabb[2] and
                                              o_aabb[2] < cell_box[3] and cell_box[3] <= o_aabb[3])
                chunk_covers_min_overlap_x = (cell_box[4] < o_aabb[4] and
                                              o_aabb[4] < cell_box[5] and cell_box[5] <= o_aabb[5])
                if chunk_covers_min_overlap_z:
                    start_index = round(o_aabb[0] - cell_box[0])
                    MASK_SLICE[2] = slice(start_index, None)
                if chunk_covers_min_overlap_y:
                    start_index = round(o_aabb[2] - cell_box[2])
                    MASK_SLICE[3] = slice(start_index, None)
                if chunk_covers_min_overlap_x:
                    start_index = round(o_aabb[4] - cell_box[4])
                    MASK_SLICE[4] = slice(start_index, None)

                chunk_covers_max_overlap_z = (o_aabb[0] <= cell_box[0] and cell_box[0] < o_aabb[1] and
                                              o_aabb[1] < cell_box[1])
                chunk_covers_max_overlap_y = (o_aabb[2] <= cell_box[2] and cell_box[2] < o_aabb[3] and
                                              o_aabb[3] < cell_box[3])
                chunk_covers_max_overlap_x = (o_aabb[4] <= cell_box[4] and cell_box[4] < o_aabb[5] and
                                              o_aabb[5] < cell_box[5])
                if chunk_covers_max_overlap_z:
                    end_index = round(o_aabb[1] - cell_box[0])
                    MASK_SLICE[2] = slice(0, end_index)
                if chunk_covers_max_overlap_y:
                    end_index = round(o_aabb[3] - cell_box[2])
                    MASK_SLICE[3] = slice(0, end_index)
                if chunk_covers_max_overlap_x:
                    end_index = round(o_aabb[5] - cell_box[4])
                    MASK_SLICE[4] = slice(0, end_index)

                occupancy_map[MASK_SLICE] = 1
                occupancy_maps[t_id].append(occupancy_map)

        # 2) Post-process occupancy maps into weight map.
        count_map = torch.zeros(chunks[0].shape)
        for t_id, t_maps in occupancy_maps.items():
            # Logical-OR occupancy maps at tile level
            net_occupancy_map = torch.zeros(chunks[0].shape)
            for t_map in t_maps:
                net_occupancy_map += t_map
            net_occupancy_map[net_occupancy_map > 1] = 1

            # Accumulate net_occupancy maps into count_map
            count_map += net_occupancy_map

        # Pass full signal to unmasked regions
        # Masked regions recieve 1/num_tiles weight.
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
        """

        # Create mask data structures
        self.tile_to_overlap_ids: dict[int, list[int]] = defaultdict(list)
        self.overlaps: dict[int, geometry.AABB] = {}
        self.tile_to_overlap_ids, self.overlaps = get_overlap_regions(tile_layout, tile_aabbs)

        # Define tile_centers for reference in conic weight function
        self.tile_centers: dict[int, tuple[float, float, float]] = {}
        for t_id, t_aabb in tile_aabbs.items():
            mz, my, mx = (t_aabb[1] - t_aabb[0],
                          t_aabb[3] - t_aabb[2],
                          t_aabb[5] - t_aabb[4])
            self.tile_centers[t_id] = (mz, my, mx)

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
        occupancy_maps: dict[list[torch.Tensor]] = defaultdict(list)
        for t_id in chunk_tile_ids:
            for o_id in self.tile_to_overlap_ids[t_id]:
                # Each tile has potentially many overlap regions.
                # Check if the current chunk is within the current overlap before proceeding.
                o_aabb = self.overlaps[o_id]
                chunk_outside_overlap_region = ((cell_box[1] <= o_aabb[0] or o_aabb[1] <= cell_box[0]) or
                                                (cell_box[3] <= o_aabb[2] or o_aabb[3] <= cell_box[2]) or
                                                (cell_box[5] <= o_aabb[4] or o_aabb[5] <= cell_box[4]))
                if chunk_outside_overlap_region:
                    continue

                # To be modified in the following colision checks.
                occupancy_map = torch.zeros(chunks[0].shape)
                MASK_SLICE = [slice(None), slice(None), slice(None), slice(None), slice(None)]

                # Derive Occupancy Map for full chunk coverings in each dimension
                # chunk: (---)
                # overlap: [---]
                # Ex: (---[-----]---)
                chunk_covers_overlap_z = (cell_box[0] < o_aabb[0] and o_aabb[1] < cell_box[1])
                chunk_covers_overlap_y = (cell_box[2] < o_aabb[2] and o_aabb[3] < cell_box[3])
                chunk_covers_overlap_x = (cell_box[4] < o_aabb[4] and o_aabb[5] < cell_box[5])
                if chunk_covers_overlap_z:
                    start_index = round(o_aabb[0] - cell_box[0])
                    end_index = round(o_aabb[1] - cell_box[0])
                    MASK_SLICE[2] = slice(start_index, end_index)

                if chunk_covers_overlap_y:
                    start_index = round(o_aabb[2] - cell_box[2])
                    end_index = round(o_aabb[3] - cell_box[2])
                    MASK_SLICE[3] = slice(start_index, end_index)

                if chunk_covers_overlap_x:
                    start_index = round(o_aabb[4] - cell_box[4])
                    end_index = round(o_aabb[5] - cell_box[4])
                    MASK_SLICE[4] = slice(start_index, end_index)

                # Derive Occupancy Map for full overlap coverings in each dimension
                # chunk: (---)
                # overlap: [---]
                # Ex: [---(-----)---]
                overlap_covers_chunk_z = (o_aabb[0] <= cell_box[0] and cell_box[1] <= o_aabb[1])
                overlap_covers_chunk_y = (o_aabb[2] <= cell_box[2] and cell_box[3] <= o_aabb[3])
                overlap_covers_chunk_x = (o_aabb[4] <= cell_box[4] and cell_box[5] <= o_aabb[5])
                if overlap_covers_chunk_z:
                    MASK_SLICE[2] = slice(None)
                if overlap_covers_chunk_y:
                    MASK_SLICE[3] = slice(None)
                if overlap_covers_chunk_x:
                    MASK_SLICE[4] = slice(None)

                # Derive Occupancy Map for partial chunk coverings in each dimension
                # chunk: (---)
                # overlap: [---]
                # Ex: (---[------)---] or [---(-----]--)
                chunk_covers_min_overlap_z = (cell_box[0] < o_aabb[0] and
                                              o_aabb[0] < cell_box[1] and cell_box[1] <= o_aabb[1])
                chunk_covers_min_overlap_y = (cell_box[2] < o_aabb[2] and
                                              o_aabb[2] < cell_box[3] and cell_box[3] <= o_aabb[3])
                chunk_covers_min_overlap_x = (cell_box[4] < o_aabb[4] and
                                              o_aabb[4] < cell_box[5] and cell_box[5] <= o_aabb[5])
                if chunk_covers_min_overlap_z:
                    start_index = round(o_aabb[0] - cell_box[0])
                    MASK_SLICE[2] = slice(start_index, None)
                if chunk_covers_min_overlap_y:
                    start_index = round(o_aabb[2] - cell_box[2])
                    MASK_SLICE[3] = slice(start_index, None)
                if chunk_covers_min_overlap_x:
                    start_index = round(o_aabb[4] - cell_box[4])
                    MASK_SLICE[4] = slice(start_index, None)

                chunk_covers_max_overlap_z = (o_aabb[0] <= cell_box[0] and cell_box[0] < o_aabb[1] and
                                              o_aabb[1] < cell_box[1])
                chunk_covers_max_overlap_y = (o_aabb[2] <= cell_box[2] and cell_box[2] < o_aabb[3] and
                                              o_aabb[3] < cell_box[3])
                chunk_covers_max_overlap_x = (o_aabb[4] <= cell_box[4] and cell_box[4] < o_aabb[5] and
                                              o_aabb[5] < cell_box[5])
                if chunk_covers_max_overlap_z:
                    end_index = round(o_aabb[1] - cell_box[0])
                    MASK_SLICE[2] = slice(0, end_index)
                if chunk_covers_max_overlap_y:
                    end_index = round(o_aabb[3] - cell_box[2])
                    MASK_SLICE[3] = slice(0, end_index)
                if chunk_covers_max_overlap_x:
                    end_index = round(o_aabb[5] - cell_box[4])
                    MASK_SLICE[4] = slice(0, end_index)

                occupancy_map[MASK_SLICE] = 1
                occupancy_maps[t_id].append(occupancy_map)

        # 2) Post-process occupancy maps into weight maps
        per_chunk_distance_maps: list[torch.Tensor] = []
        total_distance_map = torch.zeros(chunks[0].shape)
        for t_id, t_maps in occupancy_maps.items():
            # Logical-OR occupancy maps at tile level
            net_occupancy_map = torch.zeros(chunks[0].shape)
            for t_map in t_maps:
                net_occupancy_map += t_map
            net_occupancy_map[net_occupancy_map > 1] = 1

            # Each tile gets its own distance map
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

            per_chunk_distance = distance_map * net_occupancy_map
            total_distance_map += per_chunk_distance
            per_chunk_distance_maps.append(per_chunk_distance)

        weight_maps: list[torch.Tensor] = []
        for d_map, c in zip(per_chunk_distance_maps, chunks):
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


def parse_yx_tile_layout(xml_path: str) -> list[list[int]]:
    """
    Utility for parsing tile layout from a bigstitcher xml
    requested by some blending modules.

    tile_layout follows axis convention:
    +--- +x
    |
    |
    +y

    Tile ids in output tile_layout uses the same tile ids
    defined in the xml file. Spaces denoted with tile id '-1'.
    """

    # Parse stage positions
    with open(xml_path, "r") as file:
        data = xmltodict.parse(file.read())
    stage_positions_xyz: dict[int, tuple[float, float, float]] = {}
    for d in data['SpimData']['ViewRegistrations']['ViewRegistration']:
        tile_id = d['@setup']

        view_transform = d['ViewTransform']
        if isinstance(view_transform, list):
            view_transform = view_transform[-1]

        nums = [float(val) for val in view_transform["affine"].split(" ")]
        stage_positions_xyz[tile_id] = tuple(nums[3::4])

    # Calculate delta_x and delta_y
    positions_arr_xyz = np.array([pos for pos in stage_positions_xyz.values()])
    x_pos = list(set(positions_arr_xyz[:, 0]))
    x_pos = sorted(x_pos)
    delta_x = x_pos[1] - x_pos[0]
    y_pos = list(set(positions_arr_xyz[:, 1]))
    y_pos = sorted(y_pos)
    delta_y = y_pos[1] - y_pos[0]

    # Fill tile_layout
    tile_layout = np.ones((len(y_pos), len(x_pos))) * -1
    for tile_id, s_pos in stage_positions_xyz.items():
        ix = int(s_pos[0] / delta_x)
        iy = int(s_pos[1] / delta_y)

        tile_layout[iy, ix] = tile_id
    
    tile_layout = tile_layout.astype(int)

    return tile_layout
