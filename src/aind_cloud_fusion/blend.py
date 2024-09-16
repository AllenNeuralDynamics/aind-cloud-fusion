"""
Interface for generic blending.
"""
from typing import Optional

import torch

import aind_cloud_fusion.geometry as geometry


class BlendingModule:
    """
    Minimal interface for modular blending function.
    Subclass can define arbitrary constructors/attributes/members as necessary.
    """

    def blend(
        self, chunks: list[torch.Tensor], device: torch.device, kwargs={}
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

    def blend(
        self, chunks: list[torch.Tensor], device: torch.device, kwargs={}
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


class WeightedLinearBlending(BlendingModule):
    """
    Linear Blending with distance-based weights.
    NOTE: Only supports translation-only registration on square tiles.
    To modify for affine registration:
    - Forward transform overlap weights into output volume.
    - Inverse transform for local weights.
    """

    def __init__(
        self,
        tile_aabbs: dict[int, geometry.AABB],
    ) -> None:
        super().__init__()
        """
        tile_aabbs: dict of tile_id -> AABB, defined in fusion initalization.
        """
        self.tile_aabbs = tile_aabbs

    def blend(
        self, chunks: list[torch.Tensor], device: torch.device, kwargs={}
    ) -> torch.Tensor:
        """
        Parameters
        ----------
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
        chunk_tile_ids = kwargs["chunk_tile_ids"]
        cell_box = kwargs["cell_box"]

        # Calculate local weight masks
        local_weights: list[torch.Tensor] = []
        total_weight: torch.Tensor = torch.zeros(chunks[0].shape)
        for tile_id, chunk in zip(chunk_tile_ids, chunks):
            tile_aabb = self.tile_aabbs[tile_id]
            x_min = tile_aabb[4]
            cy = (tile_aabb[3] + tile_aabb[2]) / 2
            cx = (tile_aabb[5] + tile_aabb[4]) / 2

            z_indices = torch.arange(cell_box[0], cell_box[1], step=1) + 0.5
            y_indices = torch.arange(cell_box[2], cell_box[3], step=1) + 0.5
            x_indices = torch.arange(cell_box[4], cell_box[5], step=1) + 0.5

            z_grid, y_grid, x_grid = torch.meshgrid(
                z_indices,
                y_indices,
                x_indices,
                indexing="ij",  # {z_grid, y_grid, x_grid} are 3D Tensors
            )

            # Weight formula:
            # 1) Apply pyramid function wrt to center of square tile.
            # For each incoming chunk, a chunk may only have partial signal,
            # representing cells that lie between two tiles.
            # 2) After calculating pyramid weights, confine weights to actual boundary
            # of image, represented by position of non-zero values in chunk.
            weights = (cx - x_min) - torch.max(
                torch.abs(x_grid - cx), torch.abs(y_grid - cy)
            )
            signal_mask = torch.clamp(chunk, 0, 1)
            inbound_weights = weights * signal_mask

            local_weights.append(inbound_weights)
            total_weight += inbound_weights

        # Calculate fused chunk
        fused_chunk = torch.zeros(chunks[0].shape)

        for w, c in zip(local_weights, chunks):
            w /= total_weight
            w = w.to(device)
            c = c.to(device)
            fused_chunk += w * c

        return fused_chunk


class FirstWins(BlendingModule):
    """
    Overwrites tiles giving priority to tiles seen earlier.
    Given a tile layout:
        [[1, 2, 3],
        [4, 5, 6]]
    And a raster order:
        [3, 6, 2, 5, 1, 4]

    A chunk at the intersection of {2, 3, 5, 6} would be
    colored in reverse raster order: 5 -> 2 -> 6 -> 3.
    """

    def __init__(
        self,
        tile_layout: list[list[int]],
        tile_raster_order: Optional[list[int]] = None
    ) -> None:
        super().__init__()
        """
        tile_layout: 2D array of tile ids
        tile_raster_order:
            Overrides default top->down, right->left raster order.
        """

        # Default raster order
        # top->down, right->left
        # + -- x
        # |
        # y
        y_length = len(tile_layout)
        x_length = len(tile_layout[0])
        for x in reversed(range(x_length)):
            for y in range(y_length):
                tile_id = tile_layout[y][x]
                if tile_id != -1:
                    self.tile_raster_order.append(tile_id)

        # Override raster order
        if tile_raster_order:
            layout_ids = set([item for row in tile_layout for item in row])
            order_ids = set(tile_raster_order)

            if layout_ids.intersection(order_ids) != order_ids:
                raise ValueError(f"""Provided raster order do not match the tile layout provided.
                                   Tile Layout: {tile_layout},
                                   Raster Order: {tile_raster_order}""")

            self.tile_raster_order: list[int] = tile_raster_order

    def blend(
        self, chunks: list[torch.Tensor], device: torch.device, kwargs={}
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        chunks: 5d tensor(s) in 11zyx order
        kwargs:
            chunk_tile_ids:
                list of tile ids corresponding to each chunk

        Returns
        -------
        fused_chunk: combined chunk
        """

        chunk_lut: dict[int, torch.Tensor] = {}
        chunk_tile_ids = kwargs["chunk_tile_ids"]
        for c_id, chunk in zip(chunk_tile_ids, chunks):
            chunk_lut[c_id] = chunk

        # Coloring in reverse order
        composite_chunk: torch.Tensor = torch.zeros_like(chunks[0])
        for tile_id in reversed(self.tile_raster_order):
            if tile_id in chunk_tile_ids:
                curr_chunk = chunk_lut[tile_id]
                composite_chunk = torch.where(curr_chunk != 0, curr_chunk, composite_chunk)

        return composite_chunk
