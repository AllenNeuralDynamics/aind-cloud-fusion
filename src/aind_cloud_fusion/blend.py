"""
Interface for generic blending.
"""
import torch

class BlendingModule:
    """
    Minimal interface for modular blending function. 
    Subclass can define arbitrary constructors/attributes/members as necessary. 
    """
    def blend(self, chunks: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        raise NotImplementedError("Please implement in BlendingModule subclass.")        

class MaxProjection(BlendingModule): 
    """
    Simplest blending implementation possible. No constructor needed. 
    """

    # NOTE: To prevent GPU OOM, recommended to pass in 2 chunks at a time. 
    def blend(self, chunks: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        Parameters
        ----------
        chunks: list of 3D tensors to combine. Contains 2 or more elements. 

        Returns
        -------
        fused_chunk: combined chunk
        """

        assert len(chunks) > 1, \
            f"Length of input list is {len(chunks)}, blending requires 2 or more chunks."

        fused_chunk = torch.maximum(chunks[0], chunks[1])
        for c in chunks[2:]: 
            fused_chunk = torch.maximum()
        
        return fused_chunk