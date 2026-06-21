"""PyTorch reference for the clip Triton kernel: y = clamp(x, lo, hi)."""

import torch
import torch.nn as nn


class TritonClipRef(nn.Module):
    def __init__(self, lo: float = -5.0, hi: float = 5.0):
        super().__init__()
        self.lo = lo
        self.hi = hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.lo, max=self.hi)
