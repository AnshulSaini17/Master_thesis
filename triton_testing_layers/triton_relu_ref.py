"""PyTorch reference for the ReLU Triton kernel: y = max(x, 0)."""

import torch
import torch.nn as nn


class TritonReluRef(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0)
