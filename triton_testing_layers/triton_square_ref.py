"""PyTorch reference for the square Triton kernel: y = x * x."""

import torch
import torch.nn as nn


class TritonSquareRef(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x
