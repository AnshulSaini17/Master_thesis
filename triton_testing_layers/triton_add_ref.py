"""PyTorch reference for the doubling Triton kernel: y = x + x."""

import torch
import torch.nn as nn


class TritonDoubleRef(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x
