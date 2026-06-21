"""PyTorch reference for the abs Triton kernel: y = abs(x)."""

import torch
import torch.nn as nn


class TritonAbsRef(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs()
