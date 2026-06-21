"""PyTorch reference: sum all rows together. y[j] = sum_i x[i, j]."""

import torch
import torch.nn as nn


class TritonRowAccumulateRef(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=0, keepdim=True)
