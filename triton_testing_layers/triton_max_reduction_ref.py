"""PyTorch reference for the max-reduction Triton kernel: y[j] = max_i x[i, j]."""

import torch
import torch.nn as nn


class TritonMaxReductionRef(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=0, keepdim=True).values
