"""PyTorch reference for the negate Triton kernel: y = -x."""

import torch
import torch.nn as nn


class TritonNegateRef(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -x
