"""PyTorch reference for the softmax Triton kernel: y = softmax(x, dim=-1)."""

import torch
import torch.nn as nn


class TritonSoftmaxRef(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x.float(), dim=-1).to(x.dtype)
