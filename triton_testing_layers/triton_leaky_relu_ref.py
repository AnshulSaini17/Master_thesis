"""PyTorch reference for the leaky ReLU Triton kernel: y = max(x, alpha * x)."""

import torch
import torch.nn as nn


class TritonLeakyReluRef(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, self.alpha * x)
