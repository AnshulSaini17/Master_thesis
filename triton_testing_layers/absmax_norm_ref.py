import torch
import torch.nn as nn


class AbsMaxNormRef(nn.Module):
    def __init__(self, num_features: int = 64, eps: float = 1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        absmax = x.abs().max(dim=-1, keepdim=True).values
        return (x / (absmax + self.eps)) * self.weight
