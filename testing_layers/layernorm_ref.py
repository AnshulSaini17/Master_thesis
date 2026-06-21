import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormRef(nn.Module):
    def __init__(self, num_features: int = 64, eps: float = 1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, [self.num_features], self.weight, self.bias, self.eps)
