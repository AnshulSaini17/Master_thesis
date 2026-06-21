import torch


class TorchGatedMLP(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, use_rmsnorm=True, eps=1e-8):
        super().__init__()

        self.use_rmsnorm = use_rmsnorm
        if use_rmsnorm:
            self.norm_weight = torch.nn.Parameter(torch.ones(embed_dim))
            self.eps = eps

        # Two parallel projections
        self.w1 = torch.nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(embed_dim, hidden_dim, bias=False)

        # Output projection
        self.w_out = torch.nn.Linear(hidden_dim, embed_dim, bias=False)

        self.dropout = torch.nn.Dropout(dropout)

    def _rmsnorm(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.norm_weight

    def forward(self, x):
        # Optional pre-norm
        if self.use_rmsnorm:
            x_norm = self._rmsnorm(x)
        else:
            x_norm = x

        # Parallel projections
        x1 = self.w1(x_norm)  # value
        x2 = self.w2(x_norm)  # gate

        # SwiGLU activation: silu(x2) * x1
        gated = torch.nn.functional.silu(x2) * x1

        out = self.w_out(gated)
        out = self.dropout(out)

        # Residual connection
        return x + out
