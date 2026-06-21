import tensordyne.nn as nn


class TorchGatedMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, use_rmsnorm=True, eps=1e-8):
        super().__init__()

        self.use_rmsnorm = use_rmsnorm
        if use_rmsnorm:
            self.norm_weight = nn.Parameter((embed_dim,))
            self.eps = eps

        # Two parallel projections
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(embed_dim, hidden_dim, bias=False)

        # Output projection
        self.w_out = nn.Linear(hidden_dim, embed_dim, bias=False)

    def _rmsnorm(self, x):
        x_sq = nn.pow(x, 2)
        mean_sq = nn.mean(x_sq, dim=-1, keepdim=True)
        eps_tensor = nn.insert_literal(self.eps, (1,), "float32")
        rms = nn.sqrt(nn.add(mean_sq, eps_tensor))
        x_normed = nn.div(x, rms)
        return nn.mul(x_normed, self.norm_weight.build())

    def build(self, x):
        # Optional pre-norm
        if self.use_rmsnorm:
            x_norm = self._rmsnorm(x)
        else:
            x_norm = x

        # Parallel projections
        x1 = self.w1.build(x_norm)  # value
        x2 = self.w2.build(x_norm)  # gate

        # SwiGLU activation: silu(x2) * x1
        gated = nn.mul(nn.silu(x2), x1)

        out = self.w_out.build(gated)

        # Residual connection
        return nn.add(x, out)