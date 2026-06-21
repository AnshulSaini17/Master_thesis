import tensordyne.nn as nn
from tensordyne.nn.modules.normalization import BufferizedRMSNorm
from tensordyne.nn._bufferize import bufferize


def _bufferized_add(x, y):
    if not all(isinstance(dim, int) for dim in x.shape):
        return nn.add(x, y)
    return bufferize(args=(x, y), fn=nn.add, output_type=x.output_type,
                     tile_size=x.shape,
                     required_inputs_fn=lambda *t: (t, t))


def _build_rms_norm(module, hidden_states):
    return module.build(hidden_states)


class TorchGatedMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, use_rmsnorm=True, eps=1e-8):
        super().__init__()

        self.use_rmsnorm = use_rmsnorm
        if use_rmsnorm:
            self.norm = BufferizedRMSNorm(embed_dim, eps=eps)

        # Two parallel projections
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(embed_dim, hidden_dim, bias=False)

        # Output projection
        self.w_out = nn.Linear(hidden_dim, embed_dim, bias=False)

    def build(self, x):
        # Optional pre-norm
        if self.use_rmsnorm:
            x_norm = _build_rms_norm(self.norm, x)
        else:
            x_norm = x

        # Parallel projections
        x1 = self.w1.build(x_norm)  # value
        x2 = self.w2.build(x_norm)  # gate

        # SwiGLU activation: silu(x2) * x1
        gated = nn.mul(nn.silu(x2), x1)

        out = self.w_out.build(gated)

        # Residual connection
        return _bufferized_add(x, out)