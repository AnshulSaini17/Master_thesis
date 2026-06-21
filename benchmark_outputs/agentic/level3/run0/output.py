import tensordyne.ir as ir
import tensordyne.nn as nn
from tensordyne.nn.modules.linear import BufferizedLinear
from tensordyne.nn._bufferize import bufferize


def _build_linear(module, x, name):
    if any(not isinstance(dim, int) for dim in x.shape):
        return nn.linear(x, module.weight, module.bias, name=name, feature_axis=module.feature_axis)
    return module(x, name=name)


def _bufferized_add(x, y):
    if not all(isinstance(dim, int) for dim in x.shape):
        return nn.add(x, y)
    return bufferize(args=(x, y), fn=nn.add, output_type=x.output_type,
                     tile_size=x.shape,
                     required_inputs_fn=lambda *t: (t, t))


class TorchGatedMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, use_rmsnorm=True, eps=1e-8):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        if use_rmsnorm:
            self.norm_weight = nn.Parameter((embed_dim,), ir.Fp32())
            self.eps = eps

        self.w1 = BufferizedLinear(embed_dim, hidden_dim, bias=False, dtype=ir.Fp32())
        self.w2 = BufferizedLinear(embed_dim, hidden_dim, bias=False, dtype=ir.Fp32())
        self.w_out = BufferizedLinear(hidden_dim, embed_dim, bias=False, dtype=ir.Fp32())

    def build(self, x: nn.Tensor) -> nn.Tensor:
        # Optional pre-norm
        if self.use_rmsnorm:
            x_norm = nn.rmsnorm(x, normalized_shape=x.shape[-1], weight=self.norm_weight, eps=self.eps)
        else:
            x_norm = x

        # Parallel projections
        x1 = _build_linear(self.w1, x_norm, 'w1')   # value
        x2 = _build_linear(self.w2, x_norm, 'w2')   # gate

        # SwiGLU: silu(gate) * value
        gated = nn.silu(x2) * x1

        # Output projection
        out = _build_linear(self.w_out, gated, 'w_out')

        # Residual connection
        return _bufferized_add(x, out)
