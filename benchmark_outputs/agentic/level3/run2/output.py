import tensordyne.ir as ir
import tensordyne.nn as nn
from tensordyne.nn.modules.linear import BufferizedLinear
from tensordyne.nn._bufferize import bufferize


def _build_linear(module, x, name):
    if any(not isinstance(dim, int) for dim in x.shape):
        return nn.linear(x, module.weight, module.bias, name=name, feature_axis=module.feature_axis)
    return module(x, name=name)


def _build_rms_norm(module, hidden_states):
    if any(not isinstance(dim, int) for dim in hidden_states.shape):
        upcast = module._update_rms_norm_io_to_fp32 and hidden_states.dtype != ir.Fp32()
        if upcast:
            hidden_states = nn.cast(hidden_states, dtype=ir.Fp32(), name='rmsnorm_upcast')
        hidden_states = nn.rmsnorm(hidden_states, normalized_shape=module.normalized_shape,
                                   weight=module.weight, eps=module.eps, anchor_axis=module.anchor_axis)
        if upcast:
            hidden_states = nn.cast(hidden_states, dtype=module.dtype, name='rmsnorm_downcast')
        return hidden_states
    return module(hidden_states)


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
        if self.use_rmsnorm:
            x_norm = nn.rmsnorm(x, normalized_shape=x.shape[-1], weight=self.norm_weight, eps=self.eps)
        else:
            x_norm = x

        x1 = _build_linear(self.w1, x_norm, 'w1')
        x2 = _build_linear(self.w2, x_norm, 'w2')

        gated = nn.mul(nn.silu(x2), x1)

        out = _build_linear(self.w_out, gated, 'w_out')

        return _bufferized_add(x, out)
