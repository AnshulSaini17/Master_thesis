import tensordyne.ir as ir
import tensordyne.nn as nn
from tensordyne.nn.modules.linear import BufferizedLinear
from tensordyne.nn.modules.normalization import BufferizedRMSNorm
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


class TorchSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = BufferizedLinear(embed_dim, embed_dim, bias=True, dtype=ir.Fp32())
        self.k_proj = BufferizedLinear(embed_dim, embed_dim, bias=True, dtype=ir.Fp32())
        self.v_proj = BufferizedLinear(embed_dim, embed_dim, bias=True, dtype=ir.Fp32())
        self.out_proj = BufferizedLinear(embed_dim, embed_dim, bias=True, dtype=ir.Fp32())

    def build(self, x: nn.Tensor) -> nn.Tensor:
        Q = _build_linear(self.q_proj, x, 'q_proj')
        K = _build_linear(self.k_proj, x, 'k_proj')
        V = _build_linear(self.v_proj, x, 'v_proj')

        # scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V
        attn_output = nn.scaled_dot_product_attention(Q, K, V)

        return _build_linear(self.out_proj, attn_output, 'out_proj')
