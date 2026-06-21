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


class TorchMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_rmsnorm=True, eps=1e-8, causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal
        self.use_rmsnorm = use_rmsnorm

        if use_rmsnorm:
            self.norm_weight = nn.Parameter((embed_dim,), ir.Fp32())
            self.eps = eps

        # QKV fused projection
        self.qkv_proj = BufferizedLinear(embed_dim, 3 * embed_dim, bias=False, dtype=ir.Fp32())

        # Output projection
        self.out_proj = BufferizedLinear(embed_dim, embed_dim, bias=False, dtype=ir.Fp32())

    def build(self, x: nn.Tensor) -> nn.Tensor:
        # Pre-norm
        if self.use_rmsnorm:
            x_norm = nn.rmsnorm(x, normalized_shape=self.embed_dim,
                                weight=self.norm_weight, eps=self.eps)
        else:
            x_norm = x

        # Fused QKV projection: (B, S, 3*D)
        qkv = _build_linear(self.qkv_proj, x_norm, 'qkv_proj')

        # Split into Q, K, V: each (B, S, D)
        q, k, v = nn.split(qkv, [self.embed_dim, self.embed_dim, self.embed_dim], axis=-1)

        # Reshape to multi-head layout: (B, S, D) -> (B, S, H, hd) -> (B, H, S, hd)
        q = nn.unfold(q, axis=-1, new_axis_size=self.head_dim)  # (B, S, H, hd)
        q = nn.permute(q, axes=(0, 2, 1, 3))                    # (B, H, S, hd)
        k = nn.unfold(k, axis=-1, new_axis_size=self.head_dim)
        k = nn.permute(k, axes=(0, 2, 1, 3))
        v = nn.unfold(v, axis=-1, new_axis_size=self.head_dim)
        v = nn.permute(v, axes=(0, 2, 1, 3))

        # Scaled dot-product attention: (B, H, S, hd)
        attn = nn.scaled_dot_product_attention(q, k, v, is_causal=self.causal)

        # Merge heads: (B, H, S, hd) -> (B, S, H, hd) -> (B, S, D)
        attn = nn.permute(attn, axes=(0, 2, 1, 3))  # (B, S, H, hd)
        attn = nn.fold(attn, axis=-1)                # (B, S, D)

        # Output projection
        out = _build_linear(self.out_proj, attn, 'out_proj')

        # Residual connection
        return _bufferized_add(x, out)
