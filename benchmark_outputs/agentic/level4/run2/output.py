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


class TorchMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_rmsnorm=True, eps=1e-8, causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.use_rmsnorm = use_rmsnorm
        self.eps = eps

        # Pre-norm weight (raw parameter to match PyTorch weight naming exactly)
        if use_rmsnorm:
            self.norm_weight = nn.Parameter((embed_dim,), ir.Fp32())

        # QKV fused projection
        self.qkv_proj = BufferizedLinear(embed_dim, 3 * embed_dim, bias=False, dtype=ir.Fp32())

        # Output projection
        self.out_proj = BufferizedLinear(embed_dim, embed_dim, bias=False, dtype=ir.Fp32())

    def build(self, x: nn.Tensor) -> nn.Tensor:
        # x: (batch, seq, embed_dim)

        # Pre-norm
        if self.use_rmsnorm:
            x_norm = nn.rmsnorm(x, normalized_shape=[self.embed_dim],
                                weight=self.norm_weight, eps=self.eps)
        else:
            x_norm = x

        # Fused QKV projection: (B, S, 3*embed_dim)
        qkv = _build_linear(self.qkv_proj, x_norm, 'qkv_proj')

        # Split into Q, K, V: each (B, S, embed_dim)
        q, k, v = nn.split(qkv, [self.embed_dim, self.embed_dim, self.embed_dim], axis=-1)

        # Reshape to (B, num_heads, S, head_dim)
        q = nn.permute(nn.unfold(q, axis=-1, new_axis_size=self.head_dim), axes=(0, 2, 1, 3))
        k = nn.permute(nn.unfold(k, axis=-1, new_axis_size=self.head_dim), axes=(0, 2, 1, 3))
        v = nn.permute(nn.unfold(v, axis=-1, new_axis_size=self.head_dim), axes=(0, 2, 1, 3))

        # Scaled dot-product attention: (B, num_heads, S, head_dim)
        attn = nn.scaled_dot_product_attention(q, k, v, is_causal=self.causal)

        # Merge heads: (B, num_heads, S, head_dim) -> (B, S, num_heads, head_dim) -> (B, S, embed_dim)
        attn = nn.fold(nn.permute(attn, axes=(0, 2, 1, 3)), axis=-1)

        # Output projection
        out = _build_linear(self.out_proj, attn, 'out_proj')

        # Residual connection
        return _bufferized_add(x, out)
