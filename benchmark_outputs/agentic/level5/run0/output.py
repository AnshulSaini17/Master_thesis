import math
import tensordyne.ir as ir
import tensordyne.nn as nn
from tensordyne.nn.modules.linear import BufferizedLinear
from tensordyne.nn._bufferize import bufferize


def _build_linear(module, x, name):
    import tensordyne.nn.distributed as dist
    if isinstance(x, dist.MultiDeviceTensor):
        return module(x, name=name)
    if any(not isinstance(dim, int) for dim in x.shape):
        return nn.linear(x, module.weight, module.bias, name=name, feature_axis=module.feature_axis)
    return module(x, name=name)


def _bufferized_add(x, y):
    if not all(isinstance(dim, int) for dim in x.shape):
        return nn.add(x, y)
    return bufferize(args=(x, y), fn=nn.add, output_type=x.output_type,
                     tile_size=x.shape,
                     required_inputs_fn=lambda *t: (t, t))


class TorchLlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_dim,
        num_kv_heads=None,
        rope_base=10000.0,
        eps=1e-6,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_kv_groups = num_heads // self.num_kv_heads

        self.hidden_dim = hidden_dim
        self.rope_base = rope_base
        self.eps = eps

        # RMSNorm weights (plain Parameters to match PyTorch weight names)
        self.attn_norm_weight = nn.Parameter((embed_dim,), ir.Fp32())
        self.ffn_norm_weight = nn.Parameter((embed_dim,), ir.Fp32())

        # Attention projections
        self.q_proj = BufferizedLinear(embed_dim, num_heads * self.head_dim, bias=False, dtype=ir.Fp32())
        self.k_proj = BufferizedLinear(embed_dim, self.num_kv_heads * self.head_dim, bias=False, dtype=ir.Fp32())
        self.v_proj = BufferizedLinear(embed_dim, self.num_kv_heads * self.head_dim, bias=False, dtype=ir.Fp32())
        self.o_proj = BufferizedLinear(num_heads * self.head_dim, embed_dim, bias=False, dtype=ir.Fp32())

        # SwiGLU MLP
        self.gate_proj = BufferizedLinear(embed_dim, hidden_dim, bias=False, dtype=ir.Fp32())
        self.up_proj = BufferizedLinear(embed_dim, hidden_dim, bias=False, dtype=ir.Fp32())
        self.down_proj = BufferizedLinear(hidden_dim, embed_dim, bias=False, dtype=ir.Fp32())

    def _apply_rope(self, x, seq_len):
        """Apply LLaMA-style RoPE (adjacent-pair interleaved) to x: (B, H, S, head_dim)"""
        half_dim = self.head_dim // 2

        # Compute inv_freq: (half_dim,)
        k = nn.cumsum(nn.ones((half_dim,), ir.Fp32()), axis=0) - 1.0
        inv_freq = nn.exp(-(k * (2.0 / self.head_dim)) * math.log(self.rope_base))

        # Positions: (seq_len,)
        positions = nn.cumsum(nn.ones((seq_len,), ir.Fp32()), axis=0) - 1.0

        # freqs: (seq_len, half_dim)
        freqs = nn.matmul(
            nn.reshape(positions, shape=(seq_len, 1)),
            nn.reshape(inv_freq, shape=(1, half_dim))
        )

        # cos/sin: (seq_len, half_dim) -> repeat_interleave -> (seq_len, head_dim)
        cos_half = nn.cos(freqs)
        sin_half = nn.sin(freqs)
        cos_full = nn.repeat_interleave(cos_half, 2, axis=-1)   # (seq_len, head_dim)
        sin_full = nn.repeat_interleave(sin_half, 2, axis=-1)   # (seq_len, head_dim)

        # Reshape for broadcasting with x: (B, H, S, head_dim)
        cos_full = nn.reshape(cos_full, shape=(1, 1, seq_len, self.head_dim))
        sin_full = nn.reshape(sin_full, shape=(1, 1, seq_len, self.head_dim))

        # rotate_half: unfold x (B,H,S,D) -> (B,H,S,D//2,2)
        # after unfold: even-indexed in pos 0, odd-indexed in pos 1 of last dim
        x_view = nn.unfold(x, axis=-1, new_axis_size=2)               # (B, H, S, half_dim, 2)
        x_even_part, x_odd_part = nn.split(x_view, [1, 1], axis=-1)   # each (B,H,S,half_dim,1)
        x_even = nn.fold(x_even_part, axis=-1)                         # (B, H, S, half_dim)
        x_odd = nn.fold(x_odd_part, axis=-1)                           # (B, H, S, half_dim)
        # rotate_half: [-x_odd[0], x_even[0], -x_odd[1], x_even[1], ...]
        x_rotated = nn.interleave(-x_odd, x_even, axis=-1)             # (B, H, S, head_dim)

        return x * cos_full + x_rotated * sin_full

    def build(self, x: nn.Tensor) -> nn.Tensor:
        """x: (batch, seq_len, embed_dim)"""
        seq_len = x.shape[1]

        # ---- Attention block ----
        residual = x
        x_norm = nn.rmsnorm(x, normalized_shape=self.embed_dim, weight=self.attn_norm_weight, eps=self.eps)

        # QKV projections
        Q = _build_linear(self.q_proj, x_norm, 'q_proj')   # (B, S, num_heads*head_dim)
        K = _build_linear(self.k_proj, x_norm, 'k_proj')   # (B, S, num_kv_heads*head_dim)
        V = _build_linear(self.v_proj, x_norm, 'v_proj')   # (B, S, num_kv_heads*head_dim)

        # Reshape + transpose: (B,S,H*hd) -> (B,H,S,hd)
        Q = nn.permute(nn.unfold(Q, axis=-1, new_axis_size=self.head_dim), axes=(0, 2, 1, 3))
        K = nn.permute(nn.unfold(K, axis=-1, new_axis_size=self.head_dim), axes=(0, 2, 1, 3))
        V = nn.permute(nn.unfold(V, axis=-1, new_axis_size=self.head_dim), axes=(0, 2, 1, 3))

        # Apply RoPE to Q and K
        Q = self._apply_rope(Q, seq_len)
        K = self._apply_rope(K, seq_len)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_groups > 1:
            K = nn.repeat_interleave(K, self.num_kv_groups, axis=1)
            V = nn.repeat_interleave(V, self.num_kv_groups, axis=1)

        # Scaled dot-product attention (causal)
        attn_out = nn.scaled_dot_product_attention(Q, K, V, is_causal=True)  # (B,H,S,hd)

        # Merge heads: (B,H,S,hd) -> (B,S,H,hd) -> (B,S,embed_dim)
        attn_out = nn.fold(nn.permute(attn_out, axes=(0, 2, 1, 3)), axis=-1)

        # Output projection
        attn_out = _build_linear(self.o_proj, attn_out, 'o_proj')

        # Residual
        x = _bufferized_add(residual, attn_out)

        # ---- MLP block ----
        residual = x
        x_norm = nn.rmsnorm(x, normalized_shape=self.embed_dim, weight=self.ffn_norm_weight, eps=self.eps)

        # SwiGLU MLP
        gate = _build_linear(self.gate_proj, x_norm, 'gate_proj')
        up = _build_linear(self.up_proj, x_norm, 'up_proj')
        mlp_out = _build_linear(self.down_proj, nn.silu(gate) * up, 'down_proj')

        # Residual
        x = _bufferized_add(residual, mlp_out)

        return x
