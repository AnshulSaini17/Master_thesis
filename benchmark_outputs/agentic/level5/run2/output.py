import math
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

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_dim = hidden_dim
        self.rope_base = rope_base
        self.eps = eps

        # RMSNorm weights — stored as Parameters to match PyTorch weight naming exactly
        self.attn_norm_weight = nn.Parameter((embed_dim,), ir.Fp32())
        self.ffn_norm_weight = nn.Parameter((embed_dim,), ir.Fp32())

        # Attention projections
        self.q_proj = BufferizedLinear(embed_dim, num_heads * self.head_dim, bias=False, dtype=ir.Fp32())
        self.k_proj = BufferizedLinear(embed_dim, self.num_kv_heads * self.head_dim, bias=False, dtype=ir.Fp32())
        self.v_proj = BufferizedLinear(embed_dim, self.num_kv_heads * self.head_dim, bias=False, dtype=ir.Fp32())
        self.o_proj = BufferizedLinear(num_heads * self.head_dim, embed_dim, bias=False, dtype=ir.Fp32())

        # SwiGLU MLP projections
        self.gate_proj = BufferizedLinear(embed_dim, hidden_dim, bias=False, dtype=ir.Fp32())
        self.up_proj = BufferizedLinear(embed_dim, hidden_dim, bias=False, dtype=ir.Fp32())
        self.down_proj = BufferizedLinear(hidden_dim, embed_dim, bias=False, dtype=ir.Fp32())

    def _apply_rope(self, x, seq_len):
        """Apply rotary position embeddings (neox/rotate_half style).
        x: (B, H, S, head_dim)
        """
        half_dim = self.head_dim // 2

        # Build inv_freq: rope_base^(-2i/head_dim) for i in [0, ..., half_dim-1]
        # = exp(i * (-2 * log(rope_base) / head_dim))
        i = nn.cumsum(nn.ones((half_dim,), ir.Fp32()), axis=0) - 1.0  # [0, 1, ..., half_dim-1]
        log_base = math.log(self.rope_base)
        inv_freq = nn.exp(i * (-2.0 * log_base / self.head_dim))       # (half_dim,)

        # Build positions [0, 1, ..., seq_len-1]
        positions = nn.cumsum(nn.ones((seq_len,), ir.Fp32()), axis=0) - 1.0  # (S,)

        # Outer product → (S, half_dim)
        freqs = nn.matmul(
            nn.reshape(positions, shape=(seq_len, 1)),
            nn.reshape(inv_freq, shape=(1, half_dim)),
        )

        # cos/sin with repeat_interleave(2) along last dim → (S, head_dim)
        cos_half = nn.cos(freqs)                                  # (S, half_dim)
        sin_half = nn.sin(freqs)                                  # (S, half_dim)
        cos_full = nn.repeat_interleave(cos_half, 2, axis=-1)    # (S, head_dim)
        sin_full = nn.repeat_interleave(sin_half, 2, axis=-1)    # (S, head_dim)

        # Unsqueeze to (1, 1, S, head_dim) for broadcasting with (B, H, S, head_dim)
        cos_full = nn.unsqueeze(nn.unsqueeze(cos_full, axis=0), axis=0)
        sin_full = nn.unsqueeze(nn.unsqueeze(sin_full, axis=0), axis=0)

        # rotate_half: interleave consecutive pairs, negate odd elements
        # unfold: (B, H, S, head_dim) → (B, H, S, half_dim, 2)
        x_unfolded = nn.unfold(x, axis=-1, new_axis_size=2)
        # split last axis into even (x1) and odd (x2)
        x_even_raw, x_odd_raw = nn.split(x_unfolded, [1, 1], axis=-1)  # each (B,H,S,half_dim,1)
        x_even = nn.squeeze(x_even_raw, axis=-1)  # x1: (B,H,S,half_dim)
        x_odd  = nn.squeeze(x_odd_raw,  axis=-1)  # x2: (B,H,S,half_dim)
        # rotate_half = [-x2[0], x1[0], -x2[1], x1[1], ...]
        rotated = nn.interleave(-x_odd, x_even, axis=-1)  # (B,H,S,head_dim)

        return x * cos_full + rotated * sin_full

    def build(self, x: nn.Tensor, attn_mask=None) -> nn.Tensor:
        """x: (batch, seq_len, embed_dim)"""
        seq_len = x.shape[1]

        # ---- Attention block ----
        residual = x
        x_norm = nn.rmsnorm(
            x,
            normalized_shape=self.embed_dim,
            weight=self.attn_norm_weight,
            eps=self.eps,
        )

        Q = _build_linear(self.q_proj, x_norm, 'q_proj')  # (B, S, num_heads * head_dim)
        K = _build_linear(self.k_proj, x_norm, 'k_proj')  # (B, S, num_kv_heads * head_dim)
        V = _build_linear(self.v_proj, x_norm, 'v_proj')  # (B, S, num_kv_heads * head_dim)

        # Reshape + permute: (B, S, H*hd) → (B, H, S, hd)
        Q = nn.permute(nn.unfold(Q, axis=-1, new_axis_size=self.head_dim), axes=(0, 2, 1, 3))
        K = nn.permute(nn.unfold(K, axis=-1, new_axis_size=self.head_dim), axes=(0, 2, 1, 3))
        V = nn.permute(nn.unfold(V, axis=-1, new_axis_size=self.head_dim), axes=(0, 2, 1, 3))

        # Apply RoPE to Q and K
        Q = self._apply_rope(Q, seq_len)
        K = self._apply_rope(K, seq_len)

        # GQA: repeat K and V heads to match num_heads
        if self.num_kv_groups > 1:
            K = nn.repeat_interleave(K, self.num_kv_groups, axis=1)
            V = nn.repeat_interleave(V, self.num_kv_groups, axis=1)

        # Scaled dot-product attention with causal masking
        attn_out = nn.scaled_dot_product_attention(Q, K, V, is_causal=True)  # (B, H, S, head_dim)

        # Merge heads: (B,H,S,hd) → (B,S,H,hd) → (B,S,embed_dim)
        attn_out = nn.permute(attn_out, axes=(0, 2, 1, 3))
        attn_out = nn.fold(attn_out, axis=-1)
        attn_out = _build_linear(self.o_proj, attn_out, 'o_proj')

        x = _bufferized_add(residual, attn_out)

        # ---- MLP block ----
        residual = x
        x_norm = nn.rmsnorm(
            x,
            normalized_shape=self.embed_dim,
            weight=self.ffn_norm_weight,
            eps=self.eps,
        )

        gate    = _build_linear(self.gate_proj, x_norm, 'gate_proj')
        up      = _build_linear(self.up_proj,   x_norm, 'up_proj')
        mlp_out = _build_linear(self.down_proj, nn.silu(gate) * up, 'down_proj')

        x = _bufferized_add(residual, mlp_out)
        return x
