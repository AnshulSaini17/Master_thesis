import math
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


class LlamaDecoderLayer(nn.Module):
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

        # RMSNorm weights (plain Parameters matching PyTorch attribute names)
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
        """Apply rotary position embedding (even/odd interleave style).
        x: (B, H, S, head_dim)
        """
        hd = self.head_dim
        hd2 = hd // 2

        # inv_freq: (hd2,) — rope_base^(-2i/head_dim)
        idx = nn.cumsum(nn.ones((hd2,), ir.Fp32()), axis=0) - 1.0
        inv_freq = nn.exp(-idx * (2.0 * math.log(self.rope_base) / hd))

        # positions: (S,)
        positions = nn.cumsum(nn.ones((seq_len,), ir.Fp32()), axis=0) - 1.0

        # freqs: (S, hd2) = outer(positions, inv_freq)
        freqs = nn.matmul(
            nn.reshape(positions, shape=(seq_len, 1)),
            nn.reshape(inv_freq, shape=(1, hd2))
        )

        # cos/sin with repeat_interleave: (S, hd2) → (S, hd)
        cos_full = nn.repeat_interleave(nn.cos(freqs), 2, axis=-1)
        sin_full = nn.repeat_interleave(nn.sin(freqs), 2, axis=-1)

        # rotate_half(x): even/odd pairs interleaved
        # x: (B, H, S, hd) → unfold → (B, H, S, hd2, 2)
        x_pairs = nn.unfold(x, axis=-1, new_axis_size=2)
        # x_even[..., i, 0] = x[..., 2i],  x_odd[..., i, 0] = x[..., 2i+1]
        x_even, x_odd = nn.split(x_pairs, [1, 1], axis=-1)
        # interleave(-x_odd, x_even) along hd2 dim → (B, H, S, hd, 1)
        rot = nn.interleave(-x_odd, x_even, axis=-2)
        # fold (hd, 1) → (B, H, S, hd)
        rot = nn.fold(rot, axis=-1)

        # x * cos + rotate_half(x) * sin  — (S, hd) broadcasts over (B, H, S, hd)
        return x * cos_full + rot * sin_full

    def build(self, x, attn_mask=None):
        """x: (batch, seq_len, embed_dim)"""
        seq_len = x.shape[1]

        # ---- Attention block ----
        residual = x
        x_norm = nn.rmsnorm(x, normalized_shape=self.embed_dim,
                            weight=self.attn_norm_weight, eps=self.eps)

        Q = _build_linear(self.q_proj, x_norm, 'q_proj')   # (B, S, num_heads * head_dim)
        K = _build_linear(self.k_proj, x_norm, 'k_proj')   # (B, S, num_kv_heads * head_dim)
        V = _build_linear(self.v_proj, x_norm, 'v_proj')   # (B, S, num_kv_heads * head_dim)

        # (B, S, H*D) → unfold → (B, S, H, D) → permute → (B, H, S, D)
        Q = nn.permute(nn.unfold(Q, axis=-1, new_axis_size=self.head_dim), axes=(0, 2, 1, 3))
        K = nn.permute(nn.unfold(K, axis=-1, new_axis_size=self.head_dim), axes=(0, 2, 1, 3))
        V = nn.permute(nn.unfold(V, axis=-1, new_axis_size=self.head_dim), axes=(0, 2, 1, 3))

        # Apply RoPE to Q and K
        Q = self._apply_rope(Q, seq_len)
        K = self._apply_rope(K, seq_len)

        # Repeat KV heads for GQA
        if self.num_kv_groups > 1:
            K = nn.repeat_interleave(K, self.num_kv_groups, axis=1)
            V = nn.repeat_interleave(V, self.num_kv_groups, axis=1)

        # Scaled dot-product attention with causal mask
        attn_out = nn.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, is_causal=True
        )  # (B, H, S, head_dim)

        # Merge heads: permute → (B, S, H, D) → fold → (B, S, embed_dim)
        attn_out = nn.fold(nn.permute(attn_out, axes=(0, 2, 1, 3)), axis=-1)
        attn_out = _build_linear(self.o_proj, attn_out, 'o_proj')

        x = _bufferized_add(residual, attn_out)

        # ---- MLP block ----
        residual = x
        x_norm = nn.rmsnorm(x, normalized_shape=self.embed_dim,
                            weight=self.ffn_norm_weight, eps=self.eps)

        gate = _build_linear(self.gate_proj, x_norm, 'gate_proj')   # (B, S, hidden_dim)
        up   = _build_linear(self.up_proj,   x_norm, 'up_proj')     # (B, S, hidden_dim)
        mlp_out = _build_linear(self.down_proj, nn.silu(gate) * up, 'down_proj')

        x = _bufferized_add(residual, mlp_out)
        return x
