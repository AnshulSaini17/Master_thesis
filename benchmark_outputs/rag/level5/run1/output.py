import tensordyne.nn as nn
from tensordyne.nn.modules.normalization import BufferizedRMSNorm
from tensordyne.nn.modules.normalization import _build_rms_norm


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

        # RMSNorm modules
        self.attn_norm = BufferizedRMSNorm(embed_dim, eps=eps)
        self.ffn_norm = BufferizedRMSNorm(embed_dim, eps=eps)

        # Attention projections
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

        # SwiGLU MLP
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def _apply_rope(self, x):
        """x: (batch, heads, seq_len, head_dim)"""
        inv_freq = nn.pow(
            self.rope_base,
            nn.div(
                nn.arange(0, self.head_dim, 2, dtype=nn.float32()),
                self.head_dim
            )
        )
        inv_freq = nn.rdiv(1.0, inv_freq)

        seq_len = nn.shape(x, axis=2)
        positions = nn.arange(seq_len, dtype=nn.float32())

        # freqs: (seq_len, head_dim/2)
        freqs = nn.outer(positions, inv_freq)

        # repeat each freq twice -> (seq_len, head_dim)
        cos = nn.repeat_interleave(nn.cos(freqs), repeats=2, axis=-1)
        sin = nn.repeat_interleave(nn.sin(freqs), repeats=2, axis=-1)

        # broadcast to (1, 1, seq_len, head_dim)
        cos = nn.reshape(cos, (1, 1, nn.shape(cos, axis=0), self.head_dim))
        sin = nn.reshape(sin, (1, 1, nn.shape(sin, axis=0), self.head_dim))

        # rotate_half: interleave negated odd and even positions
        x_even = nn.slice(x, axis=-1, start=0, stop=None, step=2)   # (..., head_dim/2)
        x_odd  = nn.slice(x, axis=-1, start=1, stop=None, step=2)   # (..., head_dim/2)
        x_neg_odd = nn.neg(x_odd)
        # stack(-x2, x1) then flatten -> interleave
        rotated = nn.stack([x_neg_odd, x_even], axis=-1)
        rotated = nn.flatten(rotated, start_axis=-2)

        return nn.add(nn.mul(x, cos), nn.mul(rotated, sin))

    def _repeat_kv(self, x):
        """x: (batch, num_kv_heads, seq_len, head_dim)
        -> (batch, num_heads, seq_len, head_dim)
        """
        if self.num_kv_heads == self.num_heads:
            return x
        # expand along a new axis then reshape
        # x shape: (batch, num_kv_heads, seq_len, head_dim)
        x = nn.unsqueeze(x, axis=2)
        # (batch, num_kv_heads, 1, seq_len, head_dim)
        x = nn.expand(x, axis=2, times=self.num_kv_groups)
        # (batch, num_kv_heads, num_kv_groups, seq_len, head_dim)
        x = nn.reshape(x, (
            nn.shape(x, axis=0),
            self.num_heads,
            nn.shape(x, axis=3),
            self.head_dim,
        ))
        return x

    def build(self, x, attn_mask=None):
        """x: (batch, seq_len, embed_dim)

        attn_mask: optional additive mask broadcastable to
                   (batch, num_heads, seq_len, seq_len)
        """
        # ---- Attention block ----
        residual = x
        x_norm = _build_rms_norm(self.attn_norm, x)

        Q = self.q_proj.build(x_norm)
        K = self.k_proj.build(x_norm)
        V = self.v_proj.build(x_norm)

        # Reshape to (batch, heads, seq_len, head_dim)
        batch   = nn.shape(x, axis=0)
        seq_len = nn.shape(x, axis=1)

        Q = nn.transpose(
            nn.reshape(Q, (batch, seq_len, self.num_heads,    self.head_dim)),
            axes=(0, 2, 1, 3)
        )
        K = nn.transpose(
            nn.reshape(K, (batch, seq_len, self.num_kv_heads, self.head_dim)),
            axes=(0, 2, 1, 3)
        )
        V = nn.transpose(
            nn.reshape(V, (batch, seq_len, self.num_kv_heads, self.head_dim)),
            axes=(0, 2, 1, 3)
        )

        # Apply RoPE
        Q = self._apply_rope(Q)
        K = self._apply_rope(K)

        # GQA: repeat KV heads
        K = self._repeat_kv(K)
        V = self._repeat_kv(V)

        # Attention scores: (batch, heads, seq_len, seq_len)
        scores = nn.matmul(Q, K.T)
        scores = nn.mul(scores, 1.0 / (self.head_dim ** 0.5))

        # Causal mask
        causal_mask = nn.triu(
            nn.full((seq_len, seq_len), float("-inf")),
            diagonal=1
        )
        scores = nn.add(scores, causal_mask)

        # Optional external mask
        if attn_mask is not None:
            scores = nn.add(scores, attn_mask)

        attn_weights = nn.softmax(scores, axis=-1)
        attn_output = nn.matmul(attn_weights, V)

        # (batch, heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        attn_output = nn.transpose(attn_output, axes=(0, 2, 1, 3))
        attn_output = nn.reshape(attn_output, (batch, seq_len, self.embed_dim))
        attn_output = self.o_proj.build(attn_output)

        x = nn.add(residual, attn_output)

        # ---- MLP block ----
        residual = x
        x_norm = _build_rms_norm(self.ffn_norm, x)

        gate = self.gate_proj.build(x_norm)
        up   = self.up_proj.build(x_norm)
        mlp_out = self.down_proj.build(nn.mul(nn.silu(gate), up))

        x = nn.add(residual, mlp_out)
        return x