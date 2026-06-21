import tensordyne.nn as nn
from tensordyne.nn.modules.normalization import BufferizedRMSNorm
from tensordyne.nn.modules.normalization import _build_rms_norm


class TensordyneLlamaDecoderLayer(nn.Module):
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
        inv_freq = nn.rope_inv_freq(self.head_dim, base=self.rope_base)
        cos, sin = nn.rope_cos_sin(x, inv_freq)
        return nn.apply_rope(x, cos, sin)

    def _repeat_kv(self, x):
        """x: (batch, num_kv_heads, seq_len, head_dim)
        -> (batch, num_heads, seq_len, head_dim)
        """
        if self.num_kv_heads == self.num_heads:
            return x
        return nn.repeat_interleave(x, self.num_kv_groups, axis=1)

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

        # Reshape: (batch, seq, heads*head_dim) -> (batch, heads, seq, head_dim)
        Q = nn.reshape(Q, (nn.shape(Q, 0), nn.shape(Q, 1), self.num_heads, self.head_dim))
        Q = nn.transpose(Q, 1, 2)

        K = nn.reshape(K, (nn.shape(K, 0), nn.shape(K, 1), self.num_kv_heads, self.head_dim))
        K = nn.transpose(K, 1, 2)

        V = nn.reshape(V, (nn.shape(V, 0), nn.shape(V, 1), self.num_kv_heads, self.head_dim))
        V = nn.transpose(V, 1, 2)

        # Apply RoPE
        Q = self._apply_rope(Q)
        K = self._apply_rope(K)

        # Repeat KV heads for GQA
        K = self._repeat_kv(K)
        V = self._repeat_kv(V)

        # Attention scores: (batch, heads, seq, seq)
        scores = nn.matmul(Q, K.T)
        scores = scores * (self.head_dim ** -0.5)

        # Causal mask
        seq_len = nn.shape(x, 1)
        causal_mask = nn.triu(
            nn.full((seq_len, seq_len), float('-inf')),
            diagonal=1,
        )
        scores = scores + causal_mask

        # Optional external mask
        if attn_mask is not None:
            scores = scores + attn_mask

        attn_weights = nn.softmax(scores, axis=-1)
        attn_output = nn.matmul(attn_weights, V)  # (batch, heads, seq, head_dim)

        # (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)
        attn_output = nn.transpose(attn_output, 1, 2)
        attn_output = nn.reshape(
            attn_output,
            (nn.shape(attn_output, 0), nn.shape(attn_output, 1), self.embed_dim),
        )
        attn_output = self.o_proj.build(attn_output)

        x = residual + attn_output

        # ---- MLP block ----
        residual = x
        x_norm = _build_rms_norm(self.ffn_norm, x)

        gate = self.gate_proj.build(x_norm)
        up = self.up_proj.build(x_norm)
        mlp_out = self.down_proj.build(nn.silu(gate) * up)

        x = residual + mlp_out
        return x