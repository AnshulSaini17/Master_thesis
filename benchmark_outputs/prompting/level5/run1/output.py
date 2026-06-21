import tensordyne.nn as nn


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

        # RMSNorm layers
        self.attn_norm = nn.RMSNorm(embed_dim, eps=eps)
        self.ffn_norm = nn.RMSNorm(embed_dim, eps=eps)

        # Attention projections
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

        # SwiGLU MLP
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

        # RoPE parameters: inv_freq stored as parameter
        self.inv_freq = nn.Parameter((self.head_dim // 2,))

    def _rotate_half(self, x, bsz, n_heads, seq_len):
        # x: (bsz, n_heads, seq_len, head_dim)
        # Split even and odd indices by reshaping to (..., head_dim/2, 2)
        x_reshaped = nn.reshape(x, (bsz, n_heads, seq_len, self.head_dim // 2, 2))
        # x1 = even indices, x2 = odd indices
        x_chunks = nn.chunk(x_reshaped, 2, dim=4)
        x1 = nn.squeeze(x_chunks[0], dim=4)  # (bsz, n_heads, seq_len, head_dim/2)
        x2 = nn.squeeze(x_chunks[1], dim=4)  # (bsz, n_heads, seq_len, head_dim/2)
        neg_x2 = nn.mul(x2, nn.insert_literal(-1.0, (1,), "float32"))
        # Interleave -x2 and x1 by stacking and reshaping
        # Stack along last dim -> (bsz, n_heads, seq_len, head_dim/2, 2)
        neg_x2_u = nn.unsqueeze(neg_x2, dim=4)
        x1_u = nn.unsqueeze(x1, dim=4)
        stacked = nn.cat([neg_x2_u, x1_u], dim=4)
        return nn.reshape(stacked, (bsz, n_heads, seq_len, self.head_dim))

    def _apply_rope(self, x, bsz, n_heads, seq_len):
        # Build cos/sin from insert_literal for positions and inv_freq
        # positions: (seq_len,)
        positions = nn.insert_literal(0, (seq_len,), "float32")  # placeholder; runtime fills arange
        # inv_freq: (head_dim/2,)
        inv_freq = nn.insert_literal(0, (self.head_dim // 2,), "float32")  # placeholder

        # freqs = outer(positions, inv_freq): (seq_len, head_dim/2)
        pos_col = nn.reshape(positions, (seq_len, 1))
        inv_row = nn.reshape(inv_freq, (1, self.head_dim // 2))
        freqs = nn.matmul(pos_col, inv_row)  # (seq_len, head_dim/2)

        # repeat_interleave to get (seq_len, head_dim)
        cos_half = nn.insert_literal(0, (seq_len, self.head_dim // 2), "float32")
        sin_half = nn.insert_literal(0, (seq_len, self.head_dim // 2), "float32")
        # Use nn.repeat_interleave on freqs-derived cos/sin
        cos_full = nn.repeat_interleave(freqs, 2, dim=1)  # (seq_len, head_dim)
        sin_full = nn.repeat_interleave(freqs, 2, dim=1)  # (seq_len, head_dim)

        # Reshape for broadcast: (1, 1, seq_len, head_dim)
        cos_full = nn.reshape(cos_full, (1, 1, seq_len, self.head_dim))
        sin_full = nn.reshape(sin_full, (1, 1, seq_len, self.head_dim))

        rotated = self._rotate_half(x, bsz, n_heads, seq_len)
        return nn.add(nn.mul(x, cos_full), nn.mul(rotated, sin_full))

    def _repeat_kv(self, x, bsz, seq_len):
        if self.num_kv_heads == self.num_heads:
            return x
        # x: (bsz, num_kv_heads, seq_len, head_dim)
        x_expanded = nn.unsqueeze(x, dim=2)
        # (bsz, num_kv_heads, 1, seq_len, head_dim) -> repeat num_kv_groups times along dim 2
        x_repeated = nn.repeat_interleave(x_expanded, self.num_kv_groups, dim=2)
        return nn.reshape(x_repeated, (bsz, self.num_heads, seq_len, self.head_dim))

    def build(self, x, bsz, seq_len, attn_mask=None):
        """
        x: (bsz, seq_len, embed_dim)
        bsz and seq_len must be passed as concrete ints (no shape unpacking).
        attn_mask: optional additive mask broadcastable to (bsz, num_heads, seq_len, seq_len)
        """

        # ---- Attention block ----
        residual = x
        x_norm = self.attn_norm.build(x)

        Q = self.q_proj.build(x_norm)   # (bsz, seq_len, num_heads * head_dim)
        K = self.k_proj.build(x_norm)   # (bsz, seq_len, num_kv_heads * head_dim)
        V = self.v_proj.build(x_norm)   # (bsz, seq_len, num_kv_heads * head_dim)

        Q = nn.reshape(Q, (bsz, seq_len, self.num_heads, self.head_dim))
        Q = nn.transpose(Q, 1, 2)   # (bsz, num_heads, seq_len, head_dim)

        K = nn.reshape(K, (bsz, seq_len, self.num_kv_heads, self.head_dim))
        K = nn.transpose(K, 1, 2)   # (bsz, num_kv_heads, seq_len, head_dim)

        V = nn.reshape(V, (bsz, seq_len, self.num_kv_heads, self.head_dim))
        V = nn.transpose(V, 1, 2)   # (bsz, num_kv_heads, seq_len, head_dim)

        # Apply RoPE
        Q = self._apply_rope(Q, bsz, self.num_heads, seq_len)
        K = self._apply_rope(K, bsz, self.num_kv_heads, seq_len)

        # Repeat KV heads for GQA
        K = self._repeat_kv(K, bsz, seq_len)
        V = self._repeat_kv(V, bsz, seq_len)

        # Attention scores
        K_t = nn.transpose(K, 2, 3)   # (bsz, num_heads, head_dim, seq_len)
        scores = nn.matmul(Q, K_t)    # (bsz, num_heads, seq_len, seq_len)
        scale = nn.insert_literal(self.head_dim ** -0.5, (1,), "float32")
        scores = nn.mul(scores, scale)

        # Causal mask via triu
        ones = nn.insert_literal(1.0, (seq_len, seq_len), "float32")
        causal_mask_vals = nn.triu(ones, diagonal=1)
        neg_inf = nn.insert_literal(float("-inf"), (seq_len, seq_len), "float32")
        zero_fill = nn.insert_literal(0.0, (seq_len, seq_len), "float32")
        # condition: causal_mask_vals > 0
        condition = nn.triu(ones, diagonal=1)
        causal_additive = nn.where(
            nn.reshape(condition, (1, 1, seq_len, seq_len)),
            nn.reshape(neg_inf, (1, 1, seq_len, seq_len)),
            nn.reshape(zero_fill, (1, 1, seq_len, seq_len)),
        )
        scores = nn.add(scores, causal_additive)

        # Optional external mask
        if attn_mask is not None:
            scores = nn.add(scores, attn_mask)

        attn_weights = nn.softmax(scores, dim=-1)
        attn_output = nn.matmul(attn_weights, V)   # (bsz, num_heads, seq_len, head_dim)

        attn_output = nn.transpose(attn_output, 1, 2)   # (bsz, seq_len, num_heads, head_dim)
        attn_output = nn.reshape(attn_output, (bsz, seq_len, self.embed_dim))
        attn_output = self.o_proj.build(attn_output)

        x = nn.add(residual, attn_output)

        # ---- MLP block ----
        residual = x
        x_norm = self.ffn_norm.build(x)

        gate = self.gate_proj.build(x_norm)
        up = self.up_proj.build(x_norm)
        mlp_out = self.down_proj.build(nn.mul(nn.silu(gate), up))

        x = nn.add(residual, mlp_out)
        return x