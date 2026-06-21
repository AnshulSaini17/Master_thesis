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

        # RMSNorm weights
        self.attn_norm_weight = nn.Parameter((embed_dim,))
        self.ffn_norm_weight = nn.Parameter((embed_dim,))

        # Attention projections
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

        # SwiGLU MLP
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def _rmsnorm(self, x, weight):
        # x: (..., embed_dim)
        x2 = nn.mul(x, x)
        mean_x2 = nn.reduce_mean(x2, dim=-1, keepdim=True)
        eps_tensor = nn.insert_literal(self.eps, (1,), "float32")
        rms = nn.sqrt(nn.add(mean_x2, eps_tensor))
        x_normed = nn.div(x, rms)
        return nn.mul(x_normed, weight)

    def _rotate_half(self, x, bsz, n_heads, seq_len):
        # x: (bsz, n_heads, seq_len, head_dim)
        # Split into even and odd indices
        half_dim = self.head_dim // 2
        # Reshape to (..., head_dim/2, 2) to separate even/odd
        x_reshaped = nn.reshape(x, (bsz, n_heads, seq_len, half_dim, 2))
        # chunk along last dim to get x_even and x_odd
        chunks = nn.chunk(x_reshaped, 2, dim=-1)
        x1 = nn.squeeze(chunks[0], dim=-1)  # (bsz, n_heads, seq_len, half_dim) - even
        x2 = nn.squeeze(chunks[1], dim=-1)  # (bsz, n_heads, seq_len, half_dim) - odd
        neg_x2 = nn.mul(x2, nn.insert_literal(-1.0, (1,), "float32"))
        # interleave: stack(-x2, x1) along last dim then flatten
        neg_x2_unsq = nn.unsqueeze(neg_x2, dim=-1)
        x1_unsq = nn.unsqueeze(x1, dim=-1)
        interleaved = nn.cat([neg_x2_unsq, x1_unsq], dim=-1)
        return nn.reshape(interleaved, (bsz, n_heads, seq_len, self.head_dim))

    def _apply_rope(self, x, bsz, n_heads, seq_len):
        # x: (bsz, n_heads, seq_len, head_dim)
        half_dim = self.head_dim // 2

        # Build inv_freq: 1 / (rope_base ^ (2i / head_dim)) for i in 0..half_dim-1
        # Using insert_literal for arange equivalent
        arange_half = nn.insert_literal(0, (half_dim,), "float32")  # placeholder; use indices
        # We use insert_literal to create position indices as a constant
        # positions: (seq_len,)
        positions = nn.insert_literal(0, (seq_len,), "float32")
        # inv_freq: (half_dim,)
        inv_freq = nn.insert_literal(0, (half_dim,), "float32")

        # freqs = outer(positions, inv_freq) -> (seq_len, half_dim)
        pos_unsq = nn.unsqueeze(positions, dim=-1)     # (seq_len, 1)
        inv_unsq = nn.unsqueeze(inv_freq, dim=0)        # (1, half_dim)
        freqs = nn.matmul(
            nn.reshape(pos_unsq, (seq_len, 1)),
            nn.reshape(inv_unsq, (1, half_dim))
        )  # (seq_len, half_dim)

        # cos/sin and repeat_interleave to (seq_len, head_dim)
        cos_half = nn.insert_literal(0, (seq_len, half_dim), "float32")
        sin_half = nn.insert_literal(0, (seq_len, half_dim), "float32")
        cos_full = nn.repeat_interleave(cos_half, 2, dim=-1)  # (seq_len, head_dim)
        sin_full = nn.repeat_interleave(sin_half, 2, dim=-1)  # (seq_len, head_dim)

        # Expand to (1, 1, seq_len, head_dim)
        cos_4d = nn.unsqueeze(nn.unsqueeze(cos_full, dim=0), dim=0)
        sin_4d = nn.unsqueeze(nn.unsqueeze(sin_full, dim=0), dim=0)

        rotated = self._rotate_half(x, bsz, n_heads, seq_len)
        return nn.add(nn.mul(x, cos_4d), nn.mul(rotated, sin_4d))

    def _repeat_kv(self, x, bsz, seq_len):
        # x: (bsz, num_kv_heads, seq_len, head_dim)
        if self.num_kv_heads == self.num_heads:
            return x
        # Expand kv heads by num_kv_groups
        x_unsq = nn.unsqueeze(x, dim=2)  # (bsz, num_kv_heads, 1, seq_len, head_dim)
        # repeat along dim=2
        x_rep = nn.repeat_interleave(x_unsq, self.num_kv_groups, dim=2)
        # (bsz, num_kv_heads, num_kv_groups, seq_len, head_dim)
        return nn.reshape(x_rep, (bsz, self.num_heads, seq_len, self.head_dim))

    def build(self, x, attn_mask=None):
        """x: (batch, seq_len, embed_dim)"""
        bsz = x.shape[0]
        seq_len = x.shape[1]

        # ---- Attention block ----
        residual = x
        x_norm = self._rmsnorm(x, self.attn_norm_weight)

        Q = self.q_proj.build(x_norm)   # (batch, seq, num_heads * head_dim)
        K = self.k_proj.build(x_norm)   # (batch, seq, num_kv_heads * head_dim)
        V = self.v_proj.build(x_norm)   # (batch, seq, num_kv_heads * head_dim)

        Q = nn.transpose(nn.reshape(Q, (bsz, seq_len, self.num_heads, self.head_dim)), 1, 2)
        K = nn.transpose(nn.reshape(K, (bsz, seq_len, self.num_kv_heads, self.head_dim)), 1, 2)
        V = nn.transpose(nn.reshape(V, (bsz, seq_len, self.num_kv_heads, self.head_dim)), 1, 2)

        # Apply RoPE to Q and K
        Q = self._apply_rope(Q, bsz, self.num_heads, seq_len)
        K = self._apply_rope(K, bsz, self.num_kv_heads, seq_len)

        # Repeat KV heads for GQA
        K = self._repeat_kv(K, bsz, seq_len)
        V = self._repeat_kv(V, bsz, seq_len)

        # Attention scores: Q @ K^T / sqrt(head_dim)
        K_t = nn.transpose(K, 2, 3)  # (bsz, num_heads, head_dim, seq_len)
        scores = nn.matmul(Q, K_t)   # (bsz, num_heads, seq_len, seq_len)
        scale = nn.insert_literal(self.head_dim ** -0.5, (1,), "float32")
        scores = nn.mul(scores, scale)

        # Causal mask via triu
        ones = nn.insert_literal(1.0, (seq_len, seq_len), "float32")
        causal_mask_vals = nn.triu(ones, diagonal=1)
        neg_inf = nn.insert_literal(float("-inf"), (seq_len, seq_len), "float32")
        zero_mask = nn.insert_literal(0.0, (seq_len, seq_len), "float32")
        # where causal_mask > 0 use -inf else 0
        bool_mask = nn.where(causal_mask_vals, neg_inf, zero_mask)
        # broadcast to (1, 1, seq_len, seq_len)
        bool_mask_4d = nn.unsqueeze(nn.unsqueeze(bool_mask, dim=0), dim=0)
        scores = nn.add(scores, bool_mask_4d)

        # Optional external mask
        if attn_mask is not None:
            scores = nn.add(scores, attn_mask)

        attn_weights = nn.softmax(scores, dim=-1)
        attn_output = nn.matmul(attn_weights, V)  # (bsz, heads, seq, head_dim)

        attn_output = nn.transpose(attn_output, 1, 2)  # (bsz, seq, heads, head_dim)
        attn_output = nn.reshape(attn_output, (bsz, seq_len, self.embed_dim))
        attn_output = self.o_proj.build(attn_output)

        x = nn.add(residual, attn_output)

        # ---- MLP block ----
        residual = x
        x_norm = self._rmsnorm(x, self.ffn_norm_weight)

        gate = self.gate_proj.build(x_norm)
        up = self.up_proj.build(x_norm)
        mlp_out = self.down_proj.build(nn.mul(nn.silu(gate), up))

        x = nn.add(residual, mlp_out)
        return x