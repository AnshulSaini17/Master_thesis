import tensordyne.nn as nn


class TensordyneMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_rmsnorm=True, eps=1e-8, causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        # Pre-norm (optional)
        self.use_rmsnorm = use_rmsnorm
        if use_rmsnorm:
            self.norm_weight = nn.Parameter((embed_dim,))
            self.eps = eps

        # QKV projections (single fused projection for efficiency)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _rmsnorm(self, x):
        # x: (batch, seq, dim)
        x_sq = nn.pow(x, 2)
        mean_sq = nn.mean(x_sq, dim=-1, keepdim=True)
        eps_tensor = nn.insert_literal(self.eps, (1, 1, 1), dtype="float32")
        rms = nn.sqrt(nn.add(mean_sq, eps_tensor))
        return nn.mul(nn.div(x, rms), self.norm_weight)

    def build(self, x, attn_mask=None):
        """x: (batch, seq, embed_dim)
        attn_mask (optional):
          - shape broadcastable to (batch, num_heads, seq, seq)
          - should contain 0 for keep, and -inf (or very negative) for mask
        """
        # Pre-norm
        if self.use_rmsnorm:
            x_norm = self._rmsnorm(x)
        else:
            x_norm = x

        # Project to QKV: (batch, seq, 3*dim)
        qkv = self.qkv_proj.build(x_norm)

        # Split: each is (batch, seq, dim)
        Q, K, V = nn.chunk(qkv, 3, dim=-1)

        # Reshape to heads:
        # (batch, seq, dim) -> (batch, num_heads, seq, head_dim)
        Q = nn.transpose(nn.reshape(Q, (-1, Q.shape[1], self.num_heads, self.head_dim)), 1, 2)
        K = nn.transpose(nn.reshape(K, (-1, K.shape[1], self.num_heads, self.head_dim)), 1, 2)
        V = nn.transpose(nn.reshape(V, (-1, V.shape[1], self.num_heads, self.head_dim)), 1, 2)

        # Causal attention via scaled_dot_product_attention if causal
        if self.causal and attn_mask is None:
            attn_output = nn.scaled_dot_product_attention(Q, K, V, is_causal=True)
        else:
            # Attention scores: (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq)
            scores = nn.matmul(Q, nn.transpose(K, -2, -1))

            # Scale by sqrt(head_dim)
            scale = nn.insert_literal(self.head_dim ** 0.5, (1,), dtype="float32")
            scores = nn.div(scores, scale)

            # Optional causal mask
            if self.causal:
                seq_len = Q.shape[2]
                ones = nn.insert_literal(1.0, (seq_len, seq_len), dtype="float32")
                causal_mask_upper = nn.triu(ones, diagonal=1)
                neg_inf = nn.insert_literal(float("-inf"), (seq_len, seq_len), dtype="float32")
                zero = nn.insert_literal(0.0, (seq_len, seq_len), dtype="float32")
                causal_mask_additive = nn.where(causal_mask_upper, neg_inf, zero)
                scores = nn.add(scores, causal_mask_additive)

            # Optional provided attention mask (additive)
            if attn_mask is not None:
                scores = nn.add(scores, attn_mask)

            # Softmax
            attn_weights = nn.softmax(scores, dim=-1)

            # Weighted sum: (batch, heads, seq, seq) @ (batch, heads, seq, head_dim)
            attn_output = nn.matmul(attn_weights, V)

        # Merge heads: (batch, heads, seq, head_dim) -> (batch, seq, dim)
        attn_output = nn.transpose(attn_output, 1, 2)
        attn_output = nn.reshape(attn_output, (-1, attn_output.shape[1], self.embed_dim))

        # Output projection
        out = self.out_proj.build(attn_output)

        # Residual connection
        return nn.add(x, out)