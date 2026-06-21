import torch


class TorchMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_rmsnorm=True, eps=1e-8, causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal

        # Pre-norm (optional)
        self.use_rmsnorm = use_rmsnorm
        if use_rmsnorm:
            self.norm_weight = torch.nn.Parameter(torch.ones(embed_dim))
            self.eps = eps

        # QKV projections (single fused projection for efficiency)
        self.qkv_proj = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Output projection
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)

        # Dropout modules
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.resid_dropout = torch.nn.Dropout(dropout)

    def _rmsnorm(self, x):
        # x: (batch, seq, dim)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.norm_weight

    def forward(self, x, attn_mask=None):
        """x: (batch, seq, embed_dim)
        attn_mask (optional):
          - shape broadcastable to (batch, num_heads, seq, seq)
          - should contain 0 for keep, and -inf (or very negative) for mask
        """
        bsz, seq_len, _ = x.shape

        # Pre-norm
        if self.use_rmsnorm:
            x_norm = self._rmsnorm(x)
        else:
            x_norm = x

        # Project to QKV: (batch, seq, 3*dim)
        qkv = self.qkv_proj(x_norm)

        # Split: each is (batch, seq, dim)
        Q, K, V = torch.chunk(qkv, 3, dim=-1)

        # Reshape to heads:
        # (batch, seq, dim) -> (batch, num_heads, seq, head_dim)
        Q = Q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale by sqrt(head_dim)
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=x.device))

        # Optional causal mask
        if self.causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Optional provided attention mask (additive)
        if attn_mask is not None:
            scores = scores + attn_mask

        # Softmax + dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum: (batch, heads, seq, seq) @ (batch, heads, seq, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Merge heads: (batch, heads, seq, head_dim) -> (batch, seq, dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)

        # Output projection + residual dropout
        out = self.out_proj(attn_output)
        out = self.resid_dropout(out)

        # Residual connection
        return x + out
