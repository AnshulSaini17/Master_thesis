import tensordyne.nn as nn
from tensordyne.nn.modules.normalization import BufferizedRMSNorm
from tensordyne.nn.modules.normalization import _build_rms_norm


class TensordyneMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_rmsnorm=True, eps=1e-8, causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.use_rmsnorm = use_rmsnorm

        # Pre-norm (optional)
        if use_rmsnorm:
            self.norm_weight = BufferizedRMSNorm(embed_dim, eps=eps)

        # QKV projection (fused)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def build(self, x, attn_mask=None):
        # Pre-norm
        if self.use_rmsnorm:
            x_norm = _build_rms_norm(self.norm_weight, x)
        else:
            x_norm = x

        # Project to QKV: (batch, seq, 3*embed_dim)
        qkv = self.qkv_proj.build(x_norm)

        # Split into Q, K, V: each (batch, seq, embed_dim)
        Q = nn.slice(qkv, axis=-1, start=0, stop=self.embed_dim)
        K = nn.slice(qkv, axis=-1, start=self.embed_dim, stop=2 * self.embed_dim)
        V = nn.slice(qkv, axis=-1, start=2 * self.embed_dim, stop=3 * self.embed_dim)

        # Reshape: (batch, seq, embed_dim) -> (batch, seq, num_heads, head_dim)
        Q = nn.reshape(Q, (0, 1, self.num_heads, self.head_dim))
        K = nn.reshape(K, (0, 1, self.num_heads, self.head_dim))
        V = nn.reshape(V, (0, 1, self.num_heads, self.head_dim))

        # Transpose: (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        Q = nn.transpose(Q, (0, 2, 1, 3))
        K = nn.transpose(K, (0, 2, 1, 3))
        V = nn.transpose(V, (0, 2, 1, 3))

        # Attention scores: (batch, num_heads, seq, seq)
        # Q @ K^T along last two dims
        scores = nn.matmul(Q, K.T)

        # Scale
        scale = self.head_dim ** 0.5
        scores = scores * (1.0 / scale)

        # Optional causal mask
        if self.causal:
            scores = nn.causal_mask(scores)

        # Optional additive attention mask
        if attn_mask is not None:
            scores = scores + attn_mask

        # Softmax
        attn_weights = nn.softmax(scores, axis=-1)

        # Weighted sum: (batch, num_heads, seq, head_dim)
        attn_output = nn.matmul(attn_weights, V)

        # Transpose back: (batch, num_heads, seq, head_dim) -> (batch, seq, num_heads, head_dim)
        attn_output = nn.transpose(attn_output, (0, 2, 1, 3))

        # Merge heads: (batch, seq, embed_dim)
        attn_output = nn.reshape(attn_output, (0, 1, self.embed_dim))

        # Output projection
        out = self.out_proj.build(attn_output)

        # Residual connection
        return x + out