import tensordyne.nn as nn
from tensordyne.nn.modules.normalization import BufferizedRMSNorm
from tensordyne.nn.modules.normalization import _build_rms_norm
import tensordyne.ir as ir


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

        # QKV projections (single fused projection for efficiency)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def build(self, x, attn_mask=None):
        """x: (batch, seq, embed_dim)
        attn_mask (optional):
          - shape broadcastable to (batch, num_heads, seq, seq)
          - should contain 0 for keep, and -inf (or very negative) for mask
        """

        # Pre-norm
        if self.use_rmsnorm:
            x_norm = _build_rms_norm(self.norm_weight, x)
        else:
            x_norm = x

        # Project to QKV: (batch, seq, 3*embed_dim)
        qkv = self.qkv_proj.build(x_norm)

        # Split: each is (batch, seq, embed_dim)
        Q, K, V = nn.split(qkv, 3, axis=-1)

        # Reshape to heads:
        # (batch, seq, embed_dim) -> (batch, seq, num_heads, head_dim)
        # then transpose -> (batch, num_heads, seq, head_dim)
        Q = nn.reshape(Q, (nn.shape(Q, 0), nn.shape(Q, 1), self.num_heads, self.head_dim))
        Q = nn.transpose(Q, (0, 2, 1, 3))

        K = nn.reshape(K, (nn.shape(K, 0), nn.shape(K, 1), self.num_heads, self.head_dim))
        K = nn.transpose(K, (0, 2, 1, 3))

        V = nn.reshape(V, (nn.shape(V, 0), nn.shape(V, 1), self.num_heads, self.head_dim))
        V = nn.transpose(V, (0, 2, 1, 3))

        # Attention scores: (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq)
        # -> (batch, heads, seq, seq)
        scores = nn.matmul(Q, K.T)

        # Scale by sqrt(head_dim)
        scale = self.head_dim ** 0.5
        scores = scores * (1.0 / scale)

        # Optional causal mask
        if self.causal:
            seq_len = nn.shape(scores, -1)
            causal_mask = nn.triu(nn.full((seq_len, seq_len), float('-inf')), diagonal=1)
            scores = scores + causal_mask

        # Optional provided attention mask (additive)
        if attn_mask is not None:
            scores = scores + attn_mask

        # Softmax
        attn_weights = nn.softmax(scores, axis=-1)

        # Weighted sum: (batch, heads, seq, seq) @ (batch, heads, seq, head_dim)
        # -> (batch, heads, seq, head_dim)
        attn_output = nn.matmul(attn_weights, V)

        # Merge heads: (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)
        attn_output = nn.transpose(attn_output, (0, 2, 1, 3))
        attn_output = nn.reshape(
            attn_output,
            (nn.shape(attn_output, 0), nn.shape(attn_output, 1), self.embed_dim)
        )

        # Output projection
        out = self.out_proj.build(attn_output)

        # Residual connection
        return x + out