import torch
import torch.nn.functional as F


class TorchLlamaDecoderLayer(torch.nn.Module):
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
        self.attn_norm_weight = torch.nn.Parameter(torch.ones(embed_dim))
        self.ffn_norm_weight = torch.nn.Parameter(torch.ones(embed_dim))

        # Attention projections
        self.q_proj = torch.nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

        # SwiGLU MLP
        self.gate_proj = torch.nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = torch.nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = torch.nn.Linear(hidden_dim, embed_dim, bias=False)

    def _rmsnorm(self, x, weight):
        # x: (..., embed_dim)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * weight

    def _rotate_half(self, x):
        # x: (..., head_dim), head_dim must be even
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out = torch.stack((-x2, x1), dim=-1)
        return out.flatten(-2)

    def _apply_rope(self, x):
        """x: (batch, heads, seq_len, head_dim)"""
        bsz, n_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        device = x.device
        dtype = x.dtype

        inv_freq = 1.0 / (
            self.rope_base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim/2)

        cos = torch.cos(freqs).to(dtype).repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
        sin = torch.sin(freqs).to(dtype).repeat_interleave(2, dim=-1)  # (seq_len, head_dim)

        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)

        return (x * cos) + (self._rotate_half(x) * sin)

    def _repeat_kv(self, x):
        """x: (batch, num_kv_heads, seq_len, head_dim)
        -> (batch, num_heads, seq_len, head_dim)
        """
        if self.num_kv_heads == self.num_heads:
            return x
        bsz, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(bsz, num_kv_heads, self.num_kv_groups, seq_len, head_dim)
        return x.reshape(bsz, self.num_heads, seq_len, head_dim)

    def forward(self, x, attn_mask=None):
        """x: (batch, seq_len, embed_dim)

        attn_mask: optional additive mask broadcastable to
                   (batch, num_heads, seq_len, seq_len)
                   values should be 0 for keep and -inf (or very negative) for mask
        """
        bsz, seq_len, _ = x.shape

        # ---- Attention block ----
        residual = x
        x_norm = self._rmsnorm(x, self.attn_norm_weight)

        Q = self.q_proj(x_norm)  # (batch, seq, num_heads * head_dim)
        K = self.k_proj(x_norm)  # (batch, seq, num_kv_heads * head_dim)
        V = self.v_proj(x_norm)  # (batch, seq, num_kv_heads * head_dim)

        Q = Q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        Q = self._apply_rope(Q)
        K = self._apply_rope(K)

        # Repeat KV heads for GQA
        K = self._repeat_kv(K)
        V = self._repeat_kv(V)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.head_dim**0.5)

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Optional external mask
        if attn_mask is not None:
            scores = scores + attn_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch, heads, seq, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        attn_output = self.o_proj(attn_output)

        x = residual + attn_output

        # ---- MLP block ----
        residual = x
        x_norm = self._rmsnorm(x, self.ffn_norm_weight)

        gate = self.gate_proj(x_norm)
        up = self.up_proj(x_norm)
        mlp_out = self.down_proj(F.silu(gate) * up)

        x = residual + mlp_out
        return x
