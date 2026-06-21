"""LLaMA decoder with separate classes — mirrors real HuggingFace structure.

Three classes that reference each other:
  LlamaRMSNorm  (leaf)
  LlamaMLP      (leaf)
  LlamaDecoderLayer  (composes both)
"""

import torch
import torch.nn.functional as F


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class LlamaMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.input_layernorm = LlamaRMSNorm(hidden_size, eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        # --- attention block ---
        residual = x
        x_norm = self.input_layernorm(x)

        Q = self.q_proj(x_norm).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        attn_out = self.o_proj(attn_out)
        x = residual + attn_out

        # --- MLP block ---
        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))

        return x
