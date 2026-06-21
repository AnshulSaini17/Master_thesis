"""Standalone LlamaForCausalLM matching HuggingFace weight names and computation."""

import torch
import torch.nn.functional as F


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim, rope_theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len):
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, mlp_bias=False):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(torch.nn.Module):
    def __init__(
        self, hidden_size, num_attention_heads, num_key_value_heads, head_dim, attention_bias=False, layer_idx=0
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim**-0.5

        self.q_proj = torch.nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = torch.nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = torch.nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = torch.nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

    def forward(self, hidden_states, cos, sin):
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_key_value_groups > 1:
            k = torch.repeat_interleave(k, self.num_key_value_groups, dim=1)
            v = torch.repeat_interleave(v, self.num_key_value_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class LlamaDecoderLayer(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        intermediate_size = config.intermediate_size
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        attention_bias = getattr(config, "attention_bias", False)
        mlp_bias = getattr(config, "mlp_bias", False)

        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            layer_idx=layer_idx,
        )
        self.mlp = LlamaMLP(hidden_size=hidden_size, intermediate_size=intermediate_size, mlp_bias=mlp_bias)
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        num_hidden_layers = config.num_hidden_layers
        vocab_size = config.vocab_size
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        rope_theta = getattr(config, "rope_theta", 10000.0)

        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta)

    def forward(self, inputs_embeds):
        hidden_states = inputs_embeds
        seq_len = inputs_embeds.shape[1]
        cos, sin = self.rotary_emb(hidden_states, seq_len)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x):
        hidden_states = self.model(x)
        logits = self.lm_head(hidden_states)
        return logits
