import torch


class TorchSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # (batch, seq, dim) @ (batch, dim, seq) -> (batch, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale by sqrt(d_k)
        scores = scores / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))

        # Softmax over last dimension
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)

        # Final linear projection
        return self.out_proj(attn_output)
