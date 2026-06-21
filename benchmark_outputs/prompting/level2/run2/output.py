import tensordyne.nn as nn


class TorchSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def build(self, x):
        Q = self.q_proj.build(x)
        K = self.k_proj.build(x)
        V = self.v_proj.build(x)

        # (batch, seq, dim) @ (batch, dim, seq) -> (batch, seq, seq)
        scores = nn.matmul(Q, nn.transpose(K, -2, -1))

        # Scale by sqrt(d_k)
        scale = nn.insert_literal(self.embed_dim ** 0.5, (), "float32")
        scores = nn.div(scores, scale)

        # Softmax over last dimension
        attn_weights = nn.softmax(scores, dim=-1)

        # Weighted sum of values
        attn_output = nn.matmul(attn_weights, V)

        # Final linear projection
        return self.out_proj.build(attn_output)