import torch


class TransformerFFNBlock(torch.nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim

        # Layer norms
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)

        # FFN layers
        self.fc1 = torch.nn.Linear(embed_dim, ffn_dim)
        self.fc2 = torch.nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        # Pre-norm + residual around FFN
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = x + residual

        # Second norm
        x = self.norm2(x)
        return x
