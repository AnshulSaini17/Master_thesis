import torch


class TorchRoPE(torch.nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim = head_dim
        self.base = base

    def forward(self, x):
        """x: (batch, num_heads, seq_len, head_dim)
        returns: same shape, with rotary position embedding applied
        """
        bsz, num_heads, seq_len, head_dim = x.shape
        assert head_dim == self.head_dim

        device = x.device
        dtype = x.dtype

        # Positions: (seq_len,)
        positions = torch.arange(seq_len, device=device, dtype=dtype)

        # Inverse frequencies: (head_dim/2,)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))

        # Angles: (seq_len, head_dim/2)
        angles = torch.outer(positions, inv_freq)

        # cos, sin: (1, 1, seq_len, head_dim/2)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)

        # Split even and odd dimensions
        x_even = x[..., 0::2]  # (batch, heads, seq, head_dim/2)
        x_odd = x[..., 1::2]  # (batch, heads, seq, head_dim/2)

        # Apply rotation
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Interleave even and odd dimensions back together
        out = torch.stack((rotated_even, rotated_odd), dim=-1)
        out = out.flatten(-2)

        return out
