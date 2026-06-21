import math
import tensordyne.ir as ir
import tensordyne.nn as nn
from tensordyne.nn.modules.linear import BufferizedLinear
from tensordyne.nn._bufferize import bufferize


def _build_linear(module, x, name):
    if any(not isinstance(dim, int) for dim in x.shape):
        return nn.linear(x, module.weight, module.bias, name=name, feature_axis=module.feature_axis)
    return module(x, name=name)


def _bufferized_add(x, y):
    if not all(isinstance(dim, int) for dim in x.shape):
        return nn.add(x, y)
    return bufferize(args=(x, y), fn=nn.add, output_type=x.output_type,
                     tile_size=x.shape,
                     required_inputs_fn=lambda *t: (t, t))


class TorchSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = BufferizedLinear(embed_dim, embed_dim, bias=True, dtype=ir.Fp32())
        self.k_proj = BufferizedLinear(embed_dim, embed_dim, bias=True, dtype=ir.Fp32())
        self.v_proj = BufferizedLinear(embed_dim, embed_dim, bias=True, dtype=ir.Fp32())
        self.out_proj = BufferizedLinear(embed_dim, embed_dim, bias=True, dtype=ir.Fp32())

    def build(self, x: nn.Tensor) -> nn.Tensor:
        Q = _build_linear(self.q_proj, x, 'q_proj')
        K = _build_linear(self.k_proj, x, 'k_proj')
        V = _build_linear(self.v_proj, x, 'v_proj')

        # (batch, seq, dim) @ (batch, dim, seq) -> (batch, seq, seq)
        scores = nn.matmul(Q, nn.transpose(K, axes=(-2, -1)))

        # Scale by sqrt(d_k)
        scores = scores / math.sqrt(self.embed_dim)

        # Softmax over last dimension
        attn_weights = nn.softmax(scores, axis=-1)

        # Weighted sum of values
        attn_output = nn.matmul(attn_weights, V)

        # Final linear projection
        return _build_linear(self.out_proj, attn_output, 'out_proj')
