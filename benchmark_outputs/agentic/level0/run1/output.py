import tensordyne.ir as ir
import tensordyne.nn as nn
from tensordyne.nn.modules.linear import BufferizedLinear


def _build_linear(module, x, name):
    if any(not isinstance(dim, int) for dim in x.shape):
        return nn.linear(x, module.weight, module.bias, name=name, feature_axis=module.feature_axis)
    return module(x, name=name)


class SingleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = BufferizedLinear(32, 32, bias=True, dtype=ir.Fp32())

    def build(self, x: nn.Tensor) -> nn.Tensor:
        return _build_linear(self.fc, x, 'fc')
