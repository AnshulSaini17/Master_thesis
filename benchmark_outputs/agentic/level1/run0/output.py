import tensordyne.ir as ir
import tensordyne.nn as nn
from tensordyne.nn.modules.linear import BufferizedLinear
from tensordyne.nn._bufferize import bufferize


def _build_linear(module, x, name):
    if any(not isinstance(dim, int) for dim in x.shape):
        return nn.linear(x, module.weight, module.bias, name=name, feature_axis=module.feature_axis)
    return module(x, name=name)


class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BufferizedLinear(64, 128, bias=True, dtype=ir.Fp32())
        self.fc2 = BufferizedLinear(128, 128, bias=True, dtype=ir.Fp32())
        self.fc3 = BufferizedLinear(128, 32, bias=True, dtype=ir.Fp32())

    def build(self, x: nn.Tensor) -> nn.Tensor:
        x = nn.relu(_build_linear(self.fc1, x, 'fc1'))
        x = nn.relu(_build_linear(self.fc2, x, 'fc2'))
        x = _build_linear(self.fc3, x, 'fc3')
        return x
