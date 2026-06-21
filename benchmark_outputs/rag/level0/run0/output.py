import tensordyne.nn as nn


class SingleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def build(self, x):
        return self.fc.build(x)