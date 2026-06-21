import tensordyne.nn as nn


class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 32)

    def build(self, x):
        x = nn.relu(self.fc1.build(x))
        x = nn.relu(self.fc2.build(x))
        return self.fc3.build(x)