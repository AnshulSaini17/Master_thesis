import torch


class SingleLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32, 32)

    def forward(self, x):
        return self.fc(x)
