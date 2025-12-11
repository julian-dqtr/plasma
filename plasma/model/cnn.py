import torch
from torch import nn, Tensor


class PhysicsToLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * 16 * 4 * 4),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.dec(x)
        x = torch.reshape(x, shape=(-1, 32, 4, 4))
        return torch.split(x, 16, dim=1)


class LatentToPhysics(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(2 * 16 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.reshape(x, shape=(-1, 2 * 16 * 4 * 4))
        x = self.dec(x)
        return x
