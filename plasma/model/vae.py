import torch
from torch import nn, Tensor


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.enc(x)
        mu, sigma = torch.split(x, 16, dim=1)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=0),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dec(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decode = Decoder()

    def forward(self, x: Tensor) -> Tensor:
        # encode
        mu, log_sigma = self.encoder(x)

        # sample
        epsilon = torch.randn_like(mu)
        latent = mu + torch.exp(log_sigma) * epsilon

        # decode
        x = self.decode(latent)
        return x, mu, log_sigma
