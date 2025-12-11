import torch
from torch import Tensor


def reconstruction(xhat: Tensor, x: Tensor) -> Tensor:
    return torch.pow(xhat - x, 2).sum(dim=(1, 2, 3)).mean()


def kl_div_gauss(log_var: Tensor, mean: Tensor) -> Tensor:
    kl_loss = -0.5 * (1 + log_var - torch.pow(mean, 2) - torch.exp(log_var))
    kl_loss = kl_loss.sum(dim=1).mean()
    return kl_loss


def vae_training_loss(
    xhat: Tensor, x: Tensor, log_var: Tensor, mean: Tensor, beta: float = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    r_loss = reconstruction(xhat=xhat, x=x)
    kl_loss = kl_div_gauss(log_var=log_var, mean=mean)
    return r_loss + beta * kl_loss, r_loss, kl_loss
