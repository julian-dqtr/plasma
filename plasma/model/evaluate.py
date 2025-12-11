import os
from typing import Callable
from .vae import VAE
from .cnn import LatentToPhysics
from torch.utils.data import DataLoader
import torch
from torch import Tensor
import tqdm


def to_cuda(x: Tensor) -> Tensor:
    if torch.cuda.is_available():
        return x.cuda()
    return x


def evaluate_image_model(
    net: VAE,
    test_set: DataLoader,
    save_dir_path: str,
    visualize_img: Callable[[tuple[Tensor, Tensor], str], None],
):
    os.makedirs(save_dir_path, exist_ok=True)
    net.eval()
    with torch.no_grad():
        cpt_tot = 0
        for batch in test_set:
            x = to_cuda(batch[1])
            xhat, _, _ = net(x)
            for cpt in range(xhat.shape[0]):
                visualize_img(
                    (x[cpt], xhat[cpt]),
                    path=os.path.join(
                        save_dir_path, f"test_image_{cpt_tot}.png"
                    ),
                )
                cpt_tot += 1
    net.train()


def evaluate_latent_2_physics(
    vae: VAE, net: LatentToPhysics, test_set: DataLoader
):
    net.eval()
    vae.eval()
    err = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_set, desc="evaluate latent 2 physics"):
            x = to_cuda(batch[1])
            physics = to_cuda(batch[0])

            with torch.no_grad():
                _, mean, log_var = vae(x)

            pred_physics = net(torch.concat([mean, log_var], dim=1))
            err += (
                torch.abs(physics - pred_physics)
                .mean(axis=-1)
                .cpu()
                .detach()
                .numpy()
                .tolist()
            )
    print(f"MAE: {torch.mean(torch.tensor(err))}")
