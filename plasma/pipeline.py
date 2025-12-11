import model
from torch import Tensor, nn
import torch


def to_cuda(x: Tensor | nn.Module) -> Tensor | nn.Module:
    if torch.cuda.is_available():
        return x.cuda()
    return x


class Pipeline:
    def __init__(
        self,
        vae: model.VAE,
        physics_to_latent: model.PhysicsToLatent,
        latent_to_physics: model.LatentToPhysics,
    ):
        self.vae = vae
        self.physics_to_latent = physics_to_latent
        self.latent_to_physics = latent_to_physics

    def get_physics_from_image(self, image: Tensor):
        raise NotImplementedError

    def get_image_from_physics(self, physics: tuple[int, int]):
        raise NotImplementedError

    def __call__(
        self,
        image: Tensor | None = None,
        physics: tuple[int, int] | None = None,
    ):
        if image is not None:
            return self.get_physics_from_image()
        if physics is not None:
            return self.get_image_from_physics()
        raise ValueError("both image and physics are None")


def launch_pipeline(
    vae_path: str, physics_to_latent_path: str, latent_to_physics_path: str
):
    vae = model.VAE()
    vae.load_state_dict(torch.load(vae_path, weights_only=True))
    to_cuda(vae.eval())

    physics_to_latent = model.PhysicsToLatent()
    physics_to_latent.load_state_dict(
        torch.load(physics_to_latent_path, weights_only=True)
    )
    to_cuda(physics_to_latent.eval())

    latent_to_physics = model.LatentToPhysics()
    latent_to_physics.load_state_dict(
        torch.load(latent_to_physics_path, weights_only=True)
    )
    to_cuda(latent_to_physics.eval())

    pipeline = Pipeline(
        vae=vae,
        physics_to_latent=physics_to_latent,
        latent_to_physics=latent_to_physics,
    )
