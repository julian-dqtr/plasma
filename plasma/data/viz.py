import os
from torch import Tensor
from PIL import Image
import torch
import numpy as np
import tqdm


def visualize_img(image: tuple[Tensor, Tensor], path: str):
    image0 = (image[0].clip(0.0, 1.0) * 255).to(torch.uint8)[0]
    image1 = (image[1].clip(0.0, 1.0) * 255).to(torch.uint8)[0]
    img = Image.fromarray(
        torch.concat([image0, image1]).cpu().detach().numpy().astype(np.uint8)
    )
    img.save(path)


def create_gif(
    image_paths: list[str], output_gif_path: str, duration: int = 600
):
    images = [Image.open(image_path) for image_path in image_paths]
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
    )


def cross_epoch_vae_gif(source_dir: str):
    dirs = [
        os.path.join(source_dir, d)
        for d in os.listdir(source_dir)
        if "epoch" in d
    ]
    dirs.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    ref_image_file_names = [
        file_name for file_name in os.listdir(dirs[0]) if ".png" in file_name
    ]
    for ref_image_file_name in tqdm.tqdm(
        ref_image_file_names, desc="save evaluation gifs"
    ):
        create_gif(
            image_paths=[os.path.join(d, ref_image_file_name) for d in dirs],
            output_gif_path=os.path.join(
                source_dir, ref_image_file_name[:-4] + ".gif"
            ),
        )
