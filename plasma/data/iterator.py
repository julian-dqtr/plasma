from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional
import random


class PlasmaDataset(Dataset):
    def __init__(
        self,
        pairs_labels_and_image_paths: list[tuple[tuple[int, int], str]],
        size: tuple[int, int],
    ):
        self.pairs_labels_and_image_paths = pairs_labels_and_image_paths
        self.resize = transforms.Resize(size)

    def __len__(self) -> int:
        return len(self.pairs_labels_and_image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        physics, path = self.pairs_labels_and_image_paths[idx]
        image = Image.open(path)
        image = functional.pil_to_tensor(image).to(torch.float32) / 4095.0
        return torch.tensor(physics).to(torch.float32), self.resize(image)


def get_datasets(
    pairs_labels_and_image_paths: list[tuple[tuple[int, int], str]],
    train_test_split: float = 0.8,
    batch_size: int = 16,
    size: tuple[int, int] = (64, 64),
    random_seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """get train and test loaders

    Args:
        pairs_labels_and_image_paths: output from prepare_raw_data from prepare.py
        train_test_split: fraction (in 0;1) of data to use for training
        batch_size: number of samples per batch
        size: tuple of ints to define image resize target
        random_seed: before splitting the list we shuffle it

    Returns:
        train and test data loaders
    """
    random.seed(random_seed)
    random.shuffle(pairs_labels_and_image_paths)
    train_split = pairs_labels_and_image_paths[
        : int(len(pairs_labels_and_image_paths) * train_test_split)
    ]
    test_split = pairs_labels_and_image_paths[
        int(len(pairs_labels_and_image_paths) * train_test_split) :
    ]
    train_loader = DataLoader(
        PlasmaDataset(train_split, size=size),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        PlasmaDataset(test_split, size=size),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader
