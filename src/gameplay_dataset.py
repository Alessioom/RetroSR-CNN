import random
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from src.config import PATCH_SIZE, SCALE_FACTOR
from src.utils.image_utils import (
    bicubic_downsample,
    bicubic_upsample
)


class GameplayDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

        self.image_paths = []

        extensions = ["*.png", "*.jpg", "*.jpeg"]

        for ext in extensions:
            self.image_paths.extend(
                self.root_dir.rglob(ext)
            )

        self.to_tensor = ToTensor()

    def __len__(self):

        return len(self.image_paths)

    def random_crop(self, image):

        width, height = image.size

        if width < PATCH_SIZE or height < PATCH_SIZE:

            image = image.resize(
                (
                    max(PATCH_SIZE, width),
                    max(PATCH_SIZE, height)
                ),
                Image.BICUBIC
            )

            width, height = image.size

        left = random.randint(0, width - PATCH_SIZE)
        top = random.randint(0, height - PATCH_SIZE)

        return image.crop(
            (
                left,
                top,
                left + PATCH_SIZE,
                top + PATCH_SIZE
            )
        )

    def __getitem__(self, idx):

        image = Image.open(
            self.image_paths[idx]
        ).convert("RGB")

        hr = self.random_crop(image)

        lr = bicubic_downsample(
            hr,
            SCALE_FACTOR
        )

        lr = bicubic_upsample(
            lr,
            SCALE_FACTOR
        )

        return (
            self.to_tensor(lr),
            self.to_tensor(hr)
        )