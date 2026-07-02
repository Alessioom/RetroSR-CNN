"""
Dataset class for the DIV2K dataset.

This module is responsible for loading the images that will be used
during training and evaluation.

The preprocessing pipeline (cropping, downsampling, tensor conversion)
will be added progressively during the project.
"""

import os

from PIL import Image

from torch.utils.data import Dataset
from src.config import PATCH_SIZE, SCALE_FACTOR
from src.utils.image_utils import load_image, random_crop, bicubic_downsample
from src.utils.image_utils import image_to_tensor

class DIV2KDataset(Dataset):

    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):

        image_name = self.image_files[idx]

        image_path = os.path.join(self.image_dir, image_name)

        image = load_image(image_path)

        hr_patch = random_crop(image, PATCH_SIZE)

        lr_patch = bicubic_downsample(hr_patch, SCALE_FACTOR)

        lr_patch = image_to_tensor(lr_patch)
        hr_patch = image_to_tensor(hr_patch)

        return lr_patch, hr_patch
    
