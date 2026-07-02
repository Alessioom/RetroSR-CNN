"""
Utility functions for image processing.

These functions are shared across the entire project.
"""

from PIL import Image
import random
from torchvision.transforms import ToTensor


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from disk.

    Parameters
    ----------
    image_path : str
        Path to the image.

    Returns
    -------
    Image.Image
        Loaded PIL image in RGB format.
    """

    image = Image.open(image_path)

    return image.convert("RGB")

def random_crop(image: Image.Image, patch_size: int) -> Image.Image:
    """
    Extract a random square patch from an image.

    Parameters
    ----------
    image : PIL.Image
        Input image.

    patch_size : int
        Size of the square patch.

    Returns
    -------
    PIL.Image
        Cropped image patch.
    """

    width, height = image.size

    x = random.randint(0, width - patch_size)
    y = random.randint(0, height - patch_size)

    patch = image.crop(
        (
            x,
            y,
            x + patch_size,
            y + patch_size
        )
    )

    return patch

def bicubic_downsample(
    image: Image.Image,
    scale_factor: int
) -> Image.Image:
    """
    Downsample an image using bicubic interpolation.

    Parameters
    ----------
    image : PIL.Image
        High-resolution image.

    scale_factor : int
        Downsampling factor.

    Returns
    -------
    PIL.Image
        Low-resolution image.
    """

    width, height = image.size

    new_width = width // scale_factor
    new_height = height // scale_factor

    lr_image = image.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC
    )

    return lr_image

to_tensor = ToTensor()
def image_to_tensor(image: Image.Image):
    """
    Convert a PIL image into a PyTorch tensor.
    """

    return to_tensor(image)