import numpy as np


def invert_image_8bit(image: np.ndarray) -> np.ndarray:
    """Returns an inverted image of type 8-bit unsigned int"""
    return np.ones(image.shape, dtype=np.uint8) * 255 - image
