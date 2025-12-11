import random
import os.path

IMAGE_PREFIX = [
    "P21-Fg006-R-C01-R01",
    "P22-Fg008-R-C01-R01",
    "P106-Fg002-R-C01-R01",
    "P123-Fg001-R-C01-R01",
    "P123-Fg002-R-C01-R01",
    "P166-Fg002-R-C01-R01",
    "P166-Fg007-R-C01-R01",
    "P168-Fg016-R-C01-R01",
    "P172-Fg001-R-C01-R01",
    "P342-Fg001-R-C01-R01",
    "P344-Fg001-R-C01-R01",
    "P423-1-Fg002-R-C01-R01",
    "P423-1-Fg002-R-C02-R01",
    "P513-Fg001-R-C01-R01",
    "P564-Fg003-R-C01-R01",
    "P583-Fg002-R-C01-R01",
    "P583-Fg006-R-C01-R01",
    "P632-Fg001-R-C01-R01",
    "P632-Fg002-R-C01-R01",
    "P846-Fg001-R-C01-R01",
]


def _postfix(im_format: str) -> str:
    match im_format:
        case "binary":
            return "-binarized.jpg"
        case "gray":
            return "-fused.jpg"
        case "color":
            return ".jpg"
        case _:
            raise Exception("invalid image format")


def get_image(idx: int,
              im_format: str = "binary",
              parent_folder: str = "../data/image-data/") -> str:
    """
    :param idx: Index of list of possible scroll images
    :param im_format: Options are "binary", "gray", and "color"
    :param parent_folder: Parent folder where the scroll images are located, may need to be
                          specified based on where the calling script is located
    :return: Path of scroll image
    """
    return parent_folder + IMAGE_PREFIX[idx] + _postfix(im_format)


def get_image_sample(n_sample: int,
                     im_format: str = "binary",
                     parent_folder: str = "../data/image-data/") -> list:
    """
    :param n_sample: Number of images you want to sample
    :param im_format: Options are "binary", "gray", and "color"
    :param parent_folder: Parent folder where the scroll images are located, may need to be
                          specified based on where the calling script is located
    :return: List of paths to sampled scroll images
    """
    prefixes = random.sample(IMAGE_PREFIX, n_sample)
    paths = [(parent_folder + im + _postfix(im_format)) for im in prefixes]
    return paths


def get_all_images(im_format: str = "binary", parent_folder: str = "../data/image-data/") -> list:
    """
    :param im_format: Options are "binary", "gray", and "color"
    :param parent_folder: Parent folder where the scroll images are located, may need to be
                          specified based on where the calling script is located
    :return: List of paths to scroll images
    """
    paths = [(parent_folder + im + _postfix(im_format)) for im in IMAGE_PREFIX]
    return paths


if __name__ == "__main__":
    for path in get_all_images(im_format="binary"):
        assert os.path.isfile(path)
    for path in get_all_images(im_format="gray"):
        assert os.path.isfile(path)
    for path in get_all_images(im_format="color"):
        assert os.path.isfile(path)
    assert len(IMAGE_PREFIX) == 20  # change IMAGE_PREFIX when number of scroll images changes
