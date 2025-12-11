import cv2
import numpy as np
import matplotlib.pyplot as plt

from . import default_params as p
from .connected_components import ConnectedComponentsGenerator
from .get_image import get_all_images, IMAGE_PREFIX
from .invert_image import invert_image_8bit


def segment_characters(c_component_crops: list, c_components_corners: list,
                       char_length_estimate: float,
                       char_length_est_multi: float = p.CHARACTER_LENGTH_ESTIMATE_MULTIPLIER,
                       peak_fraction_thresh: float = p.PROJ_PEAK_FRACTION_THRESH
                       ) -> dict[tuple, np.ndarray]:
    """
    Takes a list of connected component crops and splits it into segmented character crops.
        Determines if the width of the component crop is large enough that it maybe needs splitting

    :param c_component_crops: List of connected component crops
    :param c_components_corners: List of top-left corners coordinates of the crops
    :param char_length_estimate: Estimate which roughly corresponds to pixel width of one character
    :param char_length_est_multi: Multiplies with the character length estimator to take component
        crops with larger width that the multiplied value as candidates for splitting
    :param peak_fraction_thresh: Minimum threshold for how high a peak needs to be at both sides of
        a split in the vertical projection before it is accepted
    :return: Dictionary with top-left corner coordinates as indices and the associated crops as
        values
    """
    multi_char_thresh = int(char_length_est_multi * char_length_estimate)
    multi_char_thresh += 1 if multi_char_thresh % 2 == 1 else 0  # needs to be even

    new_crops = {}  # NOTE about dict, what if two crops have the same corner coordinates as keys?
    for crop, corner in zip(c_component_crops, c_components_corners):
        height, width = crop.shape
        if width >= multi_char_thresh:
            split_crops, split_corners = perform_split(crop, corner, multi_char_thresh,
                                                       peak_fraction_thresh)
            for new_crop, new_corner in zip(split_crops, split_corners):
                new_crops[new_corner] = new_crop
        else:
            new_crops[corner] = crop
    return new_crops


def perform_split(crop: np.ndarray, corner: tuple[int, int], multi_char_thresh: int,
                  peak_fraction_thresh: float) -> tuple[list, list]:
    """
    Try to split a multi-character candidate crop.

    :param crop: Image of a multi-character candidate
    :param corner: Top-left corner coordinates of crop
    :param multi_char_thresh: Threshold which roughly corresponds to pixel width of two characters
    :param peak_fraction_thresh: Minimum threshold for how high a peak needs to be at both sides of
        a split in the vertical projection before it is accepted
    :return: list of new crops and list of new top-left corners
    """
    v_projection, added_term = vert_projection_and_added_term(crop, multi_char_thresh)
    min_x = int(np.argmin(v_projection + added_term))  # cast to int just to supress warning

    if not _peak_exists_to_the_left(v_projection, min_x, peak_fraction_thresh) or \
            not _peak_exists_to_the_right(v_projection, min_x, peak_fraction_thresh):
        return [crop], [corner]

    left_split, void, right_split = np.hsplit(crop, [min_x, min_x + 1])
    l_split_width, r_split_width = left_split.shape[1], right_split.shape[1]
    assert l_split_width + r_split_width + 1 == crop.shape[1]
    right_split_corner = (corner[0] + l_split_width + 1, corner[1])

    # trim the bounding box more tightly to the split crop
    left_split, corner = _trim_image(left_split, corner)
    right_split, right_split_corner = _trim_image(right_split, right_split_corner)

    new_crops, new_corners = [], []
    _split_further_and_append(left_split, corner, multi_char_thresh, new_crops, new_corners,
                              peak_fraction_thresh)
    _split_further_and_append(right_split, right_split_corner, multi_char_thresh, new_crops,
                              new_corners, peak_fraction_thresh)

    # _debug_show_split_candidate(crop, v_projection, added_term, min_x, peak_fraction_thresh)
    return new_crops, new_corners


def vert_projection_and_added_term(crop: np.ndarray, multi_char_thresh: int) -> tuple[np.ndarray,
                                                                                      np.ndarray]:
    """
    Gives the vertical projection of a (binary) image along with an added vertical term that is high
        near the edges. Sum the projection with the added term to split on and make the split not
        happen at the left and right edge of the image.

    :param crop: Image of a multi-character candidate
    :param multi_char_thresh: Threshold which roughly corresponds to pixel width of two characters
    :return: Vertical projection and added vertical term of the crop
    """
    _, crop_binary_inv = cv2.threshold(crop, thresh=int(255 / 2), maxval=1,
                                       type=cv2.THRESH_BINARY_INV)
    vert_projection = np.sum(crop_binary_inv, axis=0)
    added_term = _disincentive_split_near_edges_term(vert_projection, multi_char_thresh)

    return vert_projection, added_term


def make_segmented_characters_image(image_path: str,
                                    character_crops: dict[tuple, np.ndarray]) -> np.ndarray:
    """
    Makes an image representing the original image with bounding boxes around the found segmented
      characters.

    :param image_path: Path to the original image
    :param character_crops: Crops of the segmented characters
    :return: Image with bounding boxes around found segmented characters
    """
    red = (255, 0, 0)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    result_im = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # rgb im to add colored boxes

    for corner_coords, crop in character_crops.items():
        x0, y0 = corner_coords
        h, w = crop.shape
        result_im = cv2.rectangle(result_im, (x0, y0), (x0 + w, y0 + h), red, thickness=2)
    return result_im


def show_segmented_characters_image(image_path: str,
                                    character_crops: dict[tuple, np.ndarray]) -> None:
    """
    Show the image with bounding boxes around the found segmented characters

    :param image_path: Path to the original image
    :param character_crops: Crops of the segmented characters
    """
    image = make_segmented_characters_image(image_path, character_crops)
    plt.imshow(image)
    plt.show()


def _disincentive_split_near_edges_term(projection: np.ndarray, width_thresh: int) -> np.ndarray:
    """
    Creates a term that can be added to the vertical projection of a candidate of a multi-character
        crop such that you disincentivize splitting the crop at the start and end

    :param projection: The vertical projection of a (multi-)char crop
    :param width_thresh: Width where the crop is considered a multi-char candidate. Half of this
        width is used as a region at the beginning and end to disincentivize splitting there
    :return: 1D array with the same length as projection
    """
    # When plotted, the returned extra term looks like this:
    #   \           /
    #    \         /
    #     \_______/
    assert len(projection) >= width_thresh
    max_val = np.max(projection)

    descending = np.linspace(max_val, 0, num=(width_thresh // 2))
    ascending = np.linspace(0, max_val, num=(width_thresh // 2))
    middle_zeros = np.zeros(len(projection) - width_thresh)
    assert len(descending) + len(middle_zeros) + len(ascending) == len(projection)
    return np.hstack((descending, middle_zeros, ascending))


def _peak_exists_to_the_left(vert_projection: np.ndarray, argmin: int,
                             fraction_thresh: float) -> bool:
    return any(x >= vert_projection[argmin] / fraction_thresh for x in vert_projection[:argmin])


def _peak_exists_to_the_right(vert_projection: np.ndarray, argmin: int,
                              fraction_thresh: float) -> bool:
    return any(x >= vert_projection[argmin] / fraction_thresh for x in vert_projection[argmin:])


def _split_further_and_append(crop: np.ndarray, corner: tuple[int, int], multi_char_thresh: int,
                              crop_list: list, corner_list: list, fraction_thresh: float) -> None:
    if crop.shape[1] >= multi_char_thresh:
        sub_crops, sub_corners = perform_split(crop, corner, multi_char_thresh, fraction_thresh)
        for sub_crop, sub_corner in zip(sub_crops, sub_corners):
            crop_list.append(sub_crop)
            corner_list.append(sub_corner)
    else:
        crop_list.append(crop)
        corner_list.append(corner)


def _debug_show_split_candidate(crop: np.ndarray, projection: np.ndarray, added_term: np.ndarray,
                                argmin: int, fraction_thresh: float) -> None:
    print(f"l: {_peak_exists_to_the_left(projection, argmin, fraction_thresh)} | "
          f"r: {_peak_exists_to_the_right(projection, argmin, fraction_thresh)}")
    fig = plt.figure(figsize=(6, 3))

    im_color = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    cv2.line(im_color, (argmin, 0), (argmin, 1000), color=(0, 0, 255))
    cv2.imshow("debug", im_color)
    plt.plot(projection)
    plt.plot(projection + added_term)
    plt.axvline(x=argmin, color="black", dashes=(2.0, 1.0))

    plt.legend(["vertical projection", "projection + added term"])
    plt.xlabel("Column of image crop")
    plt.ylabel("Amount of ink in column")

    fig.savefig("test.png", bbox_inches='tight')
    fig.savefig("test.svg", bbox_inches='tight')
    plt.show()


def _trim_image(image: np.ndarray, corner: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    trim_x, trim_y, trim_w, trim_h = cv2.boundingRect(invert_image_8bit(image))
    image = image[trim_y:trim_y + trim_h, trim_x:trim_x + trim_w]
    corner = (corner[0] + trim_x, corner[1] + trim_y)
    return image, corner


def _save_character_segmentation_visuals():
    """
    For every scroll, save all segmented characters in the folder of the scroll they come from. In
        addition, save an image of the whole scroll with character bounding boxes
    """
    scrolls = get_all_images()

    for idx, scroll in enumerate(scrolls):
        ccg = ConnectedComponentsGenerator(scroll)
        crops = ccg.connected_components_crops()
        corner_coords = ccg.connected_components_corners()
        character_crops = segment_characters(crops, corner_coords, ccg.char_length_estimate())

        save_folder_chars = "data_segmented_characters/" + IMAGE_PREFIX[idx] + "/"
        for corner_coords, crop in character_crops.items():
            left, top = corner_coords
            height, width = crop.shape
            x = left + width / 2
            y = top + height / 2
            cv2.imwrite(f"{save_folder_chars}x={x}_y={y}_w={width}_h={height}.png", crop)

        character_crops_im = make_segmented_characters_image(scroll, character_crops)
        save_folder_bbox = "bbox_images/"
        cv2.imwrite(f"{save_folder_bbox}segmented_characters_scroll_{idx}.png", character_crops_im)

        """Uncomment to show results while running this file"""
        # show_segmented_characters_image(scroll, character_crops)


if __name__ == "__main__":
    _save_character_segmentation_visuals()
