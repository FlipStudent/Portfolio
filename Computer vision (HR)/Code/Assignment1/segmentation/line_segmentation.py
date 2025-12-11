import numpy as np
import cv2
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

from . import default_params as p
from .connected_components import ConnectedComponentsGenerator
from .character_segmentation import segment_characters
from .get_image import get_all_images
from .invert_image import invert_image_8bit


def segment_lines(character_crops: dict[tuple, np.ndarray], bandwidth: float = p.KDE_BANDWIDTH
                  ) -> np.ndarray:
    """
    Segment the characters in an image into one or more text lines. Calculates the center of gravity
        of the characters, takes the y-coordinate and clusters on those using kernel density
        estimation

    :param character_crops: Dictionary with the top-left corner coordinates of the characters and
        their image crops
    :param bandwidth: Smoothing width of the kernel density estimation
    :return: 2-D array that has one or more text lines. In each line there are the segmented
        characters crops as np.ndarray
    """
    crops_in_lines, _ = segment_lines_and_get_split_lines(character_crops, bandwidth)
    return crops_in_lines


def segment_lines_and_get_split_lines(character_crops: dict[tuple, np.ndarray],
                                      bandwidth: float = p.KDE_BANDWIDTH
                                      ) -> tuple[np.ndarray, list]:
    """
    Segment the characters in an image into one or more text lines. Calculates the center of gravity
        of the characters, takes the y-coordinate and clusters on those using kernel density
        estimation

    :param character_crops: Dictionary with the top-left corner coordinates of the characters and
        their image crops
    :param bandwidth: Smoothing width of the kernel density estimation
    :return: Tuple with: (1) 2-D array that has one or more text lines. In each line there are the
        segmented characters crops as np.ndarray. (2) List of y-coordinates that are the boundaries
        of the different text line clusters.
    """
    center_to_corner_map = {}
    for corner_coord, crop in character_crops.items():
        center_to_corner_map[center_of_gravity(crop, corner_coord)] = corner_coord

    corner_clusters, split_lines = _make_line_clusters(center_to_corner_map, bandwidth)
    crops_in_lines = _crops_clusters_from_corner_clusters(corner_clusters, character_crops)
    return crops_in_lines, split_lines


def center_of_gravity(crop: np.ndarray, corner: tuple[int, int]) -> tuple[float, float]:
    """
    Calculates the center of gravity of a grayscale or binary image. Background is assumed to be
        white.

    :param crop: Image or image crop, white background black letter
    :param corner: The top-left corner coordinate of the image crop related to an original image
    :return: Center of gravity as x,y coordinate as related to the original image
    """
    stability_term = 1e-5  # Added to divisor to prevent division by zero
    inverted_im = invert_image_8bit(crop)  # White letters desired to get accurate center of gravity
    moment_info = cv2.moments(inverted_im)
    center_x_local = moment_info["m10"] / (moment_info["m00"] + stability_term)
    center_y_local = moment_info["m01"] / (moment_info["m00"] + stability_term)
    return corner[0] + center_x_local, corner[1] + center_y_local


def kernel_density_estimation(center_map: dict[tuple, tuple], bandwidth: float,
                              linspace_multiplier: int = 10) -> list:
    """
    https://en.wikipedia.org/wiki/Kernel_density_estimation
    Gaussian kernel density estimation on the y-coordinates of centers. Smooths the point
      coordinates to estimate them as a probability density function.

    :param center_map: Dictionary with centers and corner coordinates
    :param bandwidth: The smoothing width for each point
    :param linspace_multiplier: Scalar to the discreteness of the probability density function
    :return: Local minima of the probability density function. Can be used as the boundaries of
        clusters.
    """
    y_values = [coord[1] for coord in center_map.keys()]
    y_reshaped = np.array(y_values).reshape(-1, 1)

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(y_reshaped)
    space = np.linspace(min(y_values) - 2 * bandwidth, max(y_values) + 2 * bandwidth,
                        num=linspace_multiplier * len(y_values))
    log_probs = kde.score_samples(space.reshape(-1, 1))

    local_mins = argrelextrema(log_probs, np.less)[0]
    split_lines = space[local_mins]

    # _show_kernel_density_estimation(space, log_probs, local_mins)
    return split_lines


def make_line_segmentation_visual(crops_in_lines: np.ndarray, im_path: str,
                                  crops_with_coords: dict[tuple, np.ndarray],
                                  split_lines: list) -> np.ndarray:
    """
    Makes an image representing of the original image with the line segmentation result.

    :param crops_in_lines: 2-D array that has one or more text lines. In each line there are the
        segmented characters crops as np.ndarray
    :param im_path: Path to the original image
    :param crops_with_coords: The corner coordinates and crops of segmented characters
    :param split_lines: List of y-coordinates of where a text line ends and a new one starts
    :return: Image with the line segmentation result
    """
    image_shape = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE).shape
    colour_cycle = ((255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (255, 255, 0), (255, 0, 255), (0, 255, 255))
    colour_dict = {}
    centers_per_line = []
    for idx, line in enumerate(crops_in_lines):
        colour = colour_cycle[idx % len(colour_cycle)]
        centers = []
        for compare_crop in line:
            compare_crop = cv2.cvtColor(compare_crop, cv2.COLOR_GRAY2RGB)
            for coords, crop_gray in crops_with_coords.items():
                crop = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)
                if np.array_equal(crop, compare_crop):
                    mask = np.all(crop == [0, 0, 0], axis=-1)
                    crop[mask] = colour
                    colour_dict[coords] = crop
                    centers.append(center_of_gravity(crop_gray, coords))
        centers_per_line.append(centers)

    lines_im = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    for coord, crop in colour_dict.items():
        left_x, top_y = coord
        h, w, _ = crop.shape
        lines_im[top_y:top_y + h, left_x:left_x + w, :] += invert_image_8bit(crop)
    lines_im = invert_image_8bit(lines_im)

    for idx, centers in enumerate(centers_per_line):
        for cx, cy in centers:
            cv2.circle(lines_im, (int(cx), int(cy)), 5, (0, 0, 0), -1)

    for draw_line_y in split_lines:
        cv2.line(lines_im, (0, int(draw_line_y)), (99999, int(draw_line_y)), (0, 0, 0))

    return lines_im


def show_line_segmentation_visual(crops_in_lines: np.ndarray, im_path: str,
                                  crops_with_coords: dict[tuple, np.ndarray],
                                  split_lines: list) -> None:
    """
    Shows the original image with the line segmentation result.

    :param crops_in_lines: 2-D array that has one or more text lines. In each line there are the
        segmented characters crops as np.ndarray
    :param im_path: Path to the original image
    :param crops_with_coords: The corner coordinates and crops of segmented characters
    :param split_lines: List of y-coordinates of where a text line ends and a new one starts
    """
    lines_im = make_line_segmentation_visual(crops_in_lines, im_path, crops_with_coords,
                                             split_lines)
    plt.imshow(lines_im)
    plt.show()


def _make_line_clusters(center_map: dict[tuple, tuple], bandwidth: float
                        ) -> tuple[list[list], list]:
    """
    Makes clusters based on y-coordinate of the center of gravity of characters.
    """
    split_lines = kernel_density_estimation(center_map, bandwidth)
    clusters = []
    prev_valley = 0
    for valley in split_lines:
        centers_in_line = dict([(center, corner) for (center, corner) in center_map.items()
                                if prev_valley <= center[1] < valley])
        # ascending sort by x-coordinate of center
        centers_in_line = dict(sorted(centers_in_line.items(), key=lambda item: item[0][0]))
        clusters.append(centers_in_line.values())
        prev_valley = valley
    clusters.append([corner for (center, corner) in center_map.items()
                     if prev_valley <= center[1]])

    return clusters, split_lines


def _crops_clusters_from_corner_clusters(corner_clusters: list[list],
                                         character_crops: dict[tuple, np.ndarray]) -> np.ndarray:
    """
    Helper functions that transverses dictionary mappings. Returns a 2-D array with text lines, each
        line contains the character crop in that line.
    """
    crops_in_lines = []
    for cluster in corner_clusters:
        line = []
        for corner_coord in cluster:
            line.append(character_crops[corner_coord])
        crops_in_lines.append(line)
    return np.array(crops_in_lines, dtype=object)


def _show_center_of_gravity_projection(im_path: str) -> None:
    ccg = ConnectedComponentsGenerator(im_path)
    crops = ccg.connected_components_crops()
    corner_coords = ccg.connected_components_corners()
    character_crops = segment_characters(crops, corner_coords, ccg.char_length_estimate())

    cx_list = []
    cy_list = []
    for corner_coord, crop in character_crops.items():
        cx, cy = center_of_gravity(crop, corner_coord)
        cx_list.append(cx)
        cy_list.append(cy)

    radius = 10
    x_zero_offset = 10
    image = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for cx, cy in zip(cx_list, cy_list):
        cv2.circle(image, (int(cx), int(cy)), radius, (255, 0, 0), -1)
        cv2.circle(image, (x_zero_offset, int(cy)), radius, (0, 255, 0), -1)  # projection to y-axis
    image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    cv2.imshow("center of gravity to y projected", image)
    cv2.waitKey(0)


def _show_kernel_density_estimation(linspace: np.ndarray, log_probs: np.ndarray,
                                    local_minima: tuple) -> None:
    plt.plot(linspace, log_probs)
    plt.plot(linspace[local_minima], log_probs[local_minima], 'ro')
    plt.xlabel("y coordinate of character center")
    plt.ylabel("log likelihood")
    plt.show()


def _save_line_segmentation_visuals():
    """
    For every scroll, create an image with the line segmentation result and save them
    """
    save_folder = "line_segment_images/"
    scrolls = get_all_images()
    for idx, scroll in enumerate(scrolls):
        # _show_center_of_gravity_projection(scroll)

        ccg = ConnectedComponentsGenerator(scroll)
        crops = ccg.connected_components_crops()
        corner_coords = ccg.connected_components_corners()
        character_crops = segment_characters(crops, corner_coords, ccg.char_length_estimate())
        char_crops_in_lines, split_lines = segment_lines_and_get_split_lines(character_crops)

        lines_im = make_line_segmentation_visual(char_crops_in_lines, scroll, character_crops,
                                                 split_lines)
        cv2.imwrite(f"{save_folder}line_segmentation_scroll_{idx}.png", lines_im)

        # show_line_segmentation_visual(char_crops_in_lines, image.shape, character_crops,
        #                               split_lines)


if __name__ == "__main__":
    _save_line_segmentation_visuals()
