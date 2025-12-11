import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

from . import default_params as p
from .get_image import get_all_images
from .invert_image import invert_image_8bit


class ConnectedComponentsGenerator:
    """
    Given an image, finds the connected components. Small connected components are merged if they
        are close together.
    Generates image crops of the found connected components and their position in the original
        image. Request them by calling their associated functions.
    """
    def __init__(self,
                 image_path: str,
                 highly_fragmented_multi: float = p.HIGHLY_FRAGMENTED_MULTIPLIER,
                 blur_cutoff_multi: float = p.AREA_BLUR_CUTOFF_MULTIPLIER,
                 blur_kernel_size_multi: float = p.BLUR_KERNEL_SIZE_MULTIPLIER,
                 leniency_small_components_multi: float = p.LENIENCY_SMALL_COMPONENTS_MULTIPLIER,
                 include_diag_neighbors: bool = True) -> None:
        """
        :param image_path: Path to image to process
        :param highly_fragmented_multi: Multiplier used in determining whether an image has many
            fragments/artifacts
        :param blur_cutoff_multi: Small components get blurred to determine connectivity. This is
            the cutoff threshold between small and large
        :param blur_kernel_size_multi: Multiplier for the blur kernel size
        :param leniency_small_components_multi: Threshold for accepting small components. The
            accepting criteria is based on the amount of ink/pixels in the connected component
        :param include_diag_neighbors: If True, direct diagonal neighbors of pixels count as
            connected
        """
        self.im_path = image_path
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(gray_image, thresh=int(255 / 2), maxval=255,
                                        type=cv2.THRESH_BINARY)
        self.image = binary_image
        self.blur_cutoff_multi = blur_cutoff_multi
        self.highly_fragmented_multi = highly_fragmented_multi
        self.blur_kernel_size_multi = blur_kernel_size_multi
        self.leniency_small_components_multi = leniency_small_components_multi
        self.connectivity = 8 if include_diag_neighbors else 4

        self.data = {}  # gets filled during processing
        self._process()

    def connected_components_crops(self) -> list[np.ndarray]:
        """
        :return: List of image crops of the found connected components
        """
        return self.data["connected_components"]

    def connected_components_corners(self) -> list[tuple[int, int]]:
        """
        :return: List of the top-left corner coordinates (in the original image) of the found
            connected components
        """
        return self.data["top_left_coords"]

    def char_length_estimate(self) -> float:
        """
        :return: Estimate of length of a character in the image
        """
        return self.data["char_length_estimate"]

    def show_connected_components_image(self, blurred_version=False) -> None:
        """
        Show the image with bounding boxes around the found final connected components

        :param blurred_version: If True shows the image with blurred small components
        """
        image = self.make_connected_components_image(blurred_version)
        plt.imshow(image)
        plt.show()

    def make_connected_components_image(self, blurred_version=False) -> np.ndarray:
        """
        Makes an image representing the original image with bounding boxes around the found
            connected components

        :param blurred_version: If True make the image with blurred small components
        """
        red = (255, 0, 0)
        if blurred_version:
            result_im = invert_image_8bit(self.data["im_inv_with_blur_fragments"])
            _, result_im = cv2.threshold(result_im, thresh=254, maxval=255,
                                         type=cv2.THRESH_BINARY)
            result_im = cv2.cvtColor(result_im, cv2.COLOR_GRAY2RGB)  # rgb im to add colored boxes
        else:
            result_im = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        for crop, corner_coord in zip(self.connected_components_crops(),
                                      self.connected_components_corners()):
            x0, y0 = corner_coord
            h, w = crop.shape
            result_im = cv2.rectangle(result_im, (x0, y0), (x0 + w, y0 + h), red, thickness=2)
        return result_im

    def _process(self) -> None:
        """
        Processes the image by finding connected components. Should be called by __init__.
        """
        _, image_inv = cv2.threshold(self.image, thresh=int(255 / 2), maxval=255,
                                     type=cv2.THRESH_BINARY_INV)
        self.data["image_inv"] = image_inv

        bboxes, im_with_comp_ids = self._first_pass_blur_second_pass(image_inv)
        self._make_connected_components_crops(bboxes, im_with_comp_ids)

    def _first_pass_blur_second_pass(self, image_inv: np.ndarray) -> tuple[list, np.ndarray]:
        """
        Does a first pass of finding connected components, blurs small components to merge some of
            them, then does a second pass of finding connected components.
        """
        non_blurred_bboxes, fragment_bboxes, _ = self._connected_component_bboxes(image_inv)
        fragments_to_blur = _filtered_image(image_inv, fragment_bboxes)

        blur_width = int(self.blur_kernel_size_multi * self._first_pass_char_length_estimate())
        blur_width += 1 if blur_width % 2 == 0 else 0  # blur bandwidth must be odd
        blurred_fragments = cv2.GaussianBlur(fragments_to_blur, (blur_width, blur_width), 0)
        image_inv_with_blur_fragments = (blurred_fragments + image_inv).astype(self.image.dtype)
        pass2_big_bboxes, pass2_small_bboxes, im_with_comp_ids = self._connected_component_bboxes(
            image_inv_with_blur_fragments)
        final_bboxes = self._keep_area_check_on_non_blur_im(pass2_big_bboxes + pass2_small_bboxes)

        final_heights = []  # update the char length estimate based on the final bboxes
        for (_, _, _, h) in final_bboxes:
            final_heights.append(h)

        self.data["char_length_estimate"] = np.median(final_heights)
        self.data["non_blurred_bboxes"] = non_blurred_bboxes
        self.data["fragment_bboxes"] = fragment_bboxes
        self.data["final_bboxes"] = final_bboxes
        self.data["im_inv_with_blur_fragments"] = image_inv_with_blur_fragments

        return final_bboxes, im_with_comp_ids

    def _connected_component_bboxes(self, image_inv: np.ndarray) -> tuple[list, list, np.ndarray]:
        """
        Finds bounding boxes of connected components into two categories, small and big. Also gives
            A labeled image with different pixel indexes to each connected component.
        """
        n_comp, im_with_comp_ids, stats, _ = cv2.connectedComponentsWithStats(
            image_inv, connectivity=self.connectivity)

        self._save_component_stats(n_comp, stats)
        area_blur_cutoff = self.blur_cutoff_multi * self._first_pass_char_length_estimate()
        self.data["area_blur_cutoff"] = area_blur_cutoff

        character_blocks_bboxes = []
        fragment_bboxes = []
        for idx in range(1, n_comp):  # ignore index 0 which is the background component
            area = stats[idx, cv2.CC_STAT_AREA]
            bbox = _bounding_box_from_stats(stats, idx)
            if area_blur_cutoff < area:
                character_blocks_bboxes.append(bbox)
            else:
                fragment_bboxes.append(bbox)
        return character_blocks_bboxes, fragment_bboxes, im_with_comp_ids

    def _save_component_stats(self, n_comp: int, stats: np.ndarray) -> None:
        heights = np.empty(n_comp - 1)
        for idx in range(1, n_comp):  # ignore index 0 which is the background component
            heights[idx - 1] = stats[idx, cv2.CC_STAT_HEIGHT]
        self.data["component_height_median"] = np.median(heights)
        self.data["component_height_mean"] = heights.mean()
        self.data["component_height_90th_percentile"] = np.percentile(heights, 90)

    def _first_pass_char_length_estimate(self) -> float:
        """Automatic estimation of the size of a character in the image (first pass, not final)"""
        height_median = self.data["component_height_median"]
        height_mean = self.data["component_height_mean"]
        height_90th_percentile = self.data["component_height_90th_percentile"]

        is_image_highly_fragmented = self.highly_fragmented_multi * height_median < height_mean
        return height_median if not is_image_highly_fragmented else height_90th_percentile

    def _keep_area_check_on_non_blur_im(self, bboxes: list[tuple]) -> list[tuple]:
        """Only keep the bounding boxes who pass the area check on non-blurred image."""
        area_blur_cutoff = self.data["area_blur_cutoff"]
        keep_bboxes = []
        for (x0, y0, w, h) in bboxes:
            im = self.data["image_inv"][y0:y0 + h, x0:x0 + w]
            number_of_pixels = np.sum(im) // 255
            if self.leniency_small_components_multi * area_blur_cutoff < number_of_pixels:
                keep_bboxes.append((x0, y0, w, h))
        return keep_bboxes

    def _make_connected_components_crops(self, bboxes: list, im_with_comp_ids: np.ndarray) -> None:
        connected_components = []
        top_left_coordinates = []
        for (x0, y0, w, h) in bboxes:
            # from the labeled blurred image, make a mask for pixels to consider in original image
            mask = im_with_comp_ids[y0:y0 + h, x0:x0 + w]
            mask = _filter_to_single_connected_component(copy.deepcopy(mask))
            mask = mask.astype(bool)
            inv_crop = np.zeros((h, w), dtype=np.uint8)
            np.putmask(inv_crop, mask, self.data["image_inv"][y0:y0 + h, x0:x0 + w])

            # trim the bounding box more tightly to the nun-blurred image
            trim_x, trim_y, trim_w, trim_h = cv2.boundingRect(inv_crop)
            inv_crop = inv_crop[trim_y:trim_y + trim_h, trim_x:trim_x + trim_w]

            # # Debug visuals
            # plt.imshow(self.data["image_inv"][y0:y0 + h, x0:x0 + w])
            # plt.show()
            # plt.imshow(crop)
            # plt.show()

            connected_components.append(invert_image_8bit(inv_crop))
            top_left_coordinates.append((x0 + trim_x, y0 + trim_y))
        self.data["connected_components"] = connected_components
        self.data["top_left_coords"] = top_left_coordinates


def _bounding_box_from_stats(stats: np.ndarray, idx: int) -> tuple[int, int, int, int]:
    left = stats[idx, cv2.CC_STAT_LEFT]
    top = stats[idx, cv2.CC_STAT_TOP]
    width = stats[idx, cv2.CC_STAT_WIDTH]
    height = stats[idx, cv2.CC_STAT_HEIGHT]
    return left, top, width, height


def _filtered_image(inverted_image: np.ndarray, bboxes: list[tuple]) -> np.ndarray:
    """
    Creates a new image that only had the content of the supplied bounding boxes (else background).

    :param inverted_image: Invert of image
    :param bboxes: Bounding boxes of locations and size of what to keep in image
    :return: New image with only the content kept in the bounding boxes
    """
    filtered_image = np.zeros(shape=inverted_image.shape, dtype=np.uint8)
    for (x0, y0, w, h) in bboxes:
        filtered_image[y0:y0+h, x0:x0+w] = inverted_image[y0:y0+h, x0:x0+w].copy()
    return filtered_image


def _filter_to_single_connected_component(crop: np.ndarray) -> np.ndarray:
    """
    Takes a crop image with one or more "labelled" connected component and filters the image such
        that in the returned image only the biggest connected component remains. Labelled here means
        that connected pixels of one component have a number (label) as their pixel values,
        different connected components have different labels.
        This function is wanted due to when cropping an image on one connected component, part of
        another connected component can be included in the crop.

    :param crop: A crop of an image with the labels of connected components as pixel values.
        Background pixels have value 0
    :return: The crop image with only the biggest connected component remaining (largest area).
        Background pixels have value 0 (black)
    """
    skip_zero_bincount = np.bincount(crop.flatten())[1:]
    keep_label = np.argmax(skip_zero_bincount) + 1
    for idx, row in enumerate(crop):
        for idy, pixel in enumerate(row):
            if pixel != 0 and pixel != keep_label:
                crop[idx][idy] = 0
    filtered_inverted = (crop // keep_label * 255).astype(np.uint8)  # '//' is floor division
    return filtered_inverted


def _save_connected_components_visuals():
    """
    For every scroll, create an image with bounding boxes around connected components and save them
    """
    save_folder = "bbox_images/"
    scrolls = get_all_images()
    for index, scroll in enumerate(scrolls):
        ccg = ConnectedComponentsGenerator(scroll)
        result_image = ccg.make_connected_components_image()
        cv2.imwrite(f"{save_folder}connected_components_scroll_{index}.png", result_image)

        # # For debugging, e.g. show a particular scroll
        # if index >= 17:
        #     ccg.show_connected_components_image(blurred_version=True)


if __name__ == "__main__":
    _save_connected_components_visuals()
