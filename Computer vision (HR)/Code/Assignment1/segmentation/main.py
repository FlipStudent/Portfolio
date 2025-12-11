import numpy as np

from . import default_params as p
from .connected_components import ConnectedComponentsGenerator
from .character_segmentation import segment_characters, show_segmented_characters_image
from .line_segmentation import segment_lines_and_get_split_lines, show_line_segmentation_visual


def segment_characters_complete(image_path: str,
                                highly_fragmented_multi: float = p.HIGHLY_FRAGMENTED_MULTIPLIER,
                                blur_cutoff_multi: float = p.AREA_BLUR_CUTOFF_MULTIPLIER,
                                blur_kernel_size_multi: float = p.BLUR_KERNEL_SIZE_MULTIPLIER,
                                leniency_small_components_multi: float = p.LENIENCY_SMALL_COMPONENTS_MULTIPLIER,
                                char_length_est_multi: int = p.CHARACTER_LENGTH_ESTIMATE_MULTIPLIER,
                                peak_fraction_thresh: float = p.PROJ_PEAK_FRACTION_THRESH,
                                bandwidth: float = p.KDE_BANDWIDTH,
                                show_visuals: bool = False
                                ) -> np.ndarray:
    """
    Segment characters. First by collecting connected components and then splitting those. The
        split characters are attributed into different text lines by line segmentation.
    This function has hyperparameters which can be left to their defaults or manually chosen.

    :param image_path: Path to a scroll image that you want to segment
    :param highly_fragmented_multi: Multiplier used in determining whether an image has many
        fragments/artifacts. Typical values between [1.0, 3.0], lower values -> fewer images
        determined as highly fragmented
    :param blur_cutoff_multi: Small components get blurred to determine connectivity. This is the
        cutoff threshold between small and large. Typical values between [8.0, 25.0],
        lower values -> less components blurred
    :param blur_kernel_size_multi: Multiplier for the blur kernel size. Typical values between
        [0.1, 0.4], lower values -> smaller kernel
    :param leniency_small_components_multi: Threshold for accepting small components. The accepting
        criteria is based on the amount of ink/pixels in the connected component. Typical values
        between [0.5, 1.0], lower values -> more small components accepted
    :param char_length_est_multi: Multiplier with the character length estimation to determine
        candidate crops to be split in the character segmentation. Typical values between [1.0, 2.5]
        lower values -> more candidates for splitting
    :param peak_fraction_thresh: Minimum threshold for how high a peak needs to be at both sides of
        a split in the vertical projection before it is accepted. Typical values between [0.25, 0.9]
        (stay below 1.0!), higher values split more
    :param bandwidth: Bandwidth of the kernel density estimation in the line segmentation. A value
        of [25.0] seems to work well across the board to determine the correct number of text lines.
        (Significantly) lower values -> more text lines determined
    :param show_visuals: If True, shows visuals for the character segmentation and line segmentation
    :return: 2-D array that has one or more text lines. In each line there are the segmented
        characters crops as np.ndarray
    """
    return _segment_characters_complete(image_path, highly_fragmented_multi, blur_cutoff_multi,
                                        blur_kernel_size_multi, leniency_small_components_multi,
                                        char_length_est_multi, peak_fraction_thresh, bandwidth,
                                        show_visuals)


def grid_search_segmentation_1(image_path: str) -> dict[str, np.ndarray]:
    """
    Grid search keeping the line segmentation bandwidth and the parameter determining whether a
        scroll is highly fragmented as default (they are expected to work nicely already)
    Runs the process 3^5 = 243 times.
    For a description of what the parameters do, see the :param: comments in the function
        segment_characters_complete()
    :return: Dictionary with as keys the name of the process with abbreviated parameter names and
        values, and as dictionary items the 2-D array that has one or more text lines. In each line
        there are the segmented characters crops as np.ndarray
    """
    blur_cutoff_multi = [8.0, 12.0, 18.0]
    blur_kernel_size_multi = [0.20, 0.30, 0.45]
    leniency_small_components_multi = [0.6, 0.75, 1.00]
    char_length_est_multi = [1.0, 1.5, 2.25]
    peak_fraction_thresh = [0.3, 0.5, 0.8]
    hfm = p.HIGHLY_FRAGMENTED_MULTIPLIER
    kde = p.KDE_BANDWIDTH

    name_result_dict = {}
    for bcm in blur_cutoff_multi:
        for bksm in blur_kernel_size_multi:
            for lscm in leniency_small_components_multi:
                for clem in char_length_est_multi:
                    for pft in peak_fraction_thresh:
                        name = f"hfm={hfm}_bcm={bcm}_bksm={bksm}_lscm={lscm}_clem={clem}_pft={pft}_kde={kde}"
                        name_result_dict[name] = _segment_characters_complete(image_path, hfm, bcm,
                                                                              bksm, lscm, clem, pft,
                                                                              kde, False)
    return name_result_dict


def _segment_characters_complete(image_path: str, highly_fragmented_multi: float,
                                 blur_cutoff_multi: float, blur_kernel_size_multi: float,
                                 leniency_small_components_multi: float,
                                 char_length_est_multi: float, peak_fraction_thresh: float,
                                 bandwidth: float, show_visuals: bool) -> np.ndarray:
    ccg = ConnectedComponentsGenerator(image_path, highly_fragmented_multi, blur_cutoff_multi,
                                       blur_kernel_size_multi, leniency_small_components_multi)
    crops = ccg.connected_components_crops()
    corner_coords = ccg.connected_components_corners()

    character_crops = segment_characters(crops, corner_coords, ccg.char_length_estimate(),
                                         char_length_est_multi, peak_fraction_thresh)

    crops_in_lines, split_lines = segment_lines_and_get_split_lines(character_crops, bandwidth)

    if show_visuals:
        show_segmented_characters_image(image_path, character_crops)
        show_line_segmentation_visual(crops_in_lines, image_path, character_crops, split_lines)

    return crops_in_lines


def _show_test_scrolls():
    im_path_1 = "../data/test_data/25-Fg001.jpg"
    im_path_2 = "../data/test_data/124-Fg004.jpg"
    segment_characters_complete(im_path_1, show_visuals=True)
    segment_characters_complete(im_path_2, show_visuals=True)


def _show_image_data_scrolls():
    from .get_image import get_all_images
    scroll_paths = get_all_images()
    for scroll in scroll_paths:
        segment_characters_complete(scroll, show_visuals=True)


if __name__ == "__main__":
    _show_test_scrolls()
    _show_image_data_scrolls()
