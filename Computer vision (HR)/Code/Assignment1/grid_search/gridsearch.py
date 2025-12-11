from pipeline import HebrewPipeline
from segmentation.main import grid_search_segmentation_1

if __name__ == "__main__":
    absolute_path_to_this_file = ""
    pipeline = HebrewPipeline(model_path=absolute_path_to_this_file + "data/conv-0",
                                name_unicode_path=absolute_path_to_this_file + "data/name_unicode.json",
                                label_char_names=absolute_path_to_this_file + "data/labels_char_names.json")

    image_dictionary = grid_search_segmentation_1("data/test_data/124-Fg004.jpg")
    pipeline.gridsearch_process_scroll(image_dictionary)
