import os
import sys

from pathlib import Path
from os import listdir, mkdir
from pipeline import HebrewPipeline

if __name__ == "__main__":
    print(sys.argv)
    path = sys.argv[1]
    current_directory = os.getcwd()
    absolute_path_to_this_file = str(Path(__file__).parent)
    pipeline = HebrewPipeline(model_path=absolute_path_to_this_file + "/models/full-with-pretraining.pt",
                              name_unicode_path=absolute_path_to_this_file + "/data/name_unicode.json",
                              label_char_names=absolute_path_to_this_file + "/data/labels_char_names.json")
    files_to_proces = listdir(path)
    if not os.path.exists(current_directory + '/results'):
        mkdir(current_directory + '/results')

    for file in files_to_proces:
        print(f"Processing scroll {file}...")
        pipeline.process_scroll(input_path=path + '/' + file, output_path="results/" + Path(file).stem, gen_docx=False)
    print("Processed all scrolls")
