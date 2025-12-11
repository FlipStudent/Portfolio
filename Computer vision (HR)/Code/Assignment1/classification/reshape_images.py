import os
import cv2
import numpy as np


def resize_and_pad_image(image_path, output_path, desired_height, desired_width):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate new dimensions while maintaining aspect ratio
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:
        new_width = desired_width
        new_height = int(original_height * desired_width/original_width)
    else:
        new_height = desired_height
        new_width = int(original_width * desired_height/original_height)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a white canvas of the desired size
    canvas = 255 * np.ones((desired_height, desired_width), dtype=np.uint8)

    # Calculate the position to paste the resized image
    y_offset = (desired_height - new_height) // 2
    x_offset = (desired_width - new_width) // 2
    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    # Save the padded image
    cv2.imwrite(output_path, canvas)


def process_folder(input_folder, output_folder, desired_height, desired_width):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each folder in the input directory
    for root, dirs, files in os.walk(input_folder):
        for directory in dirs:
            input_dir = os.path.join(root, directory)
            output_dir = os.path.join(output_folder, directory)

            # Create the corresponding output folder
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filenames = os.listdir(input_dir)
            n_files = len(filenames)
            if n_files < 10:
                test_size = 2
            elif n_files < 100:
                test_size = int(n_files * 0.2)
            elif n_files < 200:
                test_size = int(n_files * 0.1)
            else:
                test_size = int(n_files * 0.05)

            test_indices = np.random.choice(n_files, test_size)
            # Process each image in the folder
            for idx, filename in zip(range(n_files), filenames):
                if filename.endswith(".pgm"):
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, filename)
                    resize_and_pad_image(input_path, output_path, desired_height, desired_width)


if __name__ == '__main__':
    # Specify input and output directories
    input_folder = "../data/monkbrill"
    output_folder = "../data/monkbrill-reshaped"

    # Specify desired height and width, similar to mnist
    desired_height = 28
    desired_width = 28

    # Process the folders
    process_folder(input_folder, output_folder, desired_height, desired_width)
