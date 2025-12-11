import os
from PIL import Image
import matplotlib.pyplot as plt

def get_min_max_dimensions(folder_path):
    min_width = float('inf')
    min_height = float('inf')
    max_width = 0
    max_height = 0
    min_width_img = None
    min_height_img = None
    max_width_img = None
    max_height_img = None

    # Iterate through all files and subfolders in the folder
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            # Check if the file is a PGM image
            if filename.lower().endswith('.pgm'):
                # Open the image using PIL
                with Image.open(filepath) as img:
                    width, height = img.size
                    # Update min and max dimensions
                    if width < min_width:
                        min_width = width
                        min_width_img = filepath
                    if height < min_height:
                        min_height = height
                        min_height_img = filepath
                    if width > max_width:
                        max_width = width
                        max_width_img = filepath
                    if height > max_height:
                        max_height = height
                        max_height_img = filepath
    print(max_height, max_width, min_width, min_height)
    return (min_width_img, min_height_img), (max_width_img, max_height_img)

# Example usage
main_folder_path = "../data/monkbrill"
min_images, max_images = get_min_max_dimensions(main_folder_path)
print(min_images, max_images)
# Plot and show images with smallest width and height
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(Image.open(min_images[0]))
axs[0, 0].set_title("Smallest Width Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(Image.open(min_images[1]))
axs[0, 1].set_title("Smallest Height Image")
axs[0, 1].axis('off')

# Plot and show images with largest width and height
axs[1, 0].imshow(Image.open(max_images[0]))
axs[1, 0].set_title("Largest Width Image")
axs[1, 0].axis('off')

axs[1, 1].imshow(Image.open(max_images[1]))
axs[1, 1].set_title("Largest Height Image")
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()
