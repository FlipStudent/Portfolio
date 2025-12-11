#Uses pillow (you can also use another imaging library if you want)
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision import transforms
import random
import matplotlib.patches as patches
from .font_transformer import FontTransformer
import torchvision


class FontGenerator():
    def __init__(self, font_path, letter_size, letter_padding):
        self.out_size = letter_size
        self.letter_padding = letter_padding
        # Load the font and set the font size to 42
        self.font = ImageFont.truetype(font_path, 42)
        self.tensify = torchvision.transforms.ToTensor()

        # Character mapping for each of the 27 tokens
        self.char_map = {'Alef' : ')',
                    'Ayin' : '(',
                    'Bet' : 'b',
                    'Dalet' : 'd',
                    'Gimel' : 'g',
                    'He' : 'x',
                    'Het' : 'h',
                    'Kaf' : 'k',
                    'Kaf-final' : '\\',
                    'Lamed' : 'l',
                    'Mem' : '{',
                    'Mem-medial' : 'm',
                    'Nun-final' : '}',
                    'Nun-medial' : 'n',
                    'Pe' : 'p',
                    'Pe-final' : 'v',
                    'Qof' : 'q',
                    'Resh' : 'r',
                    'Samekh' : 's',
                    'Shin' : '$',
                    'Taw' : 't',
                    'Tet' : '+',
                    'Tsadi-final' : 'j',
                    'Tsadi-medial' : 'c',
                    'Waw' : 'w',
                    'Yod' : 'y',
                    'Zayin' : 'z'}

        self.char_to_label = dict((key, i) for i, key in enumerate(self.char_map.keys()))

    # Returns a grayscale image based on specified label of img_size
    def create_image(self, label, img_size):
        if (label not in self.char_map):
            raise KeyError('Unknown label!')

        # Create blank image and create a draw interface
        img = Image.new('L', img_size, 255)
        draw = ImageDraw.Draw(img)

        # Get size of the font and draw the token in the center of the blank image
        # w,h = self.font.getsize(self.char_map[label]) # Deprecated
        left, top, right, bottom = self.font.getbbox(self.char_map[label])
        w = right - left
        h = bottom - top
        draw.text(((img_size[0]-w)/2, (img_size[1]-h)/2), self.char_map[label], 0, self.font)

        transform = transforms.Compose([transforms.PILToTensor()])
        numpy_image = np.array( transform(img)[0] )

        return np.invert(numpy_image) # inverts it to have a 0-valued background TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO

    # Returns a black image with horizontal and vertical lines
    def get_test_image(self, shape):
        image = np.zeros(shape=shape)
        line_width = 2
        for i, row in enumerate(image):
            if len(row) <= i+line_width:
                break
            if i%3 == 0:
                image[i:i+line_width] = np.ones(shape=(2,len(row)))
                image[:,i:i+line_width] = np.ones(shape=(len(row),2))
        return image

    def generate_random_character(self, shape=(50,50)):
        character = random.sample(list(self.char_map), 1)[0]
        img = self.create_image(character, shape)
        return character, img

    def plot_bbox_image(self, bbox, image):
        fig, ax = plt.subplots()
        ax.imshow(image)
        rect = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()


if __name__ == "__main__":
    font_path = '../data/font/habbakuk/Habbakuk.TTF'
    font_path = './HandwritingRecognition/Assignment1/data/font/habbakuk/Habbakuk.TTF'
    print(os.path.exists(font_path))
    # print(os.listdir(''))

    generator = FontGenerator(font_path=font_path, letter_size=50, letter_padding=5)
    transformer = FontTransformer()
    while True:
        label, img = generator.generate_random_character()
        print(f"Original shape: {img.shape}")
        bbox = transformer.get_bbox(img)
        generator.plot_bbox_image(bbox, img)
        transformed_image = transformer.elastic_deformation(img, strength=4)
        print(f"Post shape: {transformed_image.shape}")


        bbox = transformer.get_bbox(transformed_image)
        generator.plot_bbox_image(bbox, transformed_image)
    generator.test_generator()