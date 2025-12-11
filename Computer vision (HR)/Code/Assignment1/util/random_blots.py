#Uses pillow (you can also use another imaging library if you want)
import torch
from .font_generator import *
import torchvision
import cv2
import skimage

# CREDITS TO: fmw42 (April 14, 2022), https://stackoverflow.com/questions/71865493/is-it-possible-to-create-a-random-shape-on-an-image-in-python

class RandomAddBlots():
    def __init__(self, probability=1, blot_smallness=3, elipse_kernel_shape=(9,9)):
        self.probability = probability
        self.blot_smallness = blot_smallness/100
        self.elipse_kernel_shape = elipse_kernel_shape
    
    def __call__(self, images: torch.Tensor):
        for idx, image_tensor in enumerate(images):
            if np.random.rand() > self.probability:
                continue
            # create random noise image
            im_sh = image_tensor.shape
            noise = np.array(np.random.rand(im_sh[0]*4, im_sh[1]*4) > (0.50 + self.blot_smallness), dtype= np.uint8)

            # blur the noise image to control the size
            blur = cv2.GaussianBlur(noise, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT)

            # stretch the blurred image to full dynamic range
            stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)

            # threshold stretched image to control the size
            thresh = cv2.threshold(stretch, 100, 255, cv2.THRESH_BINARY)[1]

            # apply morphology open and close to smooth out shapes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.elipse_kernel_shape)
            blot_layer = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            blot_layer = cv2.morphologyEx(blot_layer, cv2.MORPH_CLOSE, kernel)
            blot_tensor = torch.tensor(blot_layer/255, dtype=torch.uint8)
            resize = torchvision.transforms.Resize(im_sh)
            blot_tensor = resize(blot_tensor[None,None,:,:]).squeeze()
            images[idx] = torch.bitwise_or(blot_tensor, torch.tensor(image_tensor, dtype=torch.uint8))
            plt.imshow(images[idx])
            plt.show()
        return images
    
if __name__ == "__main__":
    font_path = './data/font/habbakuk/Habbakuk.TTF'
    print(os.path.exists(font_path))
    generator = FontGenerator(font_path=font_path, letter_size=50, letter_padding=5)
    label, img = generator.generate_random_character()
    torch.tensor(img)
    process = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RandomAddBlots(probability=0.5, blot_smallness=2, elipse_kernel_shape=(9,9))
    ])
    final_product = process(img)
    print(f"Done : {final_product.shape}")
    plt.imshow(final_product[0])
    plt.show()