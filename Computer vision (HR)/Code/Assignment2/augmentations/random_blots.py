#Uses pillow (you can also use another imaging library if you want)
import matplotlib.pyplot as plt
# import torchaudio
import numpy as np
import torch
import torchvision
import cv2
import skimage

# CREDITS TO: fmw42 (April 14, 2022), https://stackoverflow.com/questions/71865493/is-it-possible-to-create-a-random-shape-on-an-image-in-python

class RandomAddBlots():
    def __init__(self, probability=1.0, blot_smallness=3, elipse_kernel_shape=(9,9)) -> None:
        """
        Adds blots to an input tensor
        :param probability: the probability per image to have this augmentation applied 
        :param blot_smallness: a higher value will result in smaller blots
        :param elipse_kernel_shape: determines the general shape of the blots, (9,9) is round and (9,3) is an eliptoid
        """
        self.probability = probability
        self.blot_smallness = blot_smallness/100
        self.elipse_kernel_shape = elipse_kernel_shape
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: a tensor of images. Can be a single image or a batch, must be grayscale. Must be in range [0, 1]
        :return: a blotted tensor of the same shape as the input tensor
        """
        wasBatch = True

        # Ensure batch format - store original formatting for reformatting output
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
            wasBatch = False

        # Ensure grayscale - reduce dimensions to [b, H, W]
        if images.shape[1] != 1:    raise ValueError("Input tensor is not grayscale: ", images.shape)
        else:                       images = images.squeeze(1) # [b, H, W]


        for idx, image_tensor in enumerate(images):
            if np.random.rand() > self.probability:
                continue
            # create binary noise 4x the size of {image.shape}
            # More pixels are accepted according to the {blot_smallness}: 50 + {blot_smallness} %
            im_sh = image_tensor.shape # [H, W]
            noise = np.array(np.random.rand(im_sh[0]*4, im_sh[1]*4) > (0.50 + self.blot_smallness), dtype= np.uint8)

            # Blur the noise image to create 'islands'
            blur = cv2.GaussianBlur(noise, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT)

            # The intermediate result is an image with roughly-shaped 'holes'
            # To make these into blots (=rounder and more likely real-world shapes), apply a round kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.elipse_kernel_shape)

            # Remove tiny blots (=noise)
            blot_layer = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
            # Remove tiny holes (=> more 'full-bodied' blots)
            blot_layer2 = cv2.morphologyEx(blot_layer, cv2.MORPH_CLOSE, kernel)

            # Binarizing cast
            blot_tensor = torch.tensor(blot_layer2, dtype=torch.uint8)

            # Now shrink the current blot layer to the size of the input image
            resize = torchvision.transforms.Resize(im_sh)
            blot_tensor = resize(blot_tensor.unsqueeze(0)).squeeze()

            # Apply plots with a color that is opposite of the background color
            image_background = round(torch.mean(image_tensor).item())
            if image_background == 1:
                image_tensor = 1-image_tensor
                blotted_tensor = 1 - torch.bitwise_or(blot_tensor, image_tensor>0.2).to(torch.float)
            else:
                blotted_tensor = torch.bitwise_or(blot_tensor, image_tensor>0.2).to(torch.float)
            images[idx] = blotted_tensor

        # Restore original dimensions
        if wasBatch:
            return images.unsqueeze(1)
        else:
            return images
    

# Test script:
if __name__ == "__main__":
    image_path = r"C:\Users\thijs\University\Year5\Period4\HR\HandwritingRecognition\Assignment2\data\IAM\img\a01-000u-00.png"
    image_path = r"C:\Users\thijs\University\Year5\Period4\HR\HandwritingRecognition\Assignment2\data\IAM\img\a01-000x-03.png"
    image = cv2.imread(image_path)
    
    tensify = torchvision.transforms.ToTensor()
    grayify = torchvision.transforms.Grayscale()
    blotify = RandomAddBlots(probability=1.0, blot_smallness=3, elipse_kernel_shape=(9,9))

    image_batch = tensify(image).unsqueeze(0) # [1, 3, 154, 1584]
    gray_batch = grayify(image_batch) # [1, 1, 154, 1584]
    noise_batch = blotify(gray_batch)
    print(f"Done : {noise_batch.shape}")
    plt.imshow(noise_batch[0][0])
    plt.show()

    single_image = tensify(image)
    gray_image = grayify(single_image)

    noise_image = blotify(gray_image)
    print(f"Done : {noise_image.shape}")
    plt.imshow(noise_image[0])
    plt.show()
