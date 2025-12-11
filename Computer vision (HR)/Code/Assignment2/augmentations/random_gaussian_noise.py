#Uses pillow (you can also use another imaging library if you want)
import matplotlib.pyplot as plt
# import torchaudio
import numpy as np
import torch
import math
import torchvision
import cv2

class RandomGaussianNoise():
    def __init__(self, probability=1, mean=0.5, variance=0.5, invert=True):
        """
        Adds psuedo-random gaussian noise before clipping it to [0, inf] and normalizing to [0, 1]
        :param probability: the probability per image to have this augmentation applied 
        :param mean: mean of the gaussian noise
        :param variance: variance of the gaussian noise
        :param invert: inverts the input tensor before and after the application of noise, this affects the overall brightness of the output due to the intermediate clipping and normalization
        """
        self.probability = probability
        self.mean = mean
        self.variance = variance
        self.inverted = invert
    
    def __call__(self, images: torch.Tensor):
        """
        :param images: a tensor of images. Can be a single image or a batch, must be grayscale. Must be in range [0, 1]
        :return: a noisified tensor of the same shape as the input tensor
        """
        wasBatch = True

        # Ensure batch format - store original formatting for reformatting output
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
            wasBatch = False

        # Ensure grayscale - reduce dimensions to [b, H, W]
        if images.shape[1] != 1:    raise ValueError("Input tensor is not grayscale: ", images.shape)
                
        if self.inverted: 
            images = 1 - images

        # Apply a noise tensor only on places where the random_mask equals true.
        # This is the same as calling a random threshold for every image
        noise_tensor = torch.randn(images.shape) * (self.variance ** 2)
        random_mask = torch.rand(size=(images.shape[0],)) < self.probability
        images += (random_mask*noise_tensor)
        # Noise may cause small values to decrease below 0, so let us clip those
        images = torch.clip(images, min=0)
        # Noise may cause large values to increase above 1, so let us normalize the image tensor to [0, 1]
        images /= torch.max(images)

        if self.inverted:
             images = 1 - images

        # Ensure correct output format
        if wasBatch:
            return images
        else:
            return images.squeeze(1)


# Test script:
if __name__ == "__main__":
    image_path = r"C:\Users\thijs\University\Year5\Period4\HR\HandwritingRecognition\Assignment2\data\IAM\img\a01-000u-00.png"
    image_path = r"C:\Users\thijs\University\Year5\Period4\HR\HandwritingRecognition\Assignment2\data\IAM\img\a01-000x-03.png"
    image = cv2.imread(image_path)
    
    tensify = torchvision.transforms.ToTensor()
    grayify = torchvision.transforms.Grayscale()
    noisify = RandomGaussianNoise(probability=1, mean=0.5, variance=0.5, invert=True)

    image_batch = tensify(image).unsqueeze(0) # [1, 3, 154, 1584]
    gray_batch = grayify(image_batch) # [1, 1, 154, 1584]
    noise_batch = noisify(gray_batch)
    print(f"Done : {noise_batch.shape}")
    plt.imshow(noise_batch[0][0])
    plt.show()

    single_image = tensify(image)
    gray_image = grayify(single_image)

    noise_image = noisify(gray_image)
    print(f"Done : {noise_image.shape}")
    plt.imshow(noise_image[0])
    plt.show()