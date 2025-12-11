#Uses pillow (you can also use another imaging library if you want)
import matplotlib.pyplot as plt
# import torchaudio
import numpy as np
import torch
import math
import torchvision
import cv2
from itertools import repeat

class RandomElasticDeformation():
    def __init__(self, strength_reduction=1, probability=1) -> None:
        """
        Adds psuedo-random gaussian noise before clipping it to [0, inf] and normalizing to [0, 1]
        :param strength_reduction: Factor by which the deformation wave is reduced (wave / strength_reduction)
        :param probability: the probability per image to have this augmentation applied 
        """
        if strength_reduction == 0: raise ValueError("Strength reduction can only be larger or smaller than 0, not 0 itself.")
        self.strength_reduction = strength_reduction
        self.probability = probability


    def __call__(self, images: torch.Tensor) -> torch.Tensor :
        """
        :param images: a tensor of images. Can be a single image or a batch, must be grayscale. Must be in range [0, 1]
        :return: an elastically deformed tensor of the same shape as the input tensor. The output will be slightly more zoomed out compared to the input image due to intermediate padding.
        """

        # Ensure batch format - store original formatting for reformatting output
        wasBatch = True
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
            wasBatch = False

        # Ensure grayscale - reduce dimensions to [b, H, W]
        if images.shape[1] != 1:    raise ValueError("Input tensor is not grayscale: ", images.shape)
        else:                       images = images.squeeze(1)

        # To prevent major overflow due to shifting, add padding consistently to all images.
        pad_val = 10
        pad = tuple(repeat(pad_val, 4)) # (pad_val, pad_val, pad_val, pad_val)

        x_shape = images.shape[1] + (2 * pad_val)
        y_shape = images.shape[2] + (2 * pad_val)

        # Init output array, includes padding space
        deformed_images = torch.empty(size=(len(images), x_shape, y_shape))

        # Takes a batch of tensors: [b, H, W]
        for image_idx, image in enumerate(images):
            background_value = torch.round(torch.mean(image))
            if np.random.rand() > self.probability:
                deformed_images[image_idx] = torch.nn.functional.pad(image, pad, 'constant', value = background_value)
            else:
                # Pad the image so that the transformation does not overflow
                image = torch.nn.functional.pad(image, pad, 'constant', value = background_value)

                # Calculate for {n} waves: the frequency and amplitude such that the resulting waves do not combine into noise
                x_freqs, x_amps = self.get_wave_variables(n=5, W = image.shape[1], H = image.shape[0])

                # Generate the waves and combine them into {total_wave}
                total_wave = torch.zeros(size=(image.shape[1], ))
                for x_freq, x_amp in zip(x_freqs, x_amps):
                    # You do not want all the waves to start at x=0, y=0 (causes a huge bulge at the start), so shift all waves randomly
                    random_shift = torch.randint(low=0, high=image.shape[1], size=(1,)).item()
                    shifted_timesteps = range( random_shift, random_shift + image.shape[1])
                    # Compute 1 of {n} waves and add it to the total:
                    x_wave = self.get_wave(freqs = [x_freq], amps = [x_amp], timesteps = shifted_timesteps)
                    total_wave += x_wave

                # A wave may deform an image too strongly, this strength can be tuned with {strength_reduction}
                total_wave /= self.strength_reduction
                # plt.imshow(image, cmap='gray')
                # plt.plot(total_wave+(image.shape[0]/2)) 
                # plt.show()    

                # Now the wave is 'rolled' over the image, shifting the pixels at X=i with a value of {total_wave[i]}
                rolled_image = torch.empty(size=image.shape)
                for i in range(image.shape[1]):
                    rolled_image[:,i] = torch.roll(image[:,i], int(total_wave[i]), dims=0)
                
                # plt.imshow(rolled_image, cmap='gray')
                # plt.plot(total_wave+(image.shape[0]/2), alpha=0.3)
                # plt.show()

                # Save the rolled image
                deformed_images[image_idx] = rolled_image

                # Y DEFORM: not implemented for IAM
                # y_freqs, y_amps = self.get_wave_variables(5, strength, image.shape[0])
                # y_deformation_wave = self.get_wave(freqs=x_freqs, amps=x_amps, timesteps=range(image.shape[0]))
        
        # Restore original dimensions
        if wasBatch:
            return deformed_images.unsqueeze(1)
        else:
            return deformed_images

    # Generate the variables (freq, amp) for {n} waves
    def get_wave_variables(self, n: int, W: int, H: int) -> tuple[ list[torch.Tensor], list[torch.Tensor] ]:
        """
        For {n} waves, generate a frequency and amplitude such that the combination of these waves does not cause destruction of the input image
        :param n: number of waves to generate variables for
        :param W: width of the tensor to be later deformed
        :param H: height of the tensor to be later deformed
        :return: two arrays of size {n}, containing the frequencies (0) and the amplitudes (1)
        """
        freqs, amps = [], []

        # These tune the wave to 'fit' an image: a larger image requires a larger image to remain smoothely deformed
        width_tuner = 0.2/W
        height_tuner = 0.005*H #0.4*(H**2)
        for i in range(n):
            # Waves of different {freq}s are needed to create a complex {total_wave}
            freq = torch.normal(10, 2, size=(1,)).item() * width_tuner
            # This is ensured by the iteratively-adjusted {width-tuner}. 
            # It is scaled by a large value (60%) to ensure a large difference in frequencies, resulting in fewer 'huge bumps' in the wave.
            width_tuner *= 1.6

            # The {amp} is scaled by the inverse {freq}. A small frequency must never have a large ampltitude, as this causes noise-like deformation of the image.
            amp = height_tuner / torch.sqrt(torch.tensor(freq))

            freqs.append(freq)
            amps.append(amp)

        return freqs, amps
    
    def get_wave(self, freqs, amps, timesteps) -> torch.Tensor:
        """
        Creates a combined sine wave over the {timesteps}. Computes the waves for {freq[i]} and {amps[i]}, and sums their output at {timestep}.
        :param freqs: array of {n} frequencies
        :param amps: array of {n} amplitudes
        :param timesteps: array of timesteps
        :return: a combined wave tensor that can elastically deform an image
        """
        # Sanity check
        assert len(freqs) == len(amps)
        n_waves = len(freqs)

        wave = []
        for timestep in timesteps:
            deformation = 0
            for i in range(n_waves):
                deformation += amps[i] * np.sin(freqs[i]*np.pi*timestep)
            wave.append(deformation)
        return torch.FloatTensor(wave)
 
    
# Test script:
if __name__ == "__main__":
    image_path = r"C:\Users\thijs\University\Year5\Period4\HR\HandwritingRecognition\Assignment2\data\IAM\img\a01-000u-00.png"
    image_path = r"C:\Users\thijs\University\Year5\Period4\HR\HandwritingRecognition\Assignment2\data\IAM\img\a01-000x-03.png"
    image = cv2.imread(image_path)
    tensify = torchvision.transforms.ToTensor()
    deformify = RandomElasticDeformation(probability=0, strength_reduction=1)
    grayify = torchvision.transforms.Grayscale()

    image_batch = tensify(image).unsqueeze(0) # [1, 3, 154, 1584]
    gray_batch = grayify(image_batch) # [1, 1, 154, 1584]

    single_image = tensify(image)
    gray_image = grayify(single_image)

    deformed_batch = deformify(gray_batch)
    print(f"Done : {deformed_batch.shape}")
    plt.imshow(deformed_batch[0][0])
    plt.show()

    deformed_image = deformify(gray_image)
    print(f"Done : {deformed_image.shape}")
    plt.imshow(deformed_image[0])
    plt.show()