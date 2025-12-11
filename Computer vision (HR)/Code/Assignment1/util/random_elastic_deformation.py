#Uses pillow (you can also use another imaging library if you want)
import matplotlib.pyplot as plt
import torch
import math
from .font_generator import *
import torchvision


class RandomElasticDeformation():
    def __init__(self, set_strength=None, probability=1):
        self.set_strength = set_strength
        self.probability = probability
    
    def __call__(self, images: torch.Tensor):
        deformed_images = torch.empty(size=images.shape)
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        for image_idx, image in enumerate(images):
            if np.random.rand() > self.probability:
                deformed_images[image_idx] = image
            else:
                # print("Applied elastic deformation")
                image = np.array(image)
                if self.set_strength:
                    strength = self.set_strength
                else:
                    strength = torch.randint(1, 5, (1,)).item()
                original_letter_shape = self.crop_to_letter(image).shape
                original_image_shape = image.shape

                # Pad the image so that the transformation does not overflow
                image = np.pad(image, (30,), 'constant', constant_values=(0,))

                # Create sinus waves for both axes                
                x_freqs, x_amps = self.get_wave_variables(5, strength, image.shape[0])
                y_freqs, y_amps = self.get_wave_variables(5, strength, image.shape[0])

                # Roll the sinus waves over the image
                for i in range(image.shape[0]):
                    deform_x = self.get_deformation(x_freqs, x_amps, i)
                    image[i] = np.roll(image[i], int(deform_x))

                for i in range(image.shape[0]):
                    deform_y = self.get_deformation(y_freqs, y_amps, i)
                    image[:,i] = np.roll(image[:,i], int(deform_y))

                # Image is now deformed, so crop to the new letter
                deformed_cropped_letter = self.crop_to_letter(image)

                # print(f"Cropped the image, shape: {deformed_cropped_letter.shape}")

                # Resize this letter to the old letter shape (with some random scaling factor)
                x_rand_scale = int(original_letter_shape[0]*np.random.normal(loc=1, scale=0.2))
                y_rand_scale = int(original_letter_shape[1]*np.random.normal(loc=1, scale=0.2))

                # Ensure it does not overflow
                x_rand_scale = min(x_rand_scale, original_image_shape[0])
                y_rand_scale = min(y_rand_scale, original_image_shape[1])

                reshaped_deformed_letter = self.scale(deformed_cropped_letter, x_rand_scale, y_rand_scale)

                # Pad it back up to the original
                pad_x_left = math.ceil( (original_image_shape[0] - reshaped_deformed_letter.shape[0] )/2)
                pad_x_right = math.floor( (original_image_shape[0] - reshaped_deformed_letter.shape[0] )/2)
                pad_y_top = math.ceil( (original_image_shape[1] - reshaped_deformed_letter.shape[1] )/2 )
                pad_y_bot = math.floor( (original_image_shape[1] - reshaped_deformed_letter.shape[1] )/2 )
                deformed_letter = np.pad(reshaped_deformed_letter, pad_width =((pad_x_left,pad_x_right),(pad_y_top,pad_y_bot)), mode='constant', constant_values=0)
                deformed_images[image_idx] = torch.tensor(deformed_letter)
        return deformed_images
    
    def cut_line(self, image, linewidth=2):
        idx = np.random.randint(low=15, high=36)
        if np.random.randint(0, 2) == 1:
            image[idx:idx+linewidth] *= 0
        else:
            image[:, idx:idx+linewidth] *= 0
        return image

    # Generate the variables (freq, amp) for N waves
    def get_wave_variables(self, n, strength, image_size):
        freqs, amps = [], []
        for i in range(n):
            freq = np.random.rand()*0.02
            w = freq / image_size
            amp = strength / (freq+0.25)
            freqs.append(freq)
            amps.append(amp)
        return freqs, amps

    # Convert N wave variables into 1 complex wave
    def get_deformation(self, freqs, amps, t):
        out_vals = []
        for i, freq in enumerate(freqs):
            out = amps[i] * np.sin(freq*np.pi*t)
            out_vals.append(out)
        return np.mean(np.array(out_vals))
    
    # Apply complex wave to an image
    def roll(self, image, direction, less_roll_factor=None):
        if not less_roll_factor:
            less_roll_factor = np.random.randint(15, 35) 
        A = (image.shape[0] / less_roll_factor)
        w = 2.0 / image.shape[1]

        shift = lambda x: A * np.sin(2.0*np.pi*x * w)

        if direction == "vertical":
            for i in range(image.shape[0]):
                image[:,i] = np.roll(image[:,i], int(shift(i)))
        elif direction == "horizontal":
            for i in range(image.shape[0]):
                image[i] = np.roll(image[i], int(shift(i)))
        else:
            raise ValueError
        
        return image
    
    # Finds the size of the actual letter (not the letter image)
    def find_outer_edge(self, means: torch.Tensor):
        start, end = 0, len(means)
        for i, mean in enumerate(means):
            if mean > 0:
                start = i
                break
        # for j, mean in enumerate(torch.flip(means, dims=(0,))): # Torch variant
        for j, mean in enumerate(np.flip(means)):
            i = len(means)-j
            if mean > 0:
                end = i
                break
        # Fail case, return 0, max
        return start, end
    
    # Returns the bbox of the actual letter
    def get_bbox(self, image):
        ymin, ymax = self.find_outer_edge(torch.mean(image,axis=1))
        xmin, xmax = self.find_outer_edge(torch.mean(image,axis=0))
        w = xmax-xmin - 1
        h = ymax-ymin - 1
        bbox = xmin, ymin, w, h
        return bbox
    
    # Credits to Roman Kogan, https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
    def scale(self, im, nR, nC):
        nR0 = len(im)     # source number of rows 
        nC0 = len(im[0])  # source number of columns 
        return np.array([[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]  
                    for c in range(nC)] for r in range(nR)])

    def crop_to_letter(self, img:torch.Tensor):
        if torch.is_tensor(img):
            s_V, e_V = self.find_outer_edge(torch.mean(img,axis=1)) # Was np.mean(img, axis = 1)
            s_H, e_H = self.find_outer_edge(torch.mean(img,axis=0)) # Was np.mean(img, axis = 2)
        else:
            s_V, e_V = self.find_outer_edge(np.mean(img,axis=1)) # Was np.mean(img, axis = 1)
            s_H, e_H = self.find_outer_edge(np.mean(img,axis=0)) # Was np.mean(img, axis = 2)

        if (s_H + e_H) > 0:
            img = img[:, s_H:e_H]
        if (s_V + e_V) > 0:
            img = img[s_V:e_V]
        return img
    
if __name__ == "__main__":
    font_path = './data/font/habbakuk/Habbakuk.TTF'
    font_path = './data/font/habbakuk/Habbakuk.TTF'
    print(os.path.exists(font_path))
    generator = FontGenerator(font_path=font_path, letter_size=50, letter_padding=5)
    label, img = generator.generate_random_character()
    torch.tensor(img)
    process = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RandomElasticDeformation()
    ])
    final_product = process(img)
    print(f"Done : {final_product.shape}")
    plt.imshow(final_product[0])