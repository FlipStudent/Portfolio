#Uses pillow (you can also use another imaging library if you want)
import torch
from .font_generator import *
import torchvision

class RandomGaussianNoise():
    def __init__(self, probability=1, mean=0.5, variance=0.5):
        self.probability = probability
        self.mean = mean
        self.variance = variance
    
    def __call__(self, images: torch.Tensor):
        noise_tensor = torch.randn(images.shape) * (self.variance ** 2)
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        for idx, image in enumerate(images):
            if np.random.rand() < self.probability:
                noise_tensor = torch.randn(image.shape) * (self.variance ** 2)
                image += noise_tensor
                image = torch.clip(image, min=0)
                image /= torch.max(image)
                images[idx] = image
        return images
    
if __name__ == "__main__":
    font_path = './data/font/habbakuk/Habbakuk.TTF'
    print(os.path.exists(font_path))
    generator = FontGenerator(font_path=font_path, letter_size=50, letter_padding=5)
    label, img = generator.generate_random_character()
    torch.tensor(img)
    process = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RandomGaussianNoise(probability=0.5, mean=0.5, variance=0.5)
    ])
    final_product = process(img)
    print(f"Done : {final_product.shape}")
    plt.imshow(final_product[0])
    plt.show()