import torch
from torch.utils.data import Dataset
import numpy as np
import PIL
import torchvision


class CustomDataset(Dataset):
    def __init__(self, font_generator, augment_transforms):

        self.generator = font_generator
        self.augment_transforms = augment_transforms

        self.letter_size = font_generator.out_size

        self.image_shape = (self.letter_size, self.letter_size)
        self.tensify = torchvision.transforms.ToTensor()

    def __len__(self):
        return 999999

    def get_batch(self, batch_size):
        batch = torch.empty(size=(batch_size,1, self.letter_size, self.letter_size))
        labels = torch.empty(size=(batch_size,), dtype=torch.uint8)

        for i in range(batch_size):
            label, np_image = self.generator.generate_random_character(shape=self.image_shape) # int, 2D np array
            label = self.generator.char_to_label[label]
            image = self.tensify(np_image)
            image = self.augment_transforms(image)
            batch[i] = image
            labels[i] = label

        return batch, labels


if __name__ == "__main__":
    pass
