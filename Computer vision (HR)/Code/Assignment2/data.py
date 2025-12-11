import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import random
import torch
from torchvision import transforms
import json
from augmentations.random_blots import RandomAddBlots
from augmentations.random_gaussian_noise import RandomGaussianNoise
from augmentations.random_elastic_deformation import RandomElasticDeformation

device = "cuda" if torch.cuda.is_available() else "cpu"


train_transforms = transforms.Compose([
    # transforms.Grayscale(),
    # transforms.ToTensor(),
    # RandomElasticDeformation(probability=0.5, strength_reduction=2),
    # RandomAddBlots(probability=0.5, blot_smallness=3, elipse_kernel_shape=(9, 9)),
    # RandomGaussianNoise(probability=0.9, mean=0.5, variance=0.5, invert=True),
    # transforms.ToPILImage()
])


class ImageDataset(Dataset):
    """
    Image dataset used for loading datasets with or without labels in the form of d dictionary where key=filename and
    value=label.
    Created with help from the following notebook:
    https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb
    """

    def __init__(self, root_dir, label_dict: dict | None, processor, transform=None, max_target_length=128):
        self.root_dir = root_dir
        self.transforms = transform
        self.labels = label_dict
        self.processor = processor
        self.max_target_length = max_target_length
        self.filenames = os.listdir(self.root_dir)
        if device == "cpu":
            self.filenames = self.filenames[:int(0.001*len(self.filenames))]

    def __len__(self):
        if self.labels:
            return len(self.labels)
        else:
            return len(self.filenames)

    def __getitem__(self, idx):
        if self.labels:
            filenames = list(self.labels.keys())
        else:
            filenames = self.filenames
        filename = filenames[idx]

        # Load image
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")

        old_image = image.copy()
        if self.transforms:
            image = self.transforms(image)

        image = image.convert(mode="RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        encoding = {"pixel_values": pixel_values.squeeze(), "filenames": filename}
        if self.labels:
            label = self.labels[filename]
            labels = self.processor.tokenizer(label,
                                              padding="max_length",
                                              max_length=self.max_target_length).input_ids
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
            encoding["labels"] = torch.tensor(labels)
            encoding["text_labels"] = label
        return encoding


def get_iam_dataset(processor, batch_size):
    """
    Creates train and validation loaders for the IAM dataset.
    :param processor: processor used for processing images for the TrOCR model.
    :param batch_size: batch size
    :return: train loader, validation loader
    """
    label_fn = "data/IAM/iam_lines_gt.txt"
    img_dir = "data/IAM/img"

   # Read label file
    with open(label_fn, 'r') as f:
        lines = f.readlines()

    labels = []
    for i in range(0, len(lines), 3):
        filename = lines[i].strip()
        label = lines[i + 1].strip()
        labels.append((filename, label))

    random.seed(32)
    if device == "cpu":
        labels = labels[:int(0.01*len(labels))] # for testing code
    split_point = int(0.9 * len(labels))
    random.shuffle(labels)
    train_labels = labels[:split_point]
    train_dict = {}
    for (key, value) in train_labels:
        train_dict[key] = value
    val_labels = labels[split_point:]
    val_dict = {}
    for (key, value) in val_labels:
        val_dict[key] = value

    train_dataset = ImageDataset(img_dir, train_dict, processor, transform=train_transforms)
    val_dataset = ImageDataset(img_dir, val_dict, processor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader


def get_lam_dataset(processor, batch_size):
    """
    Creates train and validation loaders for the LAM dataset.
    :param processor: processor used for processing images for the TrOCR model.
    :param batch_size: batch size
    :return: train loader, validation loader
    """
    img_dir = "data/LAM/lines/img"
    train_fn = "data/LAM/lines/split/basic/train.json"
    val_fn = "data/LAM/lines/split/basic/val.json"

    with open(train_fn, 'r') as file:
        train_json = json.load(file)
    with open(val_fn, 'r') as file:
        val_json = json.load(file)

    if device == "cpu":
        train_json = train_json[:4]
        val_json = val_json[:4]
    train_dict = {item["img"]: item["text"] for item in train_json}
    val_dict = {item["img"]: item["text"] for item in val_json}

    train_dataset = ImageDataset(img_dir, train_dict, processor, transform=train_transforms)
    val_dataset = ImageDataset(img_dir, val_dict, processor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader


def get_selfmade_dataset(processor, batch_size):
    """
    Creates train and validation loaders for the 'selfmade' dataset.
    :param processor: processor used for processing images for the TrOCR model.
    :param batch_size: batch size
    :return: train loader, validation loader
    """
    img_dir = "data/selfmade/self_lines"
    label_fn = "data/selfmade/lines.txt"

    with open(label_fn, 'r') as file:
        input_text = file.read()

    # Create an empty dictionary
    file_label_dict = {}

    # Process each line
    for line in input_text.strip().split('\n'):
        parts = line.split(maxsplit=9)
        filename = parts[0] + '.png'
        label = parts[9].replace('|', ' ')
        file_label_dict[filename] = label

    keys = list(file_label_dict.keys())
    random.shuffle(keys)
    if device == "cpu":
        keys = keys[:12]
    split_index = int(len(keys) * 0.9)
    train_keys = keys[:split_index]
    validation_keys = keys[split_index:]

    # Create the training and validation dictionaries
    train_dict = {key: file_label_dict[key] for key in train_keys}
    val_dict = {key: file_label_dict[key] for key in validation_keys}

    train_dataset = ImageDataset(img_dir, train_dict, processor, transform=train_transforms)
    val_dataset = ImageDataset(img_dir, val_dict, processor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader
