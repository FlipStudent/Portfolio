import torch
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection

# Define the path to the filtered dataset directory
data_dir = 'C:\Users\thijs\University\Year4\Thesis\Pract\Data'
# Define the path to the annotations file for the filtered dataset
annotations_dir = 'C:\Users\thijs\University\Year4\Thesis\Pract\Data\filtered_annotations'
train_root = data_dir + 'train2017/train2017'
val_root = data_dir + 'val2017/val2017'

# Define the image transformation pipeline
transform = transforms.Compose([
    # Convert to Py Tensor (img representation)
    transforms.ToTensor(),
    # Subtract mean & divide by SD - ensures that each color channel has similar characteristics 
    # (reduces impact of lighting to improve CNN's ability to learn useful features)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create an instance of the CocoDetection dataset class
normalized_data = CocoDetection(root=data_dir, annFile=annotations, transform=transform)

# Create a data loader for the filtered dataset
filtered_loader = torch.utils.data.DataLoader(normalized_data, batch_size=32, shuffle=True)