from src.dataset import Fruits
from src.logger import Logger
from src.model import DiffusionUNet
from src.train import train_diffusion

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


def train(file="utils/train_fruits.csv"):
    batch_size = 64
    crossval_folds = 5
    epochs = 40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    logger = Logger()
    dataset = Fruits(file=file)
    kf = KFold(n_splits=crossval_folds, shuffle=True, random_state=64)

    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        print("Fold: ", fold)
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        model = DiffusionUNet(in_size=100, t_range=1000, img_depth=3, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-6)

        train_diffusion(fold, model, optimizer, train_dataloader, val_dataloader, logger, device, epochs)


if __name__ == "__main__":
    train()
    # train(file="utils/dummy_fruits.csv")
