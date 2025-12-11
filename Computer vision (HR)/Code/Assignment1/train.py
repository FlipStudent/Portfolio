import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from classification.model import ConvNet
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from util.random_gaussian_noise import RandomGaussianNoise
from util.random_elastic_deformation import RandomElasticDeformation
from classification.pretrain_dataset import CustomDataset
from util.font_generator import FontGenerator
import sys
import json

NUM_EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-2
MOMENTUM = 0.9
K_FOLDS = 3
NUM_CLASSES = 27
PLT_DIR = "classification/results"
MDL_DIR = "models"
SAVING = True


class Invert:
    def __call__(self, image):
        return 1.0 - image


preprocessing_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ImageFolder defaults loads RGB
    transforms.ToTensor(),  # Convert images to tensors
    Invert(),
])

data_augmentation_transforms = transforms.Compose([
    RandomElasticDeformation(probability=0.8),
    RandomGaussianNoise(probability=0.5, mean=0.5, variance=0.5),
    transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)),
    # transforms.RandomRotation(10),
])


def evaluate_model(conv_net, validation_loader, device, criterion):
    conv_net.eval()
    validation_loss = 0
    correct = 0
    #  For the confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    conf_pred = []
    conf_true = []
    with torch.no_grad():
        for data, target in validation_loader:
            inputs = data.to(device)
            labels = target.to(device)
            output = conv_net(inputs)
            validation_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            conf_pred.extend(pred.view_as(labels).tolist())  # Confusion matrix
            conf_true.extend(labels.tolist())

            for t, p in zip(labels.view(-1), pred.view(-1)):
                cm[t.long(), p.long()] += 1

    validation_loss /= len(validation_loader)
    accuracy = 100.0 * sum(x == y for x, y in zip(conf_pred, conf_true)) / len(conf_true)

    return validation_loss, accuracy, cm


def train(conv_net, device, train_loader, data_augmentation_transforms, optimizer, criterion, validation_loader=None):
    train_losses = []
    validation_losses = []
    accuracies = []
    for epoch in range(NUM_EPOCHS):
        conv_net.train()
        running_loss = 0.0
        for data, target in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs = torch.stack([data_augmentation_transforms(img) for img in data]).to(device)
            labels = target.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = conv_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        if validation_loader:
            validation_loss, accuracy, _ = evaluate_model(conv_net, validation_loader, device, criterion)
            validation_losses.append(validation_loss)
            accuracies.append(accuracy)
            print(
                f'[{epoch + 1}] train loss: {avg_train_loss:.2f} | val loss: {validation_loss} | accuracy: {accuracy}', flush=True)
        else:
            print(f'[{epoch + 1}] train loss: {avg_train_loss:.2f}', flush=True)

    return train_losses, validation_losses, accuracies


def plot_results(data_lists, ylabel, fn, legend=True, multiple_runs=True, step_size=1):
    if not SAVING:
        return
    for data, label in data_lists:
        np.save(PLT_DIR + "/" + fn + label.replace(" ", "") + ".npy", data)
        if multiple_runs:
            means = np.mean(data, axis=0)
            std_devs = np.std(data, axis=0)
            plt.plot(means, label=label)
            plt.fill_between(range(NUM_EPOCHS), means - std_devs, means + std_devs, alpha=0.15)
        else:
            plt.plot(range(0, step_size*len(data), step_size), data, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    plt.savefig(PLT_DIR + "/" + fn)
    plt.clf()


def get_weighted_sampling_dataloader(dataset: Dataset):
    class_counts = [0] * NUM_CLASSES
    for _, label in dataset:
        class_counts[label] += 1

    class_weights = 1.0 / torch.Tensor(class_counts)
    weights = [class_weights[label] for _, label in dataset]
    oversample_sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=oversample_sampler
    )
    return train_loader


def k_folds_training(k_folds, device, criterion, model_path=None):
    combined_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    accuracy_scores = []

    dataset = datasets.ImageFolder('./data/monkbrill-reshaped', transform=preprocessing_transforms)

    with open("data/labels_char_names.json",
              "w") as file:  #write lables to JSON, needed to retrieve characters
        json.dump(dataset.class_to_idx, file)

    kf = KFold(n_splits=k_folds, shuffle=True)
    train_losses = []
    validation_losses = []
    accuracies = []
    for fold, (train_indices, validation_indices) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}", flush=True)
        print("-------")

        train_dataset = Subset(dataset, train_indices)
        train_loader = get_weighted_sampling_dataloader(train_dataset)

        validation_loader = DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(validation_indices),
        )

        conv_net = ConvNet(NUM_CLASSES).to(device)
        if model_path:
            print(f"Loading pretrained model: {model_path}")
            conv_net.load_state_dict(torch.load(model_path))
        optimizer = optim.SGD(conv_net.parameters(), lr=LR, momentum=MOMENTUM)
        train_loss, validation_loss, accuracy = train(conv_net, device, train_loader, data_augmentation_transforms,
                                                      optimizer, criterion, validation_loader)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        accuracies.append(accuracy)

        if SAVING:
            save_model_path = f"{MDL_DIR}/k-fold/conv-{fold}.pt"
            torch.save(conv_net.state_dict(), save_model_path)

        _, accuracy, conf_matrix = evaluate_model(conv_net, validation_loader, device, criterion)
        accuracy_scores.append(accuracy)
        combined_confusion_matrix += conf_matrix

        plot_results([(train_losses, "Train loss"), (validation_losses, "Validation loss")], "Loss", "losses")
        plot_results([(accuracies, "Accuracy")], "Accuracy", "accuracies", legend=False)

        print(f"Accuracy: {np.mean(accuracy_scores):.2f} ({np.std(accuracy_scores):.2f})")
    print('Finished Training')

    combined_confusion_matrix = np.round(
        combined_confusion_matrix / combined_confusion_matrix.sum(axis=1, keepdims=True), 2)
    ConfusionMatrixDisplay(combined_confusion_matrix).plot()
    plt.show()


def full_train(device, criterion, model_path=None):
    dataset = datasets.ImageFolder('./data/monkbrill-reshaped', transform=preprocessing_transforms)
    train_loader = get_weighted_sampling_dataloader(dataset)
    conv_net = ConvNet(NUM_CLASSES).to(device)
    if model_path:
        print(f"Loading pretrained model: {model_path}")
        conv_net.load_state_dict(torch.load(model_path))
    optimizer = optim.SGD(conv_net.parameters(), lr=LR, momentum=MOMENTUM)
    train(conv_net, device, train_loader, data_augmentation_transforms, optimizer, criterion)

    if SAVING:
        if model_path:
            save_model_path = f"{MDL_DIR}/full-with-pretraining.pt"
        else:
            save_model_path = f"{MDL_DIR}/full.pt"
        torch.save(conv_net.state_dict(), save_model_path)


def pre_train(device, criterion):
    font_path = r'data/font/habbakuk/Habbakuk.TTF'
    validation_set_dir = r'data/monkbrill-reshaped'

    letter_size = 28
    letter_padding = 5
    font_generator = FontGenerator(font_path=font_path,
                                   letter_size=letter_size,
                                   letter_padding=letter_padding)
    
    train_dataset = CustomDataset(font_generator=font_generator,
                                  augment_transforms=data_augmentation_transforms)
    
    validation_dataset = datasets.ImageFolder(validation_set_dir,
                                              transform=preprocessing_transforms)
    
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE
    )

    conv_net = ConvNet(NUM_CLASSES).to(device)
    optimizer = optim.SGD(conv_net.parameters(), lr=LR, momentum=MOMENTUM)
    num_batches = 5_000
    measure_freq = 100
    running_loss = 0.0
    train_losses = []
    validation_losses = []
    accuracies = []
    for idx in range(num_batches):
        if idx != 0 and idx % measure_freq == 0:
            validation_loss, accuracy, cm = evaluate_model(conv_net, validation_loader, device, criterion)
            validation_losses.append(validation_loss)
            running_loss /= measure_freq
            train_losses.append(running_loss)
            accuracies.append(accuracy)
            print(f"Batch: {idx} | train loss: {running_loss:.2f} | validation loss: {validation_loss:.2f} | accuracy: {accuracy:.2f}")
            running_loss = 0.0
            plot_results([(train_losses, "Train loss"), (validation_losses, "Validation loss")], ylabel="Loss",
                         fn="pretrain-losses", multiple_runs=False, step_size=measure_freq)
            plot_results([(accuracies, "Accuracy")], ylabel="Accuracy", fn="pretrain-accuracies", legend=False,
                         multiple_runs=False, step_size=measure_freq)

        conv_net.train()
        data, labels = train_dataset.get_batch(batch_size=BATCH_SIZE)
        data = data.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = conv_net(data)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    if SAVING:
        model_path = f"{MDL_DIR}/pretrained.pt"
        torch.save(conv_net.state_dict(), model_path)


if __name__ == '__main__':
    model_path = None
    if len(sys.argv) < 2:
        print("Usage: main.py <kfold|full|pretrain> [model_path]")
        exit()
    if len(sys.argv) == 3:
        model_path = sys.argv[2]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    criterion = nn.CrossEntropyLoss()
    mode = sys.argv[1]
    if mode == "full":
        print("Training on full dataset...")
        full_train(device, criterion, model_path)
    elif mode == "kfold":
        print("Training with k-folds...")
        k_folds_training(K_FOLDS, device, criterion, model_path)
    elif mode == "pretrain":
        print("Pretraining with synthetic dataset...")
        pre_train(device, criterion)
    else:
        print("Mode must be kfold, full or pretrain")
