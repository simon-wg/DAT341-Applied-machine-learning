# %%
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

logging.basicConfig(level=logging.INFO)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

logging.info(f"Using device: {device}")

# %% Data Pipelines
# 1. Baseline training transforms (No augmentation)
base_train_transforms = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 2. Augmented training transforms (Fulfills Data Augmentation requirement)
aug_train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 3. Validation transforms (Assignment specifies 'val' folder)
val_transforms = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Loaders
train_folder_base = ImageFolder("train", transform=base_train_transforms)
train_loader_base = DataLoader(train_folder_base, batch_size=32, shuffle=True)

train_folder_aug = ImageFolder("train", transform=aug_train_transforms)
train_loader_aug = DataLoader(train_folder_aug, batch_size=32, shuffle=True)

val_folder = ImageFolder("val", transform=val_transforms)
val_loader = DataLoader(val_folder, batch_size=32, shuffle=False)


# %% Helper Training/Evaluation Function
def train_and_evaluate(
    model, train_loader, val_loader, model_name, epochs=10, lr=0.001
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logging.info(f"--- Starting training for {model_name} ---")
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = 100 * correct / total
    logging.info(f"{model_name} Validation Accuracy: {accuracy:.2f}%")
    return accuracy


# %% Models (Old & New)


# SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 128 * 128, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


# BetterCNN
class BetterCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BetterCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Normalization CNN (Fulfills Normalization requirement)
class NormCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(NormCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Residual CNN (Fulfills Residual Connections requirement)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResidualCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ResidualCNN, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.res_block = ResidualBlock(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_block(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def create_vgg16_transfer_model(num_classes=2):
    weights_id = torchvision.models.VGG16_Weights.IMAGENET1K_V1
    model = torchvision.models.vgg16(weights=weights_id)

    # Freeze the convolutional base
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final classification layer
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model


# %% Execution Block (Run and Compare)
model_transfer = create_vgg16_transfer_model().to(device)
train_and_evaluate(
    model_transfer,
    train_loader_aug,
    val_loader,
    "Transfer VGG-16 (lr=0.0005)",
    lr=0.0005,
)

model_transfer = create_vgg16_transfer_model().to(device)
train_and_evaluate(
    model_transfer, train_loader_aug, val_loader, "Transfer VGG-16 (lr=0.001)", lr=0.001
)

model_transfer = create_vgg16_transfer_model().to(device)
train_and_evaluate(
    model_transfer, train_loader_aug, val_loader, "Transfer VGG-16 (lr=0.005)", lr=0.005
)

model_transfer = create_vgg16_transfer_model().to(device)
train_and_evaluate(
    model_transfer, train_loader_aug, val_loader, "Transfer VGG-16 (lr=0.01)", lr=0.01
)

model_transfer = create_vgg16_transfer_model().to(device)
train_and_evaluate(
    model_transfer,
    train_loader_aug,
    val_loader,
    "Transfer VGG-16 (epochs=8)",
    lr=0.0005,
    epochs=8,
)

model_transfer = create_vgg16_transfer_model().to(device)
train_and_evaluate(
    model_transfer,
    train_loader_aug,
    val_loader,
    "Transfer VGG-16 (epochs=12)",
    lr=0.0005,
    epochs=12,
)
