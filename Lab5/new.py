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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info(f"Using device: {device}")

# %% Data Pipelines
# Baseline transforms
base_train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Augmented training transforms
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

# Validation transforms
val_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Loaders
train_folder_base = ImageFolder("train", transform=base_train_transforms)
train_loader_base = DataLoader(train_folder_base, batch_size=32, shuffle=True)

train_folder_aug = ImageFolder("train", transform=aug_train_transforms)
train_loader_aug = DataLoader(train_folder_aug, batch_size=32, shuffle=True)

test_folder = ImageFolder("test", transform=val_transforms)
test_loader = DataLoader(test_folder, batch_size=32, shuffle=False)

val_folder = ImageFolder("val", transform=val_transforms)
val_loader = DataLoader(val_folder, batch_size=32, shuffle=False)


# %% Helper Training/Evaluation Function
def train_and_evaluate(
    model, train_loader, val_loader, model_name, epochs=10, lr=0.001, print_steps=False
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
        if print_steps:
            accuracy = 100 * correct / total
            logging.info(
                f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {accuracy:.2f}%"
            )

    if not print_steps:
        logging.info(f"--- Finished training for {model_name} ---")
        accuracy = 100 * correct / total
        logging.info(f"{model_name} Validation Accuracy: {accuracy:.2f}%")

    return accuracy


# %% Models


# SimpleCNN, Conv -> ReLU -> FC
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


# BetterCNN, Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC
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


# Normalization CNN, BetterCNN + BatchNorm
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


# Residual CNN, BetterCNN + Residual Block
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


# Transfer Learning with VGG-16
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


# %% [python] SimpleCNN
model_simple = SimpleCNN().to(device)
train_and_evaluate(
    model_simple, train_loader_base, test_loader, "SimpleCNN (Base)", print_steps=True
)

model_simple = SimpleCNN().to(device)
train_and_evaluate(
    model_simple,
    train_loader_aug,
    test_loader,
    "SimpleCNN (Augmented)",
    print_steps=True,
)

# %% [python] BetterCNN
model_better = BetterCNN().to(device)
train_and_evaluate(
    model_better, train_loader_base, test_loader, "BetterCNN (Base)", print_steps=True
)

model_better = BetterCNN().to(device)
train_and_evaluate(
    model_better,
    train_loader_aug,
    test_loader,
    "BetterCNN (Augmented)",
    print_steps=True,
)

# %% [python] NormCNN
model_norm = NormCNN().to(device)
train_and_evaluate(
    model_norm, train_loader_base, test_loader, "NormCNN (Base Data)", print_steps=True
)

model_norm_aug = NormCNN().to(device)
train_and_evaluate(
    model_norm_aug,
    train_loader_aug,
    test_loader,
    "NormCNN (Augmented Data)",
    print_steps=True,
)

# %% [python] ResidualCNN
model_residual = ResidualCNN().to(device)
train_and_evaluate(
    model_residual,
    train_loader_aug,
    test_loader,
    "ResidualCNN (Augmented Data)",
    print_steps=True,
)

# %% [python] Transfer Learning with VGG-16
model_transfer = create_vgg16_transfer_model().to(device)
train_and_evaluate(
    model_transfer,
    train_loader_aug,
    test_loader,
    "Transfer VGG-16",
    lr=0.0005,
    print_steps=True,
)


# %% [python] Final Evaluation on Validation Set
model_final_transfer = create_vgg16_transfer_model().to(device)
train_and_evaluate(
    model_final_transfer,
    train_loader_aug,
    val_loader,
    "Final Transfer VGG-16 (Validation Set)",
    epochs=8,
    lr=0.0005,
    print_steps=False,
)
