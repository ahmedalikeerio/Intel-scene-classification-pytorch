import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models

from src.config import (
    DEVICE,
    TRANSFER_EPOCHS,
    TRANSFER_LEARNING_RATE,
    NUM_CLASSES,
    TRANSFER_MODEL_PATH,
    TRANSFER_IMAGE_SIZE
)
from src.dataset import get_dataloaders
from src.utils import plot_training_history


def build_resnet18_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last ResNet block for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final classifier layer
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    return model


def evaluate_model(model, data_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def train_transfer_model():
    print(f"Using Device: {DEVICE}")

    train_loader, test_loader, class_names = get_dataloaders(
        image_size=TRANSFER_IMAGE_SIZE
    )

    print("Classes:", class_names)
    print("Training Batches:", len(train_loader))
    print("Testing Batches:", len(test_loader))

    model = build_resnet18_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=TRANSFER_LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    best_test_accuracy = 0.0

    for epoch in range(TRANSFER_EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{TRANSFER_EPOCHS}"
        )

        for images, labels in progress_bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                "loss": loss.item()
            })

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        test_accuracy = evaluate_model(model, test_loader)

        scheduler.step(test_accuracy)

        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch [{epoch + 1}/{TRANSFER_EPOCHS}] "
            f"Loss: {epoch_loss:.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Test Acc: {test_accuracy:.2f}%"
        )

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), TRANSFER_MODEL_PATH)
            print(f"Best transfer model saved: {TRANSFER_MODEL_PATH}")

    print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")

    plot_training_history(
        train_losses,
        train_accuracies,
        test_accuracies,
        model_name="resnet18_transfer"
    )


if __name__ == "__main__":
    train_transfer_model()