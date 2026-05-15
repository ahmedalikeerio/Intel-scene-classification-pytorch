import torch
import torch.nn as nn
from torchvision import models

from src.config import (
    DEVICE,
    NUM_CLASSES,
    TRANSFER_MODEL_PATH,
    TRANSFER_IMAGE_SIZE
)
from src.dataset import get_dataloaders
from src.utils import plot_confusion_matrix, save_classification_report


def build_resnet18_model():
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    return model


def evaluate_transfer_model():
    print(f"Using Device: {DEVICE}")

    _, test_loader, class_names = get_dataloaders(
        image_size=TRANSFER_IMAGE_SIZE
    )

    model = build_resnet18_model().to(DEVICE)
    model.load_state_dict(torch.load(TRANSFER_MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    save_classification_report(
        all_labels,
        all_preds,
        class_names,
        model_name="resnet18_transfer"
    )

    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        model_name="resnet18_transfer"
    )


if __name__ == "__main__":
    evaluate_transfer_model()