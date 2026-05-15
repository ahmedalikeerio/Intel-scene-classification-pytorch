import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from src.config import PLOTS_DIR, REPORTS_DIR


def plot_training_history(train_losses, train_accuracies, test_accuracies, model_name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, marker="o")
    plt.title(f"{model_name} - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(PLOTS_DIR / f"{model_name}_training_loss.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(test_accuracies, marker="o", label="Test Accuracy")
    plt.title(f"{model_name} - Train vs Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / f"{model_name}_accuracy.png")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{model_name}_confusion_matrix.png")
    plt.show()


def save_classification_report(y_true, y_pred, class_names, model_name):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    report_path = REPORTS_DIR / f"{model_name}_classification_report.txt"

    with open(report_path, "w") as file:
        file.write(report)

    print("\nClassification Report:\n")
    print(report)
    print(f"Report saved at: {report_path}")