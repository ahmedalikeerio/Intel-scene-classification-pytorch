import torch
from sklearn.metrics import classification_report,confusion_matrix

from src.config import DEVICE, MODEL_PATH,NUM_CLASSES
from src.dataset import get_dataloaders
from src.model import SimpleCNN
from src.utils import plot_confusion_matrix, save_classification_report

def evaluate_model():
    print(f"Using Device: {DEVICE}")

    _, test_loader, class_names = get_dataloaders()

    model= SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
    model.eval()

    all_preds =[]
    all_labels=[]

    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)

            outputs=model(images)
            _, predicted =torch.max(outputs,1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    save_classification_report(
        all_labels,
        all_preds,
        class_names,
        model_name="improved_cnn"
    )

    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        model_name="improved_cnn"
    )

    print("\n Classification Report :\n")
    print(classification_report(all_labels,all_preds, target_names=class_names))

    print("\n Confusion Metrix: \n")
    print(confusion_matrix(all_labels,all_preds))

if __name__== '__main__':
    evaluate_model()