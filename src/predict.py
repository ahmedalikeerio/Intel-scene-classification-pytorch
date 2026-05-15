import argparse
import torch
from PIL import Image
from torchvision import transforms

from src.config import DEVICE, MODEL_PATH, IMAGE_SIZE, CLASS_NAMES, NUM_CLASSES
from src.model import SimpleCNN


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    return image


def predict(image_path):
    print(f"Using device: {DEVICE}")

    model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    image = load_image(image_path).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = CLASS_NAMES[predicted_class.item()]
    confidence_score = confidence.item() * 100

    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence: {confidence_score:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    predict(args.image)