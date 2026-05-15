import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, request
from torchvision import models, transforms


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "models" / "saved" / "resnet18_transfer_mps.pth"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

NUM_CLASSES = 6
IMAGE_SIZE = 224


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


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


def load_model():
    model = build_resnet18_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


def transform_image(image_path):
    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = image_transform(image)
    image = image.unsqueeze(0)

    return image.to(DEVICE)


def predict_image(image_path):
    image_tensor = transform_image(image_path)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = CLASS_NAMES[predicted_class.item()]
    confidence_score = confidence.item() * 100

    all_probs = probabilities.cpu().numpy()[0]

    class_probabilities = []
    for class_name, prob in zip(CLASS_NAMES, all_probs):
        class_probabilities.append({
            "class_name": class_name,
            "probability": round(float(prob) * 100, 2)
        })

    class_probabilities = sorted(
        class_probabilities,
        key=lambda x: x["probability"],
        reverse=True
    )

    return predicted_label, round(confidence_score, 2), class_probabilities


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None
    class_probabilities = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No image file uploaded."
            return render_template("index.html", error=error)

        file = request.files["image"]

        if file.filename == "":
            error = "Please select an image first."
            return render_template("index.html", error=error)

        allowed_extensions = {"png", "jpg", "jpeg", "webp"}

        file_extension = file.filename.rsplit(".", 1)[-1].lower()

        if file_extension not in allowed_extensions:
            error = "Only PNG, JPG, JPEG, and WEBP images are allowed."
            return render_template("index.html", error=error)

        filename = file.filename
        save_path = UPLOAD_FOLDER / filename
        file.save(save_path)

        prediction, confidence, class_probabilities = predict_image(save_path)

        image_url = f"/static/uploads/{filename}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        class_probabilities=class_probabilities,
        error=error,
        device=str(DEVICE)
    )


if __name__ == "__main__":
    app.run(debug=True)