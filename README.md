# Intel Scene Classification with PyTorch, ResNet18, Apple MPS, and Flask UI

A complete deep learning image classification project built with **PyTorch**.  
This project classifies natural scene images into six categories using three progressively improved models:

- Simple Custom CNN
- Improved Custom CNN
- ResNet18 Transfer Learning

The final deployed model uses **ResNet18 Transfer Learning** and achieved a best test accuracy of approximately **93.83%** on the Intel Image Classification dataset.

The project also includes a clean and interactive **Flask web application** where users can upload an image and receive a prediction with confidence scores and class probabilities.

---

## Project Overview

This project was created as a practical PyTorch deep learning project for learning and portfolio development. It covers the full workflow of an image classification system:

1. Dataset loading using `ImageFolder`
2. Image preprocessing and augmentation
3. PyTorch `Dataset` and `DataLoader`
4. Custom CNN model training
5. Improved CNN model training
6. Transfer learning using pretrained ResNet18
7. Model evaluation with classification report and confusion matrix
8. Prediction using a saved `.pth` model
9. Flask-based web user interface for image upload and prediction
10. Apple Silicon GPU acceleration using MPS

---

## Dataset

The project uses the **Intel Image Classification Dataset**.

Dataset classes:

```text
buildings
forest
glacier
mountain
sea
street
```

Dataset structure after download and extraction:

```text
data/raw/
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
│
├── seg_test/
│   └── seg_test/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
│
└── seg_pred/
    └── seg_pred/
```

Dataset source:

```text
https://www.kaggle.com/datasets/puneet6060/intel-image-classification
```

---

## Final Model Performance

### Model Comparison

| Model | Type | Test Accuracy |
|---|---|---:|
| Simple CNN | Custom model from scratch | 83.03% |
| Improved CNN | Deeper custom CNN from scratch | 86.70% |
| ResNet18 Transfer Learning | Pretrained transfer learning | 93.83% |

The best model is:

```text
ResNet18 Transfer Learning
```

Saved model path:

```text
models/saved/resnet18_transfer_mps.pth
```

---

## ResNet18 Classification Report

Final ResNet18 performance:

```text
              precision    recall  f1-score   support

   buildings       0.93      0.92      0.93       437
      forest       1.00      1.00      1.00       474
     glacier       0.93      0.88      0.90       553
    mountain       0.90      0.90      0.90       525
         sea       0.94      0.99      0.97       510
      street       0.93      0.95      0.94       501

    accuracy                           0.94      3000
   macro avg       0.94      0.94      0.94      3000
weighted avg       0.94      0.94      0.94      3000
```

Best test accuracy:

```text
93.83%
```

---

## Technology Stack

| Category | Tools / Libraries |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Computer Vision | Torchvision |
| Model Architecture | CNN, ResNet18 |
| Transfer Learning | ImageNet pretrained ResNet18 |
| Data Processing | PIL, Torchvision Transforms |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib |
| Web Framework | Flask |
| Frontend | HTML, CSS, Jinja Templates |
| Local Acceleration | Apple MPS GPU |
| Development Environment | VS Code, macOS Terminal |

---

## Features

### Machine Learning Features

- Image classification into six scene classes
- Custom CNN training from scratch
- Improved CNN with deeper convolution blocks
- Transfer learning using ResNet18
- Apple MPS support for MacBook GPU acceleration
- Training and testing accuracy tracking
- Classification report generation
- Confusion matrix generation
- Saved `.pth` model loading

### Web Application Features

- Beautiful interactive web UI
- Image upload support
- Predict button
- Reset button
- Uploaded image preview
- Predicted class display
- Confidence score display
- Class probability bars
- Flask backend integration
- Model inference using best ResNet18 model

---

## Project Structure

```text
Intel-scene-classification-project/
│
├── app.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── improved_model.py
│   ├── train.py
│   ├── train_improved.py
│   ├── train_transfer.py
│   ├── evaluate.py
│   ├── evaluate_improved.py
│   ├── evaluate_transfer.py
│   ├── predict.py
│   └── utils.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/
│       └── .gitkeep
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── saved/
│       └── resnet18_transfer_mps.pth
│
├── outputs/
│   ├── plots/
│   └── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Apple MPS Support

This project supports Apple Silicon GPU acceleration using PyTorch MPS.

The device is selected automatically:

```python
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
```

To test MPS support:

```bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

Expected output:

```text
True
```

If MPS is available, the project will use:

```text
mps
```

Otherwise, it will fall back to:

```text
cpu
```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/intel-scene-classification-pytorch.git
cd intel-scene-classification-pytorch
```

Replace `YOUR_USERNAME` with your GitHub username.

---

### 2. Create Virtual Environment

For macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not ready yet, install manually:

```bash
pip install torch torchvision torchaudio numpy pandas matplotlib scikit-learn pillow tqdm kaggle flask
```

---

## Kaggle Dataset Setup

### 1. Download Kaggle API Token

Go to Kaggle:

```text
Kaggle Account Settings → API → Create New Token
```

This downloads:

```text
kaggle.json
```

---

### 2. Move Kaggle Token to Correct Location

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Test Kaggle:

```bash
kaggle datasets list
```

---

### 3. Download Dataset

From the project root:

```bash
kaggle datasets download -d puneet6060/intel-image-classification -p data/raw
```

Unzip:

```bash
unzip data/raw/intel-image-classification.zip -d data/raw
```

---

## Training the Models

### Train Simple CNN

```bash
python -m src.train
```

This trains the simple custom CNN.

---

### Train Improved CNN

```bash
python -m src.train_improved
```

This trains the improved custom CNN.

---

### Train ResNet18 Transfer Learning Model

```bash
python -m src.train_transfer
```

This trains the best transfer learning model.

The trained model is saved at:

```text
models/saved/resnet18_transfer_mps.pth
```

---

## Evaluating the Models

### Evaluate Simple CNN

```bash
python -m src.evaluate
```

---

### Evaluate Improved CNN

```bash
python -m src.evaluate_improved
```

---

### Evaluate ResNet18 Transfer Model

```bash
python -m src.evaluate_transfer
```

Evaluation generates:

```text
outputs/reports/
outputs/plots/
```

These may include:

- Classification report
- Confusion matrix
- Training loss graph
- Accuracy graph

---

## Running the Flask Web App

The Flask UI uses the best trained ResNet18 model.

Make sure this file exists:

```text
models/saved/resnet18_transfer_mps.pth
```

Then run:

```bash
python app.py
```

You should see:

```text
Running on http://127.0.0.1:5000
```

Open in browser:

```text
http://127.0.0.1:5000
```

---

## How to Use the Web App

1. Open the Flask app in your browser.
2. Upload a scene image.
3. Click **Predict Image**.
4. The app will show:
   - Uploaded image preview
   - Predicted class
   - Confidence score
   - Class probability bars
5. Click **Reset** to clear the page and upload a new image.

---

## Model Inference Flow

The prediction pipeline works like this:

```text
User uploads image
        ↓
Flask saves image temporarily
        ↓
Image is resized to 224x224
        ↓
Image is normalized using ImageNet mean and std
        ↓
Image tensor is sent to MPS or CPU
        ↓
ResNet18 model predicts class probabilities
        ↓
Top prediction and confidence are displayed in UI
```

---

## Classes Supported by the Model

```python
CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]
```

---

## Important Notes

### Do Not Push Large Files to GitHub

The following should usually be ignored:

```text
data/raw/
data/processed/
.venv/
*.zip
static/uploads/
```

If your model file is larger than 100 MB, GitHub may reject it.

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.pyc
.DS_Store

data/raw/
data/processed/
*.zip

static/uploads/*
!static/uploads/.gitkeep

models/saved/*.pth

kaggle.json
```

If you want to share the model, upload it separately using:

- Google Drive
- Hugging Face Model Hub
- GitHub Releases
- Git LFS

---

## Common Issues and Fixes

### 1. MPS Not Available

Check:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

If output is `False`, use CPU or check your PyTorch/macOS setup.

---

### 2. Model Loading Error

If you see a state dictionary mismatch error, it means training and evaluation architectures are different.

Make sure the ResNet18 final layer in `train_transfer.py`, `evaluate_transfer.py`, and `app.py` is the same:

```python
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)
```

---

### 3. CSS Not Loading in Flask

Do not open `index.html` directly.

Run:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

CSS file should be here:

```text
static/css/style.css
```

HTML should link CSS like this:

```html
<link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
```

---

### 4. Dataset Path Error

Make sure dataset path is:

```text
data/raw/seg_train/seg_train/
data/raw/seg_test/seg_test/
```

The class folders should be inside these directories.

---

## Future Improvements

Possible future upgrades:

- Add EfficientNet-B0 transfer learning
- Add ResNet50 model comparison
- Add FastAPI version of the backend
- Add drag-and-drop image upload
- Add batch prediction support
- Add Grad-CAM explainability
- Deploy the app on Render, Railway, Hugging Face Spaces, or AWS
- Add Docker support
- Add GitHub Actions for testing

---

## Learning Outcomes

This project helps understand:

- PyTorch tensors
- Image preprocessing
- Dataset and DataLoader
- CNN architecture
- Batch normalization
- Dropout
- Transfer learning
- Fine-tuning pretrained models
- Model evaluation
- Saving and loading `.pth` files
- Flask web deployment
- Apple MPS acceleration

---

## Author

**Ahmed Ali**

AI / Machine Learning Developer

---

## License

This project is for learning, practice, and portfolio demonstration.  
You may modify and extend it for your own educational use.
