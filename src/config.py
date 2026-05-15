import torch
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_DIR = RAW_DATA_DIR / "seg_train" / "seg_train"
TEST_DIR = RAW_DATA_DIR / "seg_test" / "seg_test"

MODEL_DIR = BASE_DIR / "models" / "saved"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

MODEL_PATH = MODEL_DIR / "improved_cnn_mps.pth"
IMPROVED_MODEL_PATH=MODEL_DIR/'improved_cnn_mps.pth'
TRANSFER_MODEL_PATH=MODEL_DIR /'resnet18_transfer_mps.pth'

# Training settings

IMAGE_SIZE = 150
TRANSFER_IMAGE_SIZE = 224

BATCH_SIZE = 32
EPOCHS = 10
TRANSFER_EPOCHS = 8

LEARNING_RATE = 0.001
TRANSFER_LEARNING_RATE = 0.0001

NUM_CLASSES = 6

# Apple MPS device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]