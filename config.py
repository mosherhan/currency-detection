import os

# Dataset paths
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Mapping of class names to denominations
# These should match the folder names in dataset/train
CLASSES = ["10", "20", "50", "100", "200", "500"]

# Path to save/load model
MODEL_PATH = "models/currency_model.h5"

# Camera settings
CAMERA_ID = 0  # 0 for default webcam
CONFIDENCE_THRESHOLD = 0.7
