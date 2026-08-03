import os
import torch

DATA_ROOT = os.environ.get(
    "BRAIN_MRI_DATA_ROOT",
    "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset",
)
TRAIN_DIR = os.path.join(DATA_ROOT, "Training")
TEST_DIR = os.path.join(DATA_ROOT, "Testing")

CHECKPOINT_PATH = os.environ.get("BRAIN_MRI_CHECKPOINT_PATH", "best_model.pth")
ARTIFACTS_DIR = os.environ.get("BRAIN_MRI_ARTIFACTS_DIR", "artifacts")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LATENT_DIM = 256
CLS_LOSS_WEIGHT = 1.0
RECON_LOSS_WEIGHT = 0.3

EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5
SEED = 42

API_HOST = os.environ.get("BRAIN_MRI_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("BRAIN_MRI_API_PORT", "8000"))