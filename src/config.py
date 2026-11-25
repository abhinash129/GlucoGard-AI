import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")   # silences the joblib core-count warning
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
MLDM_DIR = ROOT / "reports" / "mldm"          # outputs of the ML and data-mining extension
RULES_DIR = ROOT / "src" / "rules"

RANDOM_STATE = 42          # fixed seed so results are reproducible
IMG_SIZE = (128, 128)
CNN_IMG_SIZE = (224, 224)  # MobileNetV2 input size
CGM_WINDOW = 12            # 12 readings x 5 min = 1 hour of history
CGM_HORIZON = 6            # predict 6 steps ahead = 30 minutes

HYPO_THRESHOLD = 70        # mg/dL
HYPER_THRESHOLD = 180      # mg/dL

for d in (DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, MLDM_DIR):
    d.mkdir(parents=True, exist_ok=True)