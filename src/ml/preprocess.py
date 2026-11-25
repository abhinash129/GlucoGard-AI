import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_RAW, RANDOM_STATE

# In Pima, a 0 in these columns is physically impossible, so it means "missing"
PIMA_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def _find(folder, name):
    matches = sorted(folder.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {folder}")
    return matches[0]


def load_pima():
    df = pd.read_csv(_find(DATA_RAW / "pima", "diabetes.csv"))
    df[PIMA_ZERO_AS_MISSING] = df[PIMA_ZERO_AS_MISSING].replace(0, np.nan)
    return df.drop(columns="Outcome"), df["Outcome"]


def load_brfss():
    df = pd.read_csv(_find(DATA_RAW / "brfss",
                           "diabetes_binary_health_indicators_BRFSS2015.csv"))
    return df.drop(columns="Diabetes_binary"), df["Diabetes_binary"].astype(int)


def split(X, y, test_size=0.2):
    """Stratified split keeps the same diabetic ratio in train and test."""
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=RANDOM_STATE)


                            
                            
                            
                            
COMPLICATION_COLS = ["Stroke", "HeartDiseaseorAttack"]


def load_brfss_complications():
    """Diabetic patients only; target = stroke or heart disease/attack."""
    df = pd.read_csv(_find(DATA_RAW / "brfss",
                           "diabetes_binary_health_indicators_BRFSS2015.csv"))
    df = df[df["Diabetes_binary"] == 1].copy()
    y = ((df["Stroke"] == 1) | (df["HeartDiseaseorAttack"] == 1)).astype(int)
    X = df.drop(columns=["Diabetes_binary"] + COMPLICATION_COLS)
    return X, y