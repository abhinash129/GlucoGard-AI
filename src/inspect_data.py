from pathlib import Path
import pandas as pd
from src.config import DATA_RAW


def find_one(folder, pattern):
    """Return the first file matching pattern anywhere inside folder, or None."""
    matches = sorted(Path(folder).rglob(pattern))
    return matches[0] if matches else None


def inspect_pima():
    print("=" * 60, "\nPIMA\n" + "=" * 60)
    f = find_one(DATA_RAW / "pima", "diabetes.csv")
    if f is None:
        print("NOT FOUND in data/raw/pima")
        return
    df = pd.read_csv(f)
    print("file:", f.name, "| shape:", df.shape)
    print(df.head(3).to_string())
    print("\nclass balance (Outcome):\n", df["Outcome"].value_counts())
    # In Pima, a 0 in these columns really means "value missing"
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    print("\nzeros that are really missing values:\n", (df[cols] == 0).sum())


def inspect_brfss():
    print("=" * 60, "\nBRFSS\n" + "=" * 60)
    f = find_one(DATA_RAW / "brfss", "*binary_health_indicators*2015.csv")
    if f is None:
        print("NOT FOUND in data/raw/brfss")
        return
    df = pd.read_csv(f)
    print("file:", f.name, "| shape:", df.shape)
    print("columns:", list(df.columns))
    print("\nclass balance (Diabetes_binary):\n",
          df["Diabetes_binary"].value_counts(normalize=True).round(3))
    print("missing values:", int(df.isna().sum().sum()))


def inspect_aptos():
    print("=" * 60, "\nAPTOS 2019\n" + "=" * 60)
    f = find_one(DATA_RAW / "aptos", "train.csv")
    if f is None:
        print("NOT FOUND in data/raw/aptos")
        return
    df = pd.read_csv(f)
    print("file:", f.name, "| shape:", df.shape)
    print("\ngrade counts (0=no DR ... 4=proliferative):\n",
          df["diagnosis"].value_counts().sort_index())
    img_dir = f.parent / "train_images"
    n_imgs = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
    print("images found in train_images:", n_imgs)


def inspect_cgm():
    print("=" * 60, "\nCGM (HUPA-UCM)\n" + "=" * 60)
    files = [p for p in (DATA_RAW / "cgm").rglob("*") if p.is_file()]
    if not files:
        print("NOT FOUND in data/raw/cgm")
        return
    print("files found:", len(files))
    for p in files[:10]:
        print(" -", p.relative_to(DATA_RAW / "cgm"))
    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    if csvs:
        df = pd.read_csv(csvs[0], sep=None, engine="python")
        print("\nfirst csv:", csvs[0].name, "| shape:", df.shape)
        print(df.head(3).to_string())


if __name__ == "__main__":
    inspect_pima()
    inspect_brfss()
    inspect_aptos()
    inspect_cgm()