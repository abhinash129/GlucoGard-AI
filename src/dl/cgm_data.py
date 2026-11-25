import numpy as np
import pandas as pd
from src.config import DATA_RAW, CGM_WINDOW, CGM_HORIZON

FEATURES = ["glucose", "carb_input", "bolus_volume_delivered",
            "basal_rate", "steps", "heart_rate", "calories"]   # glucose must stay first
TRAIN_END, VAL_END = 0.70, 0.85      # fractions of each patient's recording


def find_patient_files():
    files = [f for f in sorted((DATA_RAW / "cgm").rglob("HUPA*P.csv"))
             if f.parent.name == "Preprocessed"]
    if not files:
        raise FileNotFoundError("No HUPA*P.csv files found in a 'Preprocessed' folder "
                                "under data/raw/cgm")
    return files


def patient_windows(path):
    # these files are not comma-separated, so let pandas detect the delimiter
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in FEATURES + ["time"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; "
                         f"found {list(df.columns)}")
    df[FEATURES[1:]] = df[FEATURES[1:]].ffill().bfill()      # fill small sensor gaps
    t = pd.to_datetime(df["time"]).to_numpy()
    gap_ok = (np.diff(t) / np.timedelta64(1, "s")) == 300      # exactly 5 minutes apart
    vals = df[FEATURES].to_numpy(dtype=float)

    n, L = len(df), CGM_WINDOW + CGM_HORIZON
    cut1, cut2 = int(TRAIN_END * n), int(VAL_END * n)
    parts = {"train": ([], []), "val": ([], []), "test": ([], [])}

    for s in range(n - L + 1):
        e = s + L                                   # exclusive end row
        if not gap_ok[s:e - 1].all():
            continue
        window = vals[s:s + CGM_WINDOW]
        future = vals[s + CGM_WINDOW:e, 0]          # future glucose only
        if np.isnan(window).any() or np.isnan(future).any():
            continue
        if e <= cut1:
            part = "train"
        elif s >= cut1 and e <= cut2:
            part = "val"
        elif s >= cut2:
            part = "test"
        else:
            continue                                # straddles a boundary: drop (no leakage)
        parts[part][0].append(window)
        parts[part][1].append(future)
    return parts


def load_cgm():
    """Return {'train'|'val'|'test': (X [N,12,7], Y [N,6])} in raw units."""
    collected = {k: ([], []) for k in ("train", "val", "test")}
    files = find_patient_files()
    for f in files:
        parts = patient_windows(f)
        for k in collected:
            collected[k][0].extend(parts[k][0])
            collected[k][1].extend(parts[k][1])
    data = {k: (np.array(v[0]), np.array(v[1])) for k, v in collected.items()}
    print(f"patients: {len(files)} | windows -> " +
          ", ".join(f"{k}: {len(v[0])}" for k, v in data.items()))
    return data