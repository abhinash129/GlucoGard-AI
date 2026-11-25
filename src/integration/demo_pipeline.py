import datetime
import numpy as np
import pandas as pd
from src.config import DATA_RAW, HYPO_THRESHOLD
from src.dl.aptos_data import load_image
from src.dl.cgm_data import load_cgm
from src.ml.preprocess import load_pima, load_brfss_complications
from src.integration.pipeline import GlucoGuard, print_report


def aptos_image(grade):
    csv = sorted((DATA_RAW / "aptos").rglob("train.csv"))[0]
    df = pd.read_csv(csv)
    for code in df.loc[df["diagnosis"] == grade, "id_code"]:
        path = csv.parent / "train_images" / f"{code}.png"
        try:
            load_image(path)            # skip the few corrupted downloads
            return path
        except Exception:
            continue
    raise RuntimeError(f"no readable image for grade {grade}")


def main():
    gg = GlucoGuard()
    Xp, yp = load_pima()
    Xs, ys = load_brfss_complications()
    X, Y = load_cgm()["test"]
    last, end = X[:, -1, 0], Y[:, -1]

    # a window where the true glucose falls below 65, preferring one the LSTM also flags
    low = np.where((end < 65) & (last > 75) & (last < 120))[0][:300]
    pred = gg.forecast_batch(X[low])[:, -1]
    hit = low[pred < HYPO_THRESHOLD]
    low_idx = int(hit[0]) if len(hit) else int(low[0])
    stable_idx = int(np.where((last > 95) & (last < 125) & (np.abs(end - last) < 8))[0][0])

    today = datetime.date.today().weekday()      # 0 = Monday; planning starts today

    patients = {
        "PATIENT 1: high risk, severe retinopathy, falling glucose, on insulin": {
            "clinical": Xp[yp == 1].iloc[0].to_dict(),
            "survey": Xs[ys == 1].iloc[0].to_dict(),
            "fundus_path": aptos_image(3),
            "cgm_window": X[low_idx],
            "on_insulin": True},
        "PATIENT 2: low risk, healthy eyes, stable glucose": {
            "clinical": Xp[yp == 0].iloc[0].to_dict(),
            "survey": Xs[ys == 0].iloc[0].to_dict(),
            "fundus_path": aptos_image(0),
            "cgm_window": X[stable_idx],
            "on_insulin": False},
    }
    for title, patient in patients.items():
        print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
        print_report(gg.assess(patient, start_weekday=today))


if __name__ == "__main__":
    main()