import numpy as np
import pandas as pd
from src.config import DATA_RAW, HYPO_THRESHOLD
from src.dl.aptos_data import load_image
from src.dl.cgm_data import load_cgm
from src.ml.preprocess import load_pima, load_brfss_complications
from src.integration.pipeline import age_band


def aptos_image(grade):
    csv = sorted((DATA_RAW / "aptos").rglob("train.csv"))[0]
    df = pd.read_csv(csv)
    for code in df.loc[df["diagnosis"] == grade, "id_code"]:
        path = csv.parent / "train_images" / f"{code}.png"
        try:
            load_image(path)                 # skip the few corrupted downloads
            return path
        except Exception:
            continue
    raise RuntimeError(f"no readable image for grade {grade}")


def survey_for(row, sex, age):
    s = row.to_dict()
    s["Sex"] = 1 if sex == "male" else 0
    s["Age"] = age_band(age)
    return s


def build_demo_patients(gg):
    """Example patients assembled from real rows of the datasets (sex/age set by hand)."""
    Xp, yp = load_pima()
    Xs, ys = load_brfss_complications()
    X, Y = load_cgm()["test"]
    last, end = X[:, -1, 0], Y[:, -1]

    low = np.where((end < 65) & (last > 75) & (last < 120))[0][:300]
    pred = gg.forecast_batch(X[low])[:, -1]
    hit = low[pred < HYPO_THRESHOLD]
    low_idx = int(hit[0]) if len(hit) else int(low[0])
    stable_idx = int(np.where((last > 95) & (last < 125) & (np.abs(end - last) < 8))[0][0])

    pos, neg = Xp[yp == 1], Xp[yp == 0]
    sick_survey, well_survey = Xs[ys == 1].iloc[0], Xs[ys == 0].iloc[0]

    return {
        "Example 1: woman (52), high risk, severe retinopathy, falling glucose, on insulin": {
            "sex": "female", "age": 52, "pregnant": False, "on_insulin": True,
            "clinical": dict(pos.iloc[0].to_dict(), Age=52),
            "survey": survey_for(sick_survey, "female", 52),
            "fundus_path": aptos_image(3), "cgm_window": X[low_idx]},
        "Example 2: woman (31), low risk, healthy eyes, stable glucose": {
            "sex": "female", "age": 31, "pregnant": False, "on_insulin": False,
            "clinical": dict(neg.iloc[0].to_dict(), Age=31),
            "survey": survey_for(well_survey, "female", 31),
            "fundus_path": aptos_image(0), "cgm_window": X[stable_idx]},
        "Example 3: man (72), high risk, on insulin, falling glucose": {
            "sex": "male", "age": 72, "pregnant": False, "on_insulin": True,
            "clinical": dict(pos.iloc[0].to_dict(), Age=72, Pregnancies=np.nan),
            "survey": survey_for(sick_survey, "male", 72),
            "fundus_path": aptos_image(2), "cgm_window": X[low_idx]},
        "Example 4: pregnant woman (29), high glucose, on insulin": {
            "sex": "female", "age": 29, "pregnant": True, "on_insulin": True,
            "clinical": dict(pos.iloc[1].to_dict(), Age=29),
            "survey": survey_for(well_survey, "female", 29)},
    }