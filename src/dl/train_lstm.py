import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.linear_model import Ridge
from sklearn.metrics import precision_score, recall_score
from src.config import (MODELS_DIR, FIGURES_DIR, RANDOM_STATE, CGM_WINDOW,
                        CGM_HORIZON, HYPO_THRESHOLD, HYPER_THRESHOLD)
from src.dl.cgm_data import load_cgm, FEATURES

keras.utils.set_random_seed(RANDOM_STATE)
GLU_SCALE = 50.0      # the model predicts the glucose CHANGE divided by this


def to_targets(X_raw, Y_raw):
    last = X_raw[:, -1, 0]                                   # last observed glucose
    return (Y_raw - last[:, None]) / GLU_SCALE


def to_glucose(X_raw, out):
    return X_raw[:, -1, 0][:, None] + out * GLU_SCALE        # back to mg/dL


def build_lstm(n_features):
    return keras.Sequential([
        keras.layers.Input(shape=(CGM_WINDOW, n_features)),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(CGM_HORIZON),                     # 6 outputs: +5 ... +30 min
    ])


def scores(name, Y_true, Y_pred):
    err = Y_pred - Y_true
    rmse_h = np.sqrt((err ** 2).mean(axis=0))
    row = {"model": name,
           "RMSE_30min": rmse_h[-1],
           "MAE_30min": np.abs(err[:, -1]).mean(),
           "RMSE_avg": rmse_h.mean()}
    t30, p30 = Y_true[:, -1], Y_pred[:, -1]
    for label, fn in (("hypo", lambda g: g < HYPO_THRESHOLD),
                      ("hyper", lambda g: g > HYPER_THRESHOLD)):
        t, p = fn(t30), fn(p30)
        row[f"{label}_recall"] = recall_score(t, p, zero_division=0)
        row[f"{label}_precision"] = precision_score(t, p, zero_division=0)
    return row, rmse_h


def main():
    data = load_cgm()
    Xtr, Ytr = data["train"]; Xva, Yva = data["val"]; Xte, Yte = data["test"]

    mean = Xtr.reshape(-1, Xtr.shape[2]).mean(axis=0)
    std = Xtr.reshape(-1, Xtr.shape[2]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    norm = lambda X: (X - mean) / std

    ttr, tva = to_targets(Xtr, Ytr), to_targets(Xva, Yva)

    test_events = {"hypo": (Yte[:, -1] < HYPO_THRESHOLD).mean(),
                   "hyper": (Yte[:, -1] > HYPER_THRESHOLD).mean()}
    print("share of test windows with an event at +30 min:",
          {k: round(float(v), 3) for k, v in test_events.items()})

    # --- baseline 1: persistence ---
    pred_persist = np.repeat(Xte[:, -1, 0][:, None], CGM_HORIZON, axis=1)
    # --- baseline 2: Ridge regression on the flattened window ---
    ridge = Ridge(alpha=1.0).fit(norm(Xtr).reshape(len(Xtr), -1), ttr)
    pred_ridge = to_glucose(Xte, ridge.predict(norm(Xte).reshape(len(Xte), -1)))

    # --- LSTM ---
    model = build_lstm(len(FEATURES))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                         restore_best_weights=True)
    model.fit(norm(Xtr), ttr, validation_data=(norm(Xva), tva), epochs=40,
              batch_size=128, callbacks=[stop], verbose=2)
    pred_lstm = to_glucose(Xte, model.predict(norm(Xte), verbose=0))

    rows, curves = [], {}
    for name, pred in (("Persistence", pred_persist), ("Ridge", pred_ridge),
                       ("LSTM", pred_lstm)):
        row, rmse_h = scores(name, Yte, pred)
        rows.append(row); curves[name] = rmse_h
    results = pd.DataFrame(rows).set_index("model").round(3)
    print("\n", results.to_string())
    results.to_csv(FIGURES_DIR.parent / "lstm_results.csv")

    # --- figures ---
    minutes = np.arange(1, CGM_HORIZON + 1) * 5
    plt.figure(figsize=(6, 4))
    for name, c in curves.items():
        plt.plot(minutes, c, "o-", label=name)
    plt.xlabel("minutes ahead"); plt.ylabel("RMSE (mg/dL)")
    plt.title("Forecast error by horizon (test set)"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "lstm_rmse_by_horizon.png", dpi=150); plt.close()

    hist_t = np.arange(-(CGM_WINDOW - 1) * 5, 1, 5)
    fut_t = np.arange(5, CGM_HORIZON * 5 + 1, 5)
    picks = np.linspace(0, len(Xte) - 1, 4).astype(int)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, i in zip(axes.ravel(), picks):
        ax.plot(hist_t, Xte[i, :, 0], "k-o", label="history")
        ax.plot(fut_t, Yte[i], "g-o", label="true")
        ax.plot(fut_t, pred_lstm[i], "r-o", label="LSTM")
        ax.plot(fut_t, pred_persist[i], "b--", label="persistence")
        ax.axhline(HYPO_THRESHOLD, color="gray", ls=":"); ax.axhline(HYPER_THRESHOLD, color="gray", ls=":")
        ax.set_xlabel("minutes"); ax.set_ylabel("glucose (mg/dL)")
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "lstm_examples.png", dpi=150); plt.close()

    model.save(MODELS_DIR / "cgm_lstm.keras")
    joblib.dump({"mean": mean, "std": std, "features": FEATURES, "scale": GLU_SCALE,
                 "window": CGM_WINDOW, "horizon": CGM_HORIZON},
                MODELS_DIR / "cgm_lstm_meta.joblib")
    print("\nsaved models/cgm_lstm.keras and cgm_lstm_meta.joblib")


if __name__ == "__main__":
    main()