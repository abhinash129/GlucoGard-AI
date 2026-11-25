import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieter TensorFlow logs
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from src.config import MODELS_DIR, FIGURES_DIR, RANDOM_STATE
from src.ml.preprocess import load_brfss_complications, split

keras.utils.set_random_seed(RANDOM_STATE)


def report(name, y_true, score, pred):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, score),
    }


def build_mlp(n_features):
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),       # randomly drops units to reduce overfitting
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),   # output = risk score in 0..1
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    return model


def main():
    X, y = load_brfss_complications()
    print("diabetic patients:", len(X), "| complication rate:", round(y.mean(), 3))

    X_train, X_test, y_train, y_test = split(X, y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE)

    scaler = StandardScaler().fit(X_tr)          # fit on training data only
    Xtr, Xval, Xte = (scaler.transform(a) for a in (X_tr, X_val, X_test))

    # 1) Perceptron baseline (single neuron, no hidden layer)
    perc = Perceptron(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    perc.fit(Xtr, y_tr)
    rows = [report("Perceptron", y_test, perc.decision_function(Xte), perc.predict(Xte))]

    # 2) Logistic Regression as a classical reference
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=RANDOM_STATE).fit(Xtr, y_tr)
    p_lr = lr.predict_proba(Xte)[:, 1]
    rows.append(report("LogisticRegression", y_test, p_lr, (p_lr >= 0.5).astype(int)))

    # 3) MLP, with class weights because complications are the minority class
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    model = build_mlp(Xtr.shape[1])
    stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=8, restore_best_weights=True)
    hist = model.fit(Xtr, y_tr, validation_data=(Xval, y_val), epochs=60,
                     batch_size=256, class_weight={0: w[0], 1: w[1]},
                     callbacks=[stop], verbose=2)
    p_mlp = model.predict(Xte, verbose=0).ravel()
    rows.append(report("MLP", y_test, p_mlp, (p_mlp >= 0.5).astype(int)))

    results = pd.DataFrame(rows).set_index("model").round(3)
    print("\n", results.to_string())
    results.to_csv(FIGURES_DIR.parent / "mlp_results.csv")

    # training curves for the report
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val")
    ax[0].set_title("MLP loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist.history["auc"], label="train"); ax[1].plot(hist.history["val_auc"], label="val")
    ax[1].set_title("MLP AUC"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "mlp_training.png", dpi=150); plt.close()

    print("\nrisk score distribution on test set (quantiles):")
    print(pd.Series(p_mlp).quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_string())

    model.save(MODELS_DIR / "complication_mlp.keras")
    joblib.dump({"scaler": scaler, "features": list(X.columns)},
                MODELS_DIR / "complication_scaler.joblib")
    print("\nsaved models/complication_mlp.keras and complication_scaler.joblib")


if __name__ == "__main__":
    main()