import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             classification_report, roc_auc_score, recall_score)
from src.config import MODELS_DIR, FIGURES_DIR, RANDOM_STATE, CNN_IMG_SIZE
from src.dl.aptos_data import load_aptos

keras.utils.set_random_seed(RANDOM_STATE)
CLASS_NAMES = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 Proliferative"]


def build_backbone():
    """Frozen MobileNetV2 (ImageNet weights) that turns an image into 1280 numbers."""
    base = keras.applications.MobileNetV2(weights="imagenet", include_top=False,
                                          input_shape=(*CNN_IMG_SIZE, 3), pooling="avg")
    base.trainable = False
    return keras.Sequential([
        keras.layers.Input(shape=(*CNN_IMG_SIZE, 3)),
        keras.layers.Rescaling(1 / 127.5, offset=-1),   # pixels 0..255 -> -1..1
        base,
    ], name="backbone")


def main():
    X, y = load_aptos()
    print("images:", X.shape, "| grade counts:", np.bincount(y))

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    tr, va = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=RANDOM_STATE)

    backbone = build_backbone()
    print("\nextracting CNN features (one pass, a few minutes on CPU) ...")
    feats = backbone.predict(X, batch_size=32, verbose=1)

    head = keras.Sequential([
        keras.layers.Input(shape=(feats.shape[1],)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(5, activation="softmax"),
    ], name="head")
    head.compile(optimizer=keras.optimizers.Adam(1e-3),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    w = compute_class_weight("balanced", classes=np.arange(5), y=y[tr])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True)
    head.fit(feats[tr], y[tr], validation_data=(feats[va], y[va]), epochs=100,
             batch_size=64, class_weight=dict(enumerate(w)), callbacks=[stop], verbose=2)

    # ---- evaluation on the untouched test split ----
    prob = head.predict(feats[te], verbose=0)
    pred = prob.argmax(axis=1)
    acc = accuracy_score(y[te], pred)
    qwk = cohen_kappa_score(y[te], pred, weights="quadratic")
    ref_true = (y[te] >= 2).astype(int)               # referable DR = grade 2 or higher
    ref_score = prob[:, 2:].sum(axis=1)
    ref_auc = roc_auc_score(ref_true, ref_score)
    ref_recall = recall_score(ref_true, (ref_score >= 0.5).astype(int))

    print(f"\naccuracy: {acc:.3f} | quadratic weighted kappa: {qwk:.3f}")
    print(f"referable DR (grade>=2): ROC-AUC {ref_auc:.3f} | sensitivity {ref_recall:.3f}")
    cm = confusion_matrix(y[te], pred)
    print("\nconfusion matrix (rows=true, cols=predicted):\n", cm)
    print("\n", classification_report(y[te], pred, target_names=CLASS_NAMES, zero_division=0))

    pd.DataFrame([{"accuracy": acc, "quadratic_kappa": qwk,
                   "referable_auc": ref_auc, "referable_sensitivity": ref_recall}]
                 ).round(3).to_csv(FIGURES_DIR.parent / "cnn_results.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(5), range(5)); plt.yticks(range(5), range(5))
    for i in range(5):
        for j in range(5):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xlabel("predicted grade"); plt.ylabel("true grade")
    plt.title("Retinopathy grading - test set")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "cnn_confusion.png", dpi=150); plt.close()

    # ---- join backbone + head into one model that accepts raw images ----
    full = keras.Sequential([backbone, head], name="dr_cnn")
    check = np.abs(full.predict(X[te[:8]], verbose=0) - head.predict(feats[te[:8]], verbose=0)).max()
    print(f"\nfull model matches head on feature input (max diff {check:.6f})")
    full.save(MODELS_DIR / "dr_cnn.keras")
    print("saved models/dr_cnn.keras")


if __name__ == "__main__":
    main()