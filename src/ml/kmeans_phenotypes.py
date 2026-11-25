import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from src.config import MODELS_DIR, FIGURES_DIR, RANDOM_STATE
from src.ml.preprocess import load_pima

X, y = load_pima()
prep = Pipeline([("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler())])
Z = prep.fit_transform(X)

ks = list(range(2, 9))
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Z, km.labels_))
    print(f"k={k}  inertia={km.inertia_:.1f}  silhouette={sils[-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertias, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
ax[1].plot(ks, sils, "o-"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "kmeans_k_selection.png", dpi=150); plt.close()

# choose k in 3..6 with best silhouette (we want more than 2 phenotypes)
candidates = {k: s for k, s in zip(ks, sils) if 3 <= k <= 6}
best_k = max(candidates, key=candidates.get)
print("\nchosen k =", best_k)

km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(Z)

profile = X.copy()
profile["cluster"] = km.labels_
profile["diabetes_rate"] = y.values
summary = profile.groupby("cluster").agg(
    {**{c: "mean" for c in X.columns}, "diabetes_rate": "mean"}).round(2)
summary.insert(0, "patients", profile.groupby("cluster").size())
print("\nphenotype profiles (raw feature means per cluster):\n", summary.to_string())
summary.to_csv(FIGURES_DIR.parent / "kmeans_profiles.csv")

# 2-D picture of the clusters
pts = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Z)
plt.figure(figsize=(6, 5))
plt.scatter(pts[:, 0], pts[:, 1], c=km.labels_, cmap="tab10", s=12)
plt.title(f"Pima patient phenotypes (k={best_k}, PCA view)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "kmeans_clusters.png", dpi=150); plt.close()

joblib.dump({"prep": prep, "kmeans": km, "features": list(X.columns)},
            MODELS_DIR / "pima_kmeans.joblib")