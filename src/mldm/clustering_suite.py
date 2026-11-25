import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score, pairwise_distances)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
from src.config import MLDM_DIR, RANDOM_STATE
from src.ml.preprocess import load_pima, load_brfss

RS = RANDOM_STATE
K = 3          # number of clusters used in the main comparison (same k as the earlier K-Means study)


def heading(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def prepare(X):
    """Median imputation and standardisation. The diabetes label is never used for clustering."""
    return StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))


# ------------------------------------------------------------------ k-medoids (PAM)
def pam(D, k, max_iter=100):
    """Partitioning Around Medoids on a precomputed distance matrix D (BUILD, then SWAP)."""
    n = len(D)
    medoids = [int(np.argmin(D.sum(axis=1)))]
    nearest = D[:, medoids[0]].copy()
    while len(medoids) < k:                                   # BUILD: greedily add the best medoid
        gain = np.maximum(nearest[:, None] - D, 0).sum(axis=0)
        gain[medoids] = -1
        j = int(np.argmax(gain))
        medoids.append(j)
        nearest = np.minimum(nearest, D[:, j])
    medoids = np.array(medoids)
    for _ in range(max_iter):                                 # SWAP: replace a medoid if it lowers the cost
        dm = D[:, medoids]
        order = np.argsort(dm, axis=1)
        a1 = order[:, 0]
        d1 = dm[np.arange(n), a1]
        d2 = dm[np.arange(n), order[:, 1]]
        best, pair = d1.sum(), None
        m1, m2 = np.minimum(D, d1[:, None]), np.minimum(D, d2[:, None])
        for a in range(k):
            mask = a1 == a
            cost = m2[mask].sum(axis=0) + m1[~mask].sum(axis=0)
            cost[medoids] = np.inf
            h = int(np.argmin(cost))
            if cost[h] < best - 1e-9:
                best, pair = cost[h], (a, h)
        if pair is None:
            break
        medoids[pair[0]] = pair[1]
    return np.argmin(D[:, medoids], axis=1), medoids


# ------------------------------------------------------------------ SOM
def som_cluster(Z, k, grid=(3, 3), iters=5000):
    """Train a self-organising map, then merge its neurons into k clusters."""
    som = MiniSom(grid[0], grid[1], Z.shape[1], sigma=1.0, learning_rate=0.5,
                  neighborhood_function="gaussian", random_seed=RS)
    som.random_weights_init(Z)
    som.train_random(Z, iters)
    weights = som.get_weights().reshape(-1, Z.shape[1])
    win = np.array([som.winner(x) for x in Z])
    neuron = win[:, 0] * grid[1] + win[:, 1]
    merged = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(weights)
    return merged[neuron], som, neuron


METHODS = {
    "k-means": lambda Z, k: KMeans(n_clusters=k, n_init=10, random_state=RS).fit_predict(Z),
    "Hierarchical (ward)": lambda Z, k: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z),
    "Hierarchical (average)": lambda Z, k: AgglomerativeClustering(n_clusters=k, linkage="average").fit_predict(Z),
    "Hierarchical (complete)": lambda Z, k: AgglomerativeClustering(n_clusters=k, linkage="complete").fit_predict(Z),
    "k-medoids (PAM, Manhattan)": lambda Z, k: pam(pairwise_distances(Z, metric="manhattan"), k)[0],
    "SOM 3x3 (merged)": lambda Z, k: som_cluster(Z, k)[0],
}


# ------------------------------------------------------------------ validation
def evaluate(Z, labels, y):
    mask = labels >= 0                               # DBSCAN marks noise as -1
    n_clusters = len(set(labels[mask]))
    out = {"clusters": n_clusters, "noise %": round(100 * (~mask).mean(), 1),
           "silhouette": np.nan, "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan}
    if n_clusters >= 2 and mask.sum() > n_clusters:
        out["silhouette"] = silhouette_score(Z[mask], labels[mask])
        out["Davies-Bouldin"] = davies_bouldin_score(Z[mask], labels[mask])
        out["Calinski-Harabasz"] = calinski_harabasz_score(Z[mask], labels[mask])
    out["ARI vs diabetes"] = adjusted_rand_score(y, labels)
    out["NMI vs diabetes"] = normalized_mutual_info_score(y, labels)
    out["purity"] = pd.crosstab(labels, y).max(axis=1).sum() / len(y)
    rates = pd.Series(y[mask]).groupby(labels[mask]).mean() if mask.any() else pd.Series([np.nan])
    out["diabetes rate per cluster"] = f"{rates.min():.2f}-{rates.max():.2f}"
    return out


def stability(fn, Z, n_boot=8):
    """Recluster random 80% subsets and compare with the full-data clustering (adjusted Rand index)."""
    rng = np.random.default_rng(RS)
    base = fn(Z)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(len(Z), size=int(0.8 * len(Z)), replace=False)
        aris.append(adjusted_rand_score(base[idx], fn(Z[idx])))
    return float(np.mean(aris)), float(np.std(aris))


def k_selection(name, Z, ks):
    heading(f"{name.upper()}: CHOOSING THE NUMBER OF CLUSTERS")
    rows = []
    for mname in ("k-means", "Hierarchical (ward)", "k-medoids (PAM, Manhattan)"):
        for k in ks:
            lab = METHODS[mname](Z, k)
            rows.append({"method": mname, "k": k, "silhouette": silhouette_score(Z, lab),
                         "Davies-Bouldin": davies_bouldin_score(Z, lab),
                         "Calinski-Harabasz": calinski_harabasz_score(Z, lab)})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(MLDM_DIR / f"clustering_k_selection_{name}.csv", index=False)
    for metric, note in (("silhouette", "higher is better"), ("Davies-Bouldin", "lower is better"),
                         ("Calinski-Harabasz", "higher is better")):
        print(f"\n{metric} ({note}):")
        print(df.pivot(index="k", columns="method", values=metric).round(3).to_string())
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for a, metric in zip(ax, ("silhouette", "Davies-Bouldin", "Calinski-Harabasz")):
        for mname, g in df.groupby("method"):
            a.plot(g["k"], g[metric], "o-", ms=3, label=mname)
        a.set_title(metric); a.set_xlabel("k")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(MLDM_DIR / f"clustering_k_selection_{name}.png", dpi=150); plt.close(fig)


def dbscan_search(name, Z, y):
    heading(f"{name.upper()}: DBSCAN PARAMETER SEARCH")
    rows = []
    for ms in (5, 10, 16):
        kd = NearestNeighbors(n_neighbors=ms).fit(Z).kneighbors(Z)[0][:, -1]
        for q in (0.80, 0.85, 0.90, 0.95):
            eps = round(float(np.quantile(kd, q)), 3)
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z)
            rows.append({"min_samples": ms, "eps": eps, "eps quantile": q, **evaluate(Z, labels, y)})
    grid = pd.DataFrame(rows)
    show = grid[["min_samples", "eps", "clusters", "noise %", "silhouette", "ARI vs diabetes"]].round(3)
    print(show.to_string(index=False))
    grid.round(4).to_csv(MLDM_DIR / f"dbscan_grid_{name}.csv", index=False)
    valid = grid[(grid["clusters"] >= 2) & (grid["noise %"] <= 25)]
    if len(valid):
        best = valid.sort_values("silhouette", ascending=False).iloc[0]
        print(f"\nchosen: min_samples={int(best['min_samples'])}, eps={best['eps']} "
              f"(best silhouette among settings with 2 or more clusters and at most 25% noise)")
        return int(best["min_samples"]), float(best["eps"])
    print("\nno setting found 2 or more clusters with at most 25% noise; using min_samples=10, eps=90th percentile")
    kd = NearestNeighbors(n_neighbors=10).fit(Z).kneighbors(Z)[0][:, -1]
    return 10, round(float(np.quantile(kd, 0.90)), 3)


def compare(name, Z, y, k, db_params):
    heading(f"{name.upper()}: COMPARISON AT k = {k}")
    print(f"majority-class share (baseline purity): {max(np.mean(y), 1 - np.mean(y)):.3f}")
    rows, labels_all = [], {}
    for mname, fn in METHODS.items():
        t0 = time.perf_counter()
        lab = fn(Z, k)
        secs = time.perf_counter() - t0
        labels_all[mname] = lab
        m, s = stability(lambda Zs: fn(Zs, k), Z)
        rows.append({"method": mname, **evaluate(Z, lab, y), "stability ARI": m, "stability std": s,
                     "seconds": secs})
        print(f"done: {mname}", flush=True)
    ms, eps = db_params
    lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z)
    labels_all[f"DBSCAN (eps {eps}, min {ms})"] = lab
    rows.append({"method": f"DBSCAN (eps {eps}, min {ms})", **evaluate(Z, lab, y),
                 "stability ARI": np.nan, "stability std": np.nan, "seconds": np.nan})
    tbl = pd.DataFrame(rows).set_index("method")
    print("\n", tbl.round(3).to_string())
    tbl.round(4).to_csv(MLDM_DIR / f"clustering_comparison_{name}.csv")
    return labels_all


# ------------------------------------------------------------------ figures (Pima)
def pima_figures(Z, y, labels_all):
    link = linkage(Z, "ward")
    cut = (link[-3, 2] + link[-2, 2]) / 2
    fig, ax = plt.subplots(figsize=(9, 4))
    dendrogram(link, truncate_mode="lastp", p=30, show_leaf_counts=True, no_labels=True, ax=ax,
               color_threshold=cut)
    ax.axhline(cut, color="k", ls="--", lw=0.8)
    ax.set_title("Pima: ward dendrogram (30 last merges; dashed line gives 3 clusters)")
    ax.set_ylabel("merge distance")
    fig.tight_layout(); fig.savefig(MLDM_DIR / "pima_dendrogram.png", dpi=150); plt.close(fig)

    print("\ncophenetic correlation (how faithfully the tree keeps the original distances):")
    dist = pdist(Z)
    for method in ("ward", "average", "complete"):
        c, _ = cophenet(linkage(Z, method), dist)
        print(f"  {method}: {c:.3f}")

    kd = np.sort(NearestNeighbors(n_neighbors=10).fit(Z).kneighbors(Z)[0][:, -1])
    plt.figure(figsize=(5.5, 3.5))
    plt.plot(kd)
    plt.xlabel("points sorted by distance"); plt.ylabel("distance to 10th neighbour")
    plt.title("Pima: k-distance plot for DBSCAN"); plt.tight_layout()
    plt.savefig(MLDM_DIR / "pima_kdistance.png", dpi=150); plt.close()

    _, som, neuron = som_cluster(Z, K)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(som.distance_map().T, cmap="Greys")
    plt.colorbar(im, ax=ax, label="distance to neighbouring neurons (U-matrix)")
    win = np.array([som.winner(x) for x in Z])
    for i in range(3):
        for j in range(3):
            sel = (win[:, 0] == i) & (win[:, 1] == j)
            if sel.any():
                ax.text(i, j, f"n={sel.sum()}\n{100 * y[sel].mean():.0f}% diab.", ha="center", va="center",
                        fontsize=7, color="tab:red")
    ax.set_title("Pima: SOM map (n = patients, % with diabetes)", fontsize=9)
    fig.tight_layout(); fig.savefig(MLDM_DIR / "pima_som.png", dpi=150); plt.close(fig)
    print(f"\nSOM quantisation error {som.quantization_error(Z):.3f}, topographic error "
          f"{som.topographic_error(Z):.3f}")

    pts = PCA(n_components=2, random_state=RS).fit_transform(Z)
    names = list(labels_all)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for a, mname in zip(axes.ravel(), names):
        lab = labels_all[mname]
        a.scatter(pts[lab < 0, 0], pts[lab < 0, 1], c="lightgray", s=6)
        a.scatter(pts[lab >= 0, 0], pts[lab >= 0, 1], c=lab[lab >= 0], cmap="tab10", s=8)
        a.set_title(mname, fontsize=8)
    for a in axes.ravel()[len(names):]:
        a.axis("off")
    fig.suptitle("Pima clusters in the first two principal components (grey = noise)", fontsize=10)
    fig.tight_layout(); fig.savefig(MLDM_DIR / "pima_clusters_pca.png", dpi=150); plt.close(fig)


def main():
    Xp, yp = load_pima()
    Zp, yp = prepare(Xp), yp.to_numpy()
    k_selection("pima", Zp, range(2, 9))
    db = dbscan_search("pima", Zp, yp)
    labels = compare("pima", Zp, yp, K, db)
    pima_figures(Zp, yp, labels)

    Xb, yb = load_brfss()
    sample = Xb.sample(2000, random_state=RS)
    Zb, yb_s = prepare(sample), yb.loc[sample.index].to_numpy()
    k_selection("brfss_sample", Zb, range(2, 7))
    db_b = dbscan_search("brfss_sample", Zb, yb_s)
    compare("brfss_sample", Zb, yb_s, K, db_b)
    print(f"\nAll tables and figures were saved in {MLDM_DIR}")


if __name__ == "__main__":
    main()