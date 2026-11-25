import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from src.config import DATA_RAW, MLDM_DIR, RANDOM_STATE
from src.ml.preprocess import load_pima, load_brfss
from src.dl.cgm_data import find_patient_files

BRFSS_ORDINAL = {"GenHlth", "Age", "Education", "Income"}   # coded categories with a natural order
BRFSS_COUNTS = {"MentHlth", "PhysHlth"}                      # days out of 30
PIMA_COUNTS = {"Pregnancies"}


def heading(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def pima_frame():
    X, y = load_pima()                      # zeros in 5 columns are already turned into missing values
    return X.assign(Outcome=y)


def brfss_frame():
    X, y = load_brfss()
    return X.assign(Diabetes_binary=y)


# ------------------------------------------------------------------ 1. KDD framing
def kdd_table():
    heading("1. KDD PROCESS APPLIED TO THIS PROJECT")
    rows = [
        ("1. Selection", "Choose data and targets: Pima, BRFSS 2015, APTOS 2019, HUPA-UCM; targets are the diabetes "
         "label, a complication proxy, the retinopathy grade and future glucose",
         "inspect_data.py, preprocess.py"),
        ("2. Pre-processing", "Zeros as missing values, median imputation, outlier checks, class balancing "
         "(SMOTE, undersampling), image cleaning, removal of CGM gaps",
         "preprocess.py, aptos_data.py, cgm_data.py, eda.py"),
        ("3. Transformation", "Standardisation, Ben Graham image contrast, 18-step CGM windows, age bands, "
         "probability-to-level conversion", "train_classical.py, aptos_data.py, cgm_data.py, pipeline.py"),
        ("4. Data mining", "Classification, regression, clustering, association rules, ensembles, neural networks, "
         "sequence models", "train_classical.py, linear_regression.py, kmeans_phenotypes.py, train_mlp.py, "
         "train_cnn.py, train_lstm.py, clustering_suite.py, association.py, ensembles.py"),
        ("5. Interpretation and evaluation", "Recall, F1, ROC-AUC, kappa, RMSE; ablation; rule-based explanations; "
         "dashboard", "ablation.py, app.py, reports/"),
    ]
    df = pd.DataFrame(rows, columns=["KDD stage", "What was done", "Files"])
    print(df.to_string(index=False))
    df.to_csv(MLDM_DIR / "kdd_mapping.csv", index=False)


# ------------------------------------------------------------------ 2. data types
def column_table(df, dataset, target, ordinal=(), counts=()):
    rows = []
    for c in df.columns:
        s = df[c]
        nun = int(s.nunique(dropna=True))
        if c == target:
            kind, scale = "binary target", "nominal"
        elif nun == 2:
            kind, scale = "binary", "nominal"
        elif c in ordinal:
            kind, scale = "ordinal", "ordinal"
        elif c in counts:
            kind, scale = "discrete count", "ratio"
        else:
            kind, scale = "continuous numeric", "ratio"
        rows.append({"dataset": dataset, "column": c, "data type": kind, "scale": scale, "distinct": nun,
                     "missing": int(s.isna().sum()), "min": round(float(s.min()), 2),
                     "max": round(float(s.max()), 2), "mean": round(float(s.mean()), 2)})
    return pd.DataFrame(rows)


def data_types(pima, brfss):
    heading("2. DATA TYPES")
    structure = pd.DataFrame([
        ("Pima", "tabular, 768 records", "8 numeric measurements", "binary diabetes outcome"),
        ("BRFSS 2015", "tabular, 253,680 records", "binary, ordinal and count survey items", "binary diabetes label"),
        ("APTOS 2019", "images (unstructured), 3,662 files", "fundus photographs", "ordinal grade 0-4"),
        ("HUPA-UCM", "time series, 5-minute steps, 25 patients", "glucose, insulin, carbohydrates, steps, "
         "heart rate, calories", "future glucose (numeric)"),
    ], columns=["dataset", "structure", "content", "target"])
    print(structure.to_string(index=False))
    structure.to_csv(MLDM_DIR / "dataset_structure.csv", index=False)

    cols = pd.concat([
        column_table(pima, "Pima", "Outcome", counts=PIMA_COUNTS),
        column_table(brfss, "BRFSS", "Diabetes_binary", ordinal=BRFSS_ORDINAL, counts=BRFSS_COUNTS)])
    print("\n", cols.to_string(index=False))
    cols.to_csv(MLDM_DIR / "data_types.csv", index=False)
    print("\ncolumn types per dataset:\n", cols.groupby(["dataset", "data type"]).size().to_string())


# ------------------------------------------------------------------ 3. class balance and missing values
def class_balance_and_missing(pima, brfss):
    heading("3. CLASS BALANCE AND MISSING VALUES")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for a, (name, df, t) in zip(ax, [("Pima", pima, "Outcome"), ("BRFSS", brfss, "Diabetes_binary")]):
        counts = df[t].value_counts().sort_index()
        a.bar(["no diabetes", "diabetes"], counts.values, color=["#4c72b0", "#c44e52"])
        for i, v in enumerate(counts.values):
            a.text(i, v, f"{v:,}\n({100 * v / counts.sum():.1f}%)", ha="center", va="bottom", fontsize=8)
        a.set_title(f"{name}: imbalance ratio {counts.max() / counts.min():.1f} : 1")
        a.set_ylim(0, counts.max() * 1.25)
    fig.tight_layout(); fig.savefig(MLDM_DIR / "class_balance.png", dpi=150); plt.close(fig)
    print("saved class_balance.png")

    miss = pima.drop(columns="Outcome").isna().sum()
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.barh(miss.index, miss.values, color="#4c72b0")
    for i, v in enumerate(miss.values):
        ax.text(v, i, f" {v} ({100 * v / len(pima):.0f}%)", va="center", fontsize=8)
    ax.set_xlim(0, miss.max() * 1.3); ax.set_title("Pima: missing values (zeros treated as missing)")
    fig.tight_layout(); fig.savefig(MLDM_DIR / "pima_missing.png", dpi=150); plt.close(fig)

    print("\nPima missing values per column:\n", miss.to_string())
    print("\nshare missing, by outcome class (does missingness depend on the label?):")
    print((pima.groupby("Outcome")[list(miss.index)].apply(lambda g: g.isna().mean()).round(3)).to_string())
    print("\nBRFSS missing values:", int(brfss.isna().sum().sum()))


# ------------------------------------------------------------------ 4. distributions
def plot_distributions(df, target, cols, fname, ncols=4):
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        bins = min(25, int(df[c].nunique()))
        for val, color, lab in ((0, "#4c72b0", "no diabetes"), (1, "#c44e52", "diabetes")):
            ax.hist(df.loc[df[target] == val, c].dropna(), bins=bins, alpha=0.55, density=True,
                    color=color, label=lab)
        ax.set_title(c, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(MLDM_DIR / fname, dpi=150); plt.close(fig)


def distributions(pima, brfss):
    heading("4. DISTRIBUTIONS AND SKEWNESS")
    plot_distributions(pima, "Outcome", [c for c in pima.columns if c != "Outcome"], "pima_distributions.png")
    brfss_cols = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
    plot_distributions(brfss, "Diabetes_binary", brfss_cols, "brfss_distributions.png")

    bins = [c for c in brfss.columns if c != "Diabetes_binary" and brfss[c].nunique() == 2]
    rates = brfss.groupby("Diabetes_binary")[bins].mean().T
    rates.columns = ["no diabetes", "diabetes"]
    ax = rates.sort_values("diabetes").plot.barh(figsize=(7, 6))
    ax.set_xlabel("share of respondents with the indicator"); ax.set_title("BRFSS: binary indicators by class")
    plt.tight_layout(); plt.savefig(MLDM_DIR / "brfss_binary_by_class.png", dpi=150); plt.close()

    rows = []
    for name, df, t in (("Pima", pima, "Outcome"), ("BRFSS", brfss, "Diabetes_binary")):
        for c in df.columns:
            if c != t and df[c].nunique() > 2:
                s = df[c].dropna()
                rows.append({"dataset": name, "column": c, "skewness": round(float(stats.skew(s)), 2),
                             "excess kurtosis": round(float(stats.kurtosis(s)), 2)})
    sk = pd.DataFrame(rows)
    sk.to_csv(MLDM_DIR / "skewness.csv", index=False)
    print("most skewed columns (|skewness| above 1 suggests a log or similar transform):")
    print(sk[sk["skewness"].abs() > 1].sort_values("skewness", key=abs, ascending=False).to_string(index=False))
    print("\nlargest differences in indicator rates between classes (BRFSS):")
    diff = (rates["diabetes"] - rates["no diabetes"]).sort_values(ascending=False)
    print(diff.round(3).head(6).to_string())


# ------------------------------------------------------------------ 5. outliers
def outliers(pima, brfss):
    heading("5. OUTLIERS (IQR RULE AND Z-SCORE)")
    rows = []
    for name, df, t in (("Pima", pima, "Outcome"), ("BRFSS", brfss, "Diabetes_binary")):
        for c in df.columns:
            if c == t or df[c].nunique() <= 2:
                continue
            s = df[c].dropna()
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_iqr = int(((s < lo) | (s > hi)).sum())
            n_z = int((np.abs(stats.zscore(s)) > 3).sum())
            rows.append({"dataset": name, "column": c, "lower fence": round(lo, 2), "upper fence": round(hi, 2),
                         "IQR outliers": n_iqr, "% of rows": round(100 * n_iqr / len(s), 2), "|z| > 3": n_z})
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    tbl.to_csv(MLDM_DIR / "outliers.csv", index=False)

    feats = [c for c in pima.columns if c != "Outcome"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2))
    for ax, c in zip(axes.ravel(), feats):
        groups = [pima.loc[pima["Outcome"] == v, c].dropna() for v in (0, 1)]
        ax.boxplot(groups, showfliers=True, flierprops={"markersize": 3})
        ax.set_xticklabels(["no", "yes"])
        ax.set_title(c, fontsize=9)
    fig.suptitle("Pima: boxplots by diabetes outcome", fontsize=10)
    fig.tight_layout(); fig.savefig(MLDM_DIR / "pima_boxplots.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ 6. correlations
def correlations(pima, brfss):
    heading("6. CORRELATIONS")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, m in zip(axes, ("pearson", "spearman")):
        sns.heatmap(pima.corr(method=m), ax=ax, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f",
                    annot_kws={"size": 7}, cbar=False)
        ax.set_title(f"Pima: {m}")
    fig.tight_layout(); fig.savefig(MLDM_DIR / "pima_corr.png", dpi=150); plt.close(fig)

    corr_b = brfss.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_b, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 5})
    ax.set_title("BRFSS: Spearman correlation (suits ordinal and binary items)")
    fig.tight_layout(); fig.savefig(MLDM_DIR / "brfss_corr.png", dpi=150); plt.close(fig)

    rows = []
    for name, df, t, cm in (("Pima", pima, "Outcome", pima.corr(method="spearman")),
                            ("BRFSS", brfss, "Diabetes_binary", corr_b)):
        pear = df.corr(method="pearson")[t]
        for c in df.columns:
            if c != t:
                rows.append({"dataset": name, "feature": c, "pearson": round(pear[c], 3),
                             "spearman": round(cm.loc[c, t], 3)})
    tt = pd.DataFrame(rows)
    tt["abs"] = tt["spearman"].abs()
    tt = tt.sort_values(["dataset", "abs"], ascending=[True, False]).drop(columns="abs")
    tt.to_csv(MLDM_DIR / "correlation_with_target.csv", index=False)
    for name in ("Pima", "BRFSS"):
        print(f"\nstrongest correlations with the diabetes label ({name}):")
        print(tt[tt["dataset"] == name].head(6).to_string(index=False))

    print("\nhighly correlated feature pairs (|r| above 0.5; possible redundancy):")
    for name, df, t in (("Pima", pima, "Outcome"), ("BRFSS", brfss, "Diabetes_binary")):
        c = df.drop(columns=t).corr(method="spearman")
        pairs = [(a, b, round(c.loc[a, b], 2)) for i, a in enumerate(c.columns) for b in c.columns[i + 1:]
                 if abs(c.loc[a, b]) > 0.5]
        print(f"  {name}:", pairs if pairs else "none")


# ------------------------------------------------------------------ 7. CGM and APTOS
def eda_cgm_aptos():
    heading("7. TIME-SERIES AND IMAGE DATA")
    frames = []
    for f in find_patient_files():
        d = pd.read_csv(f, sep=None, engine="python")
        d.columns = [c.strip() for c in d.columns]
        d["patient"] = f.stem
        frames.append(d[["patient", "glucose"]])
    cgm = pd.concat(frames)
    g = cgm["glucose"]
    print(f"CGM readings: {len(cgm):,} from {cgm['patient'].nunique()} patients; "
          f"missing glucose values: {int(g.isna().sum())}")
    stat = cgm.groupby("patient")["glucose"].agg(mean="mean", std="std", n="count").round(1)
    stat["% below 70"] = cgm.groupby("patient")["glucose"].apply(lambda s: round(100 * (s < 70).mean(), 1))
    stat["% 70-180"] = cgm.groupby("patient")["glucose"].apply(lambda s: round(100 * s.between(70, 180).mean(), 1))
    stat["% above 180"] = cgm.groupby("patient")["glucose"].apply(lambda s: round(100 * (s > 180).mean(), 1))
    stat.to_csv(MLDM_DIR / "cgm_patient_stats.csv")
    print("overall: mean %.0f, median %.0f, below 70: %.1f%%, in range 70-180: %.1f%%, above 180: %.1f%%" % (
        g.mean(), g.median(), 100 * (g < 70).mean(), 100 * g.between(70, 180).mean(), 100 * (g > 180).mean()))
    print(stat.head(8).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].hist(g.dropna(), bins=60, color="#4c72b0")
    ax[0].axvline(70, color="k", ls=":"); ax[0].axvline(180, color="k", ls=":")
    ax[0].set_title("CGM glucose, all patients (dotted: 70 and 180 mg/dL)"); ax[0].set_xlabel("mg/dL")
    order = list(stat.index)
    ax[1].boxplot([cgm.loc[cgm["patient"] == p, "glucose"].dropna() for p in order], showfliers=False)
    ax[1].set_xticks(range(1, len(order) + 1)); ax[1].set_xticklabels(range(1, len(order) + 1), fontsize=7)
    ax[1].set_title("Glucose by patient"); ax[1].set_xlabel("patient number")
    fig.tight_layout(); fig.savefig(MLDM_DIR / "cgm_glucose.png", dpi=150); plt.close(fig)

    csv = sorted((DATA_RAW / "aptos").rglob("train.csv"))[0]
    ap = pd.read_csv(csv)
    counts = ap["diagnosis"].value_counts().sort_index()
    print("\nAPTOS grade counts (ordinal label 0-4):", counts.to_dict())
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(counts.index.astype(str), counts.values, color="#4c72b0")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_title("APTOS 2019: retinopathy grades"); ax.set_xlabel("grade"); ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout(); fig.savefig(MLDM_DIR / "aptos_grades.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ 8. preprocessing experiment
class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips each feature to the IQR fences learned on the training data only."""

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        self.low_, self.high_ = q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


VARIANTS = {1: "raw (zeros kept, no scaling)", 2: "+ zeros as missing, median imputation",
            3: "+ standardisation", 4: "+ outlier clipping (IQR)", 5: "+ SMOTE"}


def build(variant, model):
    steps = []
    if variant >= 2:
        steps.append(("imputer", SimpleImputer(strategy="median")))
    if variant >= 4:
        steps.append(("clip", IQRClipper()))
    if variant >= 3:
        steps.append(("scaler", StandardScaler()))
    if variant >= 5:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


def preprocessing_experiment():
    heading("8. EFFECT OF PRE-PROCESSING ON PIMA (5-FOLD CROSS-VALIDATION)")
    X, y = load_pima()
    X_raw = X.fillna(0)                      # puts the original zeros back
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"auc": "roc_auc", "recall": "recall", "f1": "f1"}
    models = {"Logistic Regression": lambda: LogisticRegression(max_iter=2000),
              "KNN (k=15)": lambda: KNeighborsClassifier(n_neighbors=15),
              "Decision Tree": lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                                              random_state=RANDOM_STATE)}
    rows = []
    for mname, make in models.items():
        for v, vname in VARIANTS.items():
            res = cross_validate(build(v, make()), X_raw if v == 1 else X, y, cv=cv, scoring=scoring)
            rows.append({"model": mname, "step": v, "pre-processing": vname,
                         "ROC-AUC": res["test_auc"].mean(), "AUC std": res["test_auc"].std(),
                         "recall": res["test_recall"].mean(), "F1": res["test_f1"].mean()})
    tbl = pd.DataFrame(rows)
    show = tbl.copy()
    show["ROC-AUC"] = show.apply(lambda r: f"{r['ROC-AUC']:.3f} +/- {r['AUC std']:.3f}", axis=1)
    show[["recall", "F1"]] = show[["recall", "F1"]].round(3)
    print(show.drop(columns="AUC std").to_string(index=False))
    tbl.round(4).to_csv(MLDM_DIR / "preprocessing_effect.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.26
    for i, mname in enumerate(models):
        sub = tbl[tbl["model"] == mname]
        ax.bar(np.arange(5) + (i - 1) * w, sub["ROC-AUC"], w, label=mname)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"step {k}" for k in VARIANTS])
    ax.set_ylim(0.6, 0.9); ax.set_ylabel("ROC-AUC (5-fold CV)")
    ax.set_title("Each step adds one pre-processing technique"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(MLDM_DIR / "preprocessing_effect.png", dpi=150); plt.close(fig)


def main():
    pima, brfss = pima_frame(), brfss_frame()
    kdd_table()
    data_types(pima, brfss)
    class_balance_and_missing(pima, brfss)
    distributions(pima, brfss)
    outliers(pima, brfss)
    correlations(pima, brfss)
    eda_cgm_aptos()
    preprocessing_experiment()
    print(f"\nAll tables and figures were saved in {MLDM_DIR}")


if __name__ == "__main__":
    main()