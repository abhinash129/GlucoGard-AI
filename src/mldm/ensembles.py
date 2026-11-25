import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)
from src.config import MLDM_DIR, RANDOM_STATE
from src.ml.preprocess import load_pima, load_brfss, split

RS = RANDOM_STATE


def heading(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def get_models(big):
    """big=True for BRFSS (about 56,000 balanced training rows): fewer, larger trees to stay fast."""
    leaf = 20 if big else 3
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000), True),
        "Decision Tree (depth 5)": (DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=RS), False),
        "Decision Tree (unpruned)": (DecisionTreeClassifier(random_state=RS), False),
        "Bagging (trees)": (BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS),
                                              n_estimators=50 if big else 100, random_state=RS, n_jobs=-1), False),
        "Random Forest": (RandomForestClassifier(n_estimators=200 if big else 300, min_samples_leaf=leaf,
                                                 random_state=RS, n_jobs=-1), False),
        "AdaBoost (stumps)": (AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=200,
                                                 learning_rate=0.5, algorithm="SAMME", random_state=RS), False),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                                         subsample=0.8, random_state=RS), False),
    }


def build(model, scale, sampler):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps += [("sampler", sampler), ("model", model)]
    return Pipeline(steps)


def metrics(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred),
            "roc_auc": roc_auc_score(y_true, proba)}


def run_dataset(name, X, y, sampler_factory, big, do_cv):
    heading(f"ENSEMBLES ON {name.upper()}")
    X_train, X_test, y_train, y_test = split(X, y)
    print("train:", X_train.shape, "test:", X_test.shape)
    rows, fitted, probas = [], {}, {}
    for mname, (model, scale) in get_models(big).items():
        pipe = build(model, scale, sampler_factory())
        print(f"training {mname} ...", flush=True)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        proba = pipe.predict_proba(X_test)[:, 1]
        row = {"model": mname, **metrics(y_test, proba)}
        row["train_auc"] = roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])
        row["gap"] = row["train_auc"] - row["roc_auc"]
        row["fit_seconds"] = fit_s
        if do_cv:    # 5 folds repeated 3 times: a steadier estimate on a small dataset
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
            m2, s2 = get_models(big)[mname]
            scores = cross_val_score(build(m2, s2, sampler_factory()), X_train, y_train, cv=cv, scoring="roc_auc")
            row["cv_auc"], row["cv_std"] = scores.mean(), scores.std()
        rows.append(row)
        fitted[mname], probas[mname] = pipe, proba

    res = pd.DataFrame(rows).set_index("model")
    print("\n", res.round(3).to_string())
    res.round(4).to_csv(MLDM_DIR / f"ensemble_results_{name}.csv")

    plt.figure(figsize=(6.2, 5))
    for mname, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, label=f"{mname} ({res.loc[mname, 'roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
    plt.title(f"ROC curves, {name}"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(MLDM_DIR / f"ensemble_roc_{name}.png", dpi=150); plt.close()
    return fitted, (X_train, X_test, y_train, y_test)


def importances(name, fitted, X, data, rows, big):
    X_train, X_test, y_train, y_test = data
    for mname in ("Random Forest", "Gradient Boosting"):
        imp = fitted[mname].named_steps["model"].feature_importances_
        for f, v in zip(X.columns, imp):
            rows.append({"dataset": name, "model": mname, "feature": f, "importance": v})
    rf = pd.Series(fitted["Random Forest"].named_steps["model"].feature_importances_, index=X.columns)
    print(f"\ntop Random Forest importances ({name}):")
    print(rf.sort_values(ascending=False).head(6).round(3).to_string())
    top = rf.sort_values().tail(10)
    plt.figure(figsize=(5.5, 3.8))
    plt.barh(top.index, top.values, color="#4c72b0")
    plt.title(f"Random Forest feature importance, {name}")
    plt.tight_layout(); plt.savefig(MLDM_DIR / f"ensemble_importance_{name}.png", dpi=150); plt.close()
    if not big:   # permutation importance needs many predictions; run it on the small dataset only
        pi = permutation_importance(fitted["Random Forest"], X_test, y_test, scoring="roc_auc",
                                    n_repeats=30, random_state=RS)
        ps = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
        print(f"\npermutation importance of the Random Forest on the {name} test set (drop in AUC):")
        print(ps.round(3).to_string())


def size_curve(X, y):
    heading("EFFECT OF ENSEMBLE SIZE (PIMA TEST AUC)")
    X_tr, X_te, y_tr, y_te = split(X, y)
    imp = SimpleImputer(strategy="median").fit(X_tr)
    Xr, yr = SMOTE(random_state=RS).fit_resample(imp.transform(X_tr), y_tr)
    Xt = imp.transform(X_te)
    sizes = [1, 2, 5, 10, 25, 50, 100, 200, 300]

    def auc(m):
        return roc_auc_score(y_te, m.fit(Xr, yr).predict_proba(Xt)[:, 1])

    curves = {
        "Bagging": [auc(BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS), n_estimators=n,
                                          random_state=RS)) for n in sizes],
        "Random Forest": [auc(RandomForestClassifier(n_estimators=n, min_samples_leaf=3, random_state=RS))
                          for n in sizes]}
    for label, model in (("AdaBoost", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                                          n_estimators=300, learning_rate=0.5,
                                                          algorithm="SAMME", random_state=RS)),
                         ("Gradient Boosting", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                                                          max_depth=3, subsample=0.8,
                                                                          random_state=RS))):
        model.fit(Xr, yr)
        staged = [roc_auc_score(y_te, p[:, 1]) for p in model.staged_predict_proba(Xt)]
        curves[label] = [staged[n - 1] for n in sizes]
    tbl = pd.DataFrame(curves, index=sizes).round(3)
    tbl.index.name = "trees"
    print(tbl.to_string())
    tbl.to_csv(MLDM_DIR / "ensemble_size_curve.csv")
    plt.figure(figsize=(6.2, 3.8))
    for label, vals in curves.items():
        plt.plot(sizes, vals, "o-", ms=3, label=label)
    plt.xscale("log"); plt.xlabel("number of base learners"); plt.ylabel("test ROC-AUC")
    plt.title("Pima: effect of ensemble size (one split, so noisy)"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(MLDM_DIR / "ensemble_size_curve.png", dpi=150); plt.close()


def tuning(X, y):
    heading("HYPERPARAMETER TUNING ON PIMA (GRID SEARCH, 5-FOLD CV ON THE TRAINING SPLIT)")
    X_train, X_test, y_train, y_test = split(X, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    grids = {
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=RS, n_jobs=-1),
                          {"model__max_depth": [3, 5, None], "model__min_samples_leaf": [1, 3, 10]}),
        "Gradient Boosting": (GradientBoostingClassifier(subsample=0.8, random_state=RS),
                              {"model__learning_rate": [0.03, 0.1], "model__n_estimators": [100, 300],
                               "model__max_depth": [2, 3]}),
    }
    for mname, (model, grid) in grids.items():
        gs = GridSearchCV(build(model, False, SMOTE(random_state=RS)), grid, cv=cv, scoring="roc_auc")
        gs.fit(X_train, y_train)
        m = metrics(y_test, gs.predict_proba(X_test)[:, 1])
        print(f"\n{mname}: best parameters {gs.best_params_}")
        print(f"  CV ROC-AUC {gs.best_score_:.3f} | test: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()))


def main():
    rows = []
    Xp, yp = load_pima()
    fitted_p, data_p = run_dataset("pima", Xp, yp, lambda: SMOTE(random_state=RS), big=False, do_cv=True)
    importances("pima", fitted_p, Xp, data_p, rows, big=False)

    Xb, yb = load_brfss()
    fitted_b, data_b = run_dataset("brfss", Xb, yb, lambda: RandomUnderSampler(random_state=RS),
                                   big=True, do_cv=False)
    importances("brfss", fitted_b, Xb, data_b, rows, big=True)
    pd.DataFrame(rows).round(4).to_csv(MLDM_DIR / "ensemble_feature_importance.csv", index=False)

    size_curve(Xp, yp)
    tuning(Xp, yp)
    print(f"\nAll tables and figures were saved in {MLDM_DIR}")


if __name__ == "__main__":
    main()