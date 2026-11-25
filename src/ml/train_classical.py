import matplotlib
matplotlib.use("Agg")  # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from src.config import MODELS_DIR, FIGURES_DIR, RANDOM_STATE
from src.ml.preprocess import load_pima, load_brfss, split

REPORTS_DIR = FIGURES_DIR.parent


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                               random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }


def build_pipeline(model, sampler):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # fill missing values
        ("scaler", StandardScaler()),                   # needed by KNN and LR
        ("sampler", sampler),                           # applied only during fit
        ("model", model),
    ])


def evaluate(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    return metrics, proba, pred


def run_dataset(name, X, y, sampler_factory, do_cv):
    print("\n" + "=" * 60 + f"\nDATASET: {name.upper()}\n" + "=" * 60)
    X_train, X_test, y_train, y_test = split(X, y)
    print("train:", X_train.shape, "test:", X_test.shape)

    rows, fitted, probas = [], {}, {}
    for model_name, model in get_models().items():
        pipe = build_pipeline(model, sampler_factory())
        print(f"training {model_name} ...", flush=True)
        pipe.fit(X_train, y_train)
        metrics, proba, pred = evaluate(pipe, X_test, y_test)

        if do_cv:  # 5-fold CV gives a more stable estimate on small data
            cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
            metrics["cv_roc_auc"] = cross_val_score(
                build_pipeline(get_models()[model_name], sampler_factory()),
                X_train, y_train, cv=cv, scoring="roc_auc").mean()

        rows.append({"model": model_name, **metrics})
        fitted[model_name], probas[model_name] = pipe, proba
        print(f"confusion matrix {model_name}:\n", confusion_matrix(y_test, pred))

    results = pd.DataFrame(rows).set_index("model").round(3)
    print("\n", results.to_string())
    results.to_csv(REPORTS_DIR / f"classical_results_{name}.csv")

    # ROC curves for all models in one figure
    plt.figure(figsize=(6, 5))
    for model_name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={results.loc[model_name, 'roc_auc']})")
    plt.plot([0, 1], [0, 1], "k--", label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curves - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"roc_{name}.png", dpi=150)
    plt.close()

    # Save every model, plus the best one by ROC-AUC for the final pipeline
    for model_name, pipe in fitted.items():
        joblib.dump({"pipeline": pipe, "features": list(X.columns)},
                    MODELS_DIR / f"{name}_{model_name}.joblib")
    best = results["roc_auc"].idxmax()
    joblib.dump({"pipeline": fitted[best], "features": list(X.columns), "name": best},
                MODELS_DIR / f"{name}_best.joblib")
    print(f"\nbest by ROC-AUC: {best}")

    # Simple explainability: which features the decision tree relied on
    tree = fitted["DecisionTree"].named_steps["model"]
    imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("top decision-tree features:\n", imp.head(5).round(3).to_string())


def main():
    X, y = load_pima()
    run_dataset("pima", X, y, lambda: SMOTE(random_state=RANDOM_STATE), do_cv=True)

    X, y = load_brfss()
    run_dataset("brfss", X, y,
                lambda: RandomUnderSampler(random_state=RANDOM_STATE), do_cv=False)


if __name__ == "__main__":
    main()