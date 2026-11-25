import itertools
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from src.config import MLDM_DIR, RANDOM_STATE
from src.ml.preprocess import load_brfss


import warnings
warnings.filterwarnings("ignore", module=r"(matplotlib|pyparsing).*")

RS = RANDOM_STATE
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)
RUN_FPGROWTH_ON_FULL_DATA = False      # set to False to skip the slowest step
RUN_ALGORITHM_COMPARISON = False      # results are already saved; set to True to rerun them


def heading(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# ------------------------------------------------------------------ transactions
def build_items(X, y):
    """One row per respondent, one True/False column per item (a 'basket' of items)."""
    d = pd.DataFrame(index=X.index)
    for c in ["HighBP", "HighChol", "Smoker", "Stroke", "HeartDiseaseorAttack", "HvyAlcoholConsump",
              "NoDocbcCost", "DiffWalk", "PhysActivity", "Fruits", "Veggies"]:
        d[c] = X[c] == 1
    bmi, g, a, inc, e = X["BMI"], X["GenHlth"], X["Age"], X["Income"], X["Education"]
    d["BMI_under25"] = bmi < 25
    d["BMI_25to30"] = (bmi >= 25) & (bmi < 30)
    d["BMI_30plus"] = bmi >= 30
    d["GenHlth_excellent_verygood"] = g <= 2
    d["GenHlth_good"] = g == 3
    d["GenHlth_fair_poor"] = g >= 4
    d["Age_18to44"] = a <= 5
    d["Age_45to64"] = (a >= 6) & (a <= 9)
    d["Age_65plus"] = a >= 10
    d["Income_low"] = inc <= 4
    d["Income_mid"] = (inc >= 5) & (inc <= 6)
    d["Income_high"] = inc >= 7
    d["Edu_highschool_or_less"] = e <= 4
    d["Edu_college"] = e >= 5
    d["PhysHlth_14plus_days"] = X["PhysHlth"] >= 14
    d["MentHlth_14plus_days"] = X["MentHlth"] >= 14
    d["Male"] = X["Sex"] == 1
    d["Female"] = X["Sex"] == 0
    d["Diabetes"] = y == 1
    return d.astype(bool)


# ------------------------------------------------------------------ Eclat (written from scratch)
def eclat(items, min_support, max_len):
    """Eclat: depth-first search on vertical data. Each item keeps the set of transactions that
    contain it (a bitset); the support of a longer itemset is the size of an intersection."""
    n = len(items)
    min_count = math.ceil(min_support * n - 1e-9)
    level1 = []
    for col in items.columns:
        tids = int.from_bytes(np.packbits(items[col].to_numpy()).tobytes(), "big")
        count = tids.bit_count()
        if count >= min_count:
            level1.append((col, tids, count))
    level1.sort(key=lambda x: x[2])
    found = {}

    def grow(prefix, candidates):
        for i, (item, tids, count) in enumerate(candidates):
            itemset = prefix + (item,)
            found[frozenset(itemset)] = count / n
            if len(itemset) < max_len:
                nxt = []
                for item2, tids2, _ in candidates[i + 1:]:
                    both = tids & tids2
                    c = both.bit_count()
                    if c >= min_count:
                        nxt.append((item2, both, c))
                if nxt:
                    grow(itemset, nxt)

    grow((), level1)
    return found


def to_dict(df):
    return {frozenset(i): s for i, s in zip(df["itemsets"], df["support"])}


def same(a, b):
    return a.keys() == b.keys() and max(abs(a[k] - b[k]) for k in a) < 1e-9


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def compare_algorithms(items, max_len=3):
    heading("APRIORI, FP-GROWTH AND ECLAT COMPARED (20,000-ROW SAMPLE, ITEMSETS UP TO 3 ITEMS)")
    sample = items.sample(20000, random_state=RS)
    rows = []
    for ms in (0.20, 0.10, 0.06, 0.04):
        ap, t_ap = timed(lambda: apriori(sample, min_support=ms, use_colnames=True, max_len=max_len,
                                         low_memory=True))
        fp, t_fp = timed(lambda: fpgrowth(sample, min_support=ms, use_colnames=True, max_len=max_len))
        ec, t_ec = timed(lambda: eclat(sample, ms, max_len))
        ok = same(to_dict(ap), to_dict(fp)) and same(to_dict(fp), ec)
        rows.append({"min support": ms, "frequent itemsets": len(ec), "Apriori (s)": t_ap,
                     "FP-growth (s)": t_fp, "Eclat (s)": t_ec, "identical results": ok})
        print(f"done: min support {ms}", flush=True)
    tbl = pd.DataFrame(rows)
    print("\n", tbl.round(3).to_string(index=False))
    tbl.round(4).to_csv(MLDM_DIR / "association_algorithm_comparison.csv", index=False)

    plt.figure(figsize=(6, 3.8))
    for col, mk in (("Apriori (s)", "o-"), ("FP-growth (s)", "s-"), ("Eclat (s)", "^-")):
        plt.plot(tbl["min support"], tbl[col], mk, label=col.replace(" (s)", ""))
    plt.yscale("log"); plt.xscale("log"); plt.gca().invert_xaxis()
    plt.xlabel("minimum support (log scale, lower = harder)"); plt.ylabel("seconds (log scale)")
    plt.title("Mining time on 20,000 baskets"); plt.legend()
    plt.tight_layout(); plt.savefig(MLDM_DIR / "association_runtime.png", dpi=150); plt.close()


# ------------------------------------------------------------------ rules
def metrics(s_ab, s_a, s_b):
    conf = s_ab / s_a
    return {"support": s_ab, "antecedent support": s_a, "consequent support": s_b, "confidence": conf,
            "lift": conf / s_b, "leverage": s_ab - s_a * s_b,
            "conviction": (1 - s_b) / (1 - conf) if conf < 1 else np.inf}


def diabetes_rules(freq):
    """Rules  A -> Diabetes, built from the frequent itemsets that contain Diabetes."""
    s_d = freq[frozenset(["Diabetes"])]
    rows = []
    for itemset, s in freq.items():
        if "Diabetes" in itemset and len(itemset) >= 2:
            ante = itemset - {"Diabetes"}
            rows.append({"antecedent": " + ".join(sorted(ante)), "size": len(ante),
                         **metrics(s, freq[ante], s_d)})
    df = pd.DataFrame(rows)
    # non-redundant: confidence must beat that of every simpler rule (one item removed) by at least 0.02
    conf = {frozenset(a.split(" + ")): c for a, c in zip(df["antecedent"], df["confidence"])}
    best_sub = []
    for a, sz in zip(df["antecedent"], df["size"]):
        items = a.split(" + ")
        subs = [conf[frozenset(items) - {x}] for x in items if len(items) > 1 and (frozenset(items) - {x}) in conf]
        best_sub.append(max(subs) if subs else s_d)
    df["improvement"] = df["confidence"] - np.array(best_sub)
    return df, s_d


def general_rules(freq, min_support=0.05, max_len=3, min_conf=0.6):
    sub = {k: v for k, v in freq.items() if v >= min_support and len(k) <= max_len}
    rows = []
    for itemset, s in sub.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset)
        for r in range(1, len(items)):
            for cons in itertools.combinations(items, r):
                cons = frozenset(cons)
                ante = itemset - cons
                m = metrics(s, sub[ante], sub[cons])
                if m["confidence"] >= min_conf:
                    rows.append({"antecedent": " + ".join(sorted(ante)), "consequent": " + ".join(sorted(cons)), **m})
    return pd.DataFrame(rows)


def show(df, cols, n, title):
    print(f"\n{title}")
    print(df[cols].head(n).round(3).to_string(index=False))


def main():
    X, y = load_brfss()
    items = build_items(X, y)
    heading("TRANSACTIONS")
    print(f"{items.shape[0]:,} baskets, {items.shape[1]} items, "
          f"{items.sum(axis=1).mean():.1f} items per basket on average")
    sup = items.mean().sort_values(ascending=False)
    sup.rename("support").round(4).to_csv(MLDM_DIR / "association_item_support.csv")
    print("most frequent items:\n", sup.head(5).round(3).to_string())
    print("rarest items:\n", sup.tail(5).round(3).to_string())

    if RUN_ALGORITHM_COMPARISON:
        compare_algorithms(items)

    heading("MINING THE FULL DATA (ECLAT, MIN SUPPORT 0.02, UP TO 4 ITEMS)")
    freq, t_ec = timed(lambda: eclat(items, 0.02, 4))
    print(f"{len(freq):,} frequent itemsets found by Eclat in {t_ec:.1f} s")
    if RUN_FPGROWTH_ON_FULL_DATA:
        try:
            fp, t_fp = timed(lambda: fpgrowth(items, min_support=0.02, use_colnames=True, max_len=4))
            print(f"FP-growth: {len(fp):,} itemsets in {t_fp:.1f} s; identical to Eclat: {same(to_dict(fp), freq)}")
        except Exception as e:                       # the check is optional; do not stop the study
            print("FP-growth on the full data was skipped:", e)

    heading("RULES THAT PREDICT DIABETES (SUPPORT AT LEAST 0.02)")
    rules, base = diabetes_rules(freq)
    print(f"baseline: {100 * base:.1f}% of respondents have diabetes, so a rule with lift 3 means "
          f"about {300 * base:.0f}% confidence")
    rules.sort_values("lift", ascending=False).round(5).to_csv(MLDM_DIR / "association_rules_diabetes.csv", index=False)
    cols = ["antecedent", "support", "confidence", "lift", "leverage", "conviction"]
    nonred = rules[rules["improvement"] >= 0.02].sort_values("lift", ascending=False)
    print(f"{len(rules):,} rules in total; {len(nonred):,} are non-redundant")
    show(nonred, cols, 12, "top non-redundant rules by lift:")
    show(rules[rules["size"] == 1].sort_values("lift", ascending=False), cols, 6,
         "best single-item rules:")
    show(rules[rules["support"] >= 0.05].sort_values("confidence", ascending=False), cols, 6,
         "highest confidence among rules with support at least 0.05:")
    show(rules.sort_values("lift"), cols, 8, "lowest lift (protective patterns, lift below 1):")

    heading("GENERAL RULES (ANY CONSEQUENT, SUPPORT 0.05, UP TO 3 ITEMS, CONFIDENCE 0.60 OR MORE)")
    gen = general_rules(freq).sort_values("lift", ascending=False)
    print(f"{len(gen):,} rules")
    gen.round(5).to_csv(MLDM_DIR / "association_rules_general.csv", index=False)
    show(gen, ["antecedent", "consequent", "support", "confidence", "lift", "conviction"], 10,
         "top rules by lift:")

    best = nonred.iloc[0]
    ante = best["antecedent"].split(" + ")
    has = items[ante].all(axis=1)
    both = (has & items["Diabetes"]).mean()
    print("\nDIRECT CHECK of the best rule against the data:")
    print(f"  {best['antecedent']} -> Diabetes")
    print(f"  support {both:.4f} (table {best['support']:.4f}), confidence {both / has.mean():.4f} "
          f"(table {best['confidence']:.4f}), lift {both / has.mean() / base:.3f} (table {best['lift']:.3f})")

    plt.figure(figsize=(6.5, 4.4))
    sc = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], s=10, cmap="viridis")
    plt.colorbar(sc, label="lift")
    plt.axhline(base, color="k", ls=":", lw=0.8)
    plt.xlabel("support"); plt.ylabel("confidence")
    plt.title("Rules predicting diabetes (dotted line: baseline rate)")
    plt.tight_layout(); plt.savefig(MLDM_DIR / "association_rules_scatter.png", dpi=150); plt.close()
    print(f"\nAll tables and figures were saved in {MLDM_DIR}")


if __name__ == "__main__":
    main()