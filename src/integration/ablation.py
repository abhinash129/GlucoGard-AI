import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, FIGURES_DIR, HYPO_THRESHOLD, HYPER_THRESHOLD
from src.ml.preprocess import load_pima, load_brfss_complications, split
from src.dl.aptos_data import load_aptos
from src.dl.cgm_data import load_cgm
from src.integration.pipeline import (GlucoGuard, level, DIABETES_CUTS,
                                      COMPLICATION_CUTS, DR_LEVEL)
from src.rules.safety import check_plan
from src.csp.scheduler import tasks_from_facts, plan_week, build_csp, check_solution

N = 300
ACTION_KINDS = {"refer", "test", "alert", "advise", "priority"}
CRITICAL = [("refer", "urgent_eye_exam"), ("alert", "hypoglycemia_risk"),
            ("refer", "cardiology_checkup")]


def glucose_state(g):
    return "hypo" if g < HYPO_THRESHOLD else "hyper" if g > HYPER_THRESHOLD else "normal"


def score(variant, oracle):
    tp = fp = fn = 0
    for v, o in zip(variant, oracle):
        va = {f for f in v if f[0] in ACTION_KINDS}
        oa = {f for f in o if f[0] in ACTION_KINDS}
        tp += len(va & oa); fp += len(va - oa); fn += len(oa - va)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    crit = []
    for target in CRITICAL:
        need = [k for k, o in enumerate(oracle) if target in o]
        got = sum(target in variant[k] for k in need)
        crit.append(got / len(need) if need else float("nan"))
    return prec, rec, f1, crit


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    gg = GlucoGuard()

    # ---- untouched TEST pools (same splits as training) ----
    Xp, yp = load_pima();                   _, Xp_te, _, yp_te = split(Xp, yp)
    Xs, ys = load_brfss_complications();    _, Xs_te, _, ys_te = split(Xs, ys)
    X_img, y_img = load_aptos()
    _, te_img = train_test_split(np.arange(len(y_img)), test_size=0.2,
                                 stratify=y_img, random_state=RANDOM_STATE)
    Xc, Yc = load_cgm()["test"]

    # ---- 300 composite patients ----
    ip = rng.integers(0, len(Xp_te), N)
    isv = rng.integers(0, len(Xs_te), N)
    ii = te_img[rng.integers(0, len(te_img), N)]
    ic = rng.integers(0, len(Xc), N)
    insulin = rng.random(N) < 0.5

    # ---- model outputs ----
    p_d = gg.diabetes["pipeline"].predict_proba(
        Xp_te.iloc[ip][gg.diabetes["features"]])[:, 1]
    rows = Xs_te.iloc[isv][gg.mlp_meta["features"]]
    p_c = gg.mlp.predict(gg.mlp_meta["scaler"].transform(rows), verbose=0).ravel()
    grades = gg.cnn.predict(X_img[ii].astype("float32"), batch_size=32, verbose=0).argmax(axis=1)
    fc_lstm = gg.forecast_batch(Xc[ic])[:, -1]
    fc_flat = Xc[ic][:, -1, 0]                         # "glucose stays where it is"

    m_d = np.array([level(p, DIABETES_CUTS) for p in p_d])
    m_c = np.array([level(p, COMPLICATION_CUTS) for p in p_c])
    m_r = np.array([DR_LEVEL[int(g)] for g in grades])
    m_f = np.array([glucose_state(g) for g in fc_lstm])
    m_fp = np.array([glucose_state(g) for g in fc_flat])

    # ---- the "ideal" facts, built from the true labels ----
    t_d = np.where(yp_te.iloc[ip].to_numpy() == 1, "high", "low")
    t_c = np.where(ys_te.iloc[isv].to_numpy() == 1, "high", "low")
    t_r = np.array([DR_LEVEL[int(g)] for g in y_img[ii]])
    t_f = np.array([glucose_state(g) for g in Yc[ic][:, -1]])

    def facts_for(i, d, c, r, f):
        facts = {("on_insulin", "yes" if insulin[i] else "no")}
        for name, arr in (("diabetes_risk", d), ("complication_risk", c),
                          ("retinopathy", r), ("glucose_forecast", f)):
            if arr is not None:
                facts.add((name, str(arr[i])))
        return facts

    def derive(d, c, r, f):
        out = []
        for i in range(N):
            derived, _ = gg.engine.forward_chain(facts_for(i, d, c, r, f))
            out.append(derived)
        return out

    oracle = derive(t_d, t_c, t_r, t_f)

    # ================= 1. module ablation =================
    variants = {
        "Full system":         (m_d, m_c, m_r, m_f),
        "- diabetes model":    (None, m_c, m_r, m_f),
        "- MLP complication":  (m_d, None, m_r, m_f),
        "- CNN retinopathy":   (m_d, m_c, None, m_f),
        "- LSTM forecast":     (m_d, m_c, m_r, None),
        "LSTM -> flat guess":  (m_d, m_c, m_r, m_fp),
    }
    print(f"\nMODULE ABLATION ({N} composite test patients; ideal = rules on true labels)")
    print(f"{'variant':22s}{'prec':>7s}{'recall':>8s}{'F1':>7s}"
          f"{'urgent eye':>12s}{'hypo alert':>12s}{'cardiology':>12s}")
    n_crit = [sum(t in o for o in oracle) for t in CRITICAL]
    table, full_derived = [], None
    for name, (d, c, r, f) in variants.items():
        derived = derive(d, c, r, f)
        if name == "Full system":
            full_derived = derived
        prec, rec, f1, crit = score(derived, oracle)
        print(f"{name:22s}{prec:7.2f}{rec:8.2f}{f1:7.2f}"
              f"{crit[0]:12.2f}{crit[1]:12.2f}{crit[2]:12.2f}")
        table.append({"variant": name, "precision": prec, "recall": rec, "f1": f1,
                      "urgent_eye_recall": crit[0], "hypo_alert_recall": crit[1],
                      "cardiology_recall": crit[2]})
    print(f"(patients where the ideal plan contains: urgent eye exam {n_crit[0]}, "
          f"hypo alert {n_crit[1]}, cardiology {n_crit[2]})")
    pd.DataFrame(table).round(3).to_csv(FIGURES_DIR.parent / "ablation_modules.csv", index=False)

    # ================= 2. safety-check ablation =================
    unsafe_before = unsafe_after = 0
    final_plans = []
    for i, derived in enumerate(full_derived):
        facts = facts_for(i, m_d, m_c, m_r, m_f)
        plan = [f[1] for f in sorted(derived) if f[0] == "suggest"]
        res = check_plan(facts, plan)
        unsafe_before += bool(res["conflicts"])
        unsafe_after += bool(check_plan(facts, res["final_plan"])["conflicts"])
        final_plans.append(res["final_plan"])
    print("\nSAFETY CHECK ABLATION")
    print(f"  plans with an unsafe item WITHOUT the check: {unsafe_before}/{N}")
    print(f"  plans with an unsafe item WITH the check:    {unsafe_after}/{N}")

    # ================= 3. scheduler ablation =================
    start = datetime.date.today().weekday()
    multi = bad_naive = bad_csp = dropped_total = 0
    for derived, final_plan in zip(full_derived, final_plans):
        tasks = tasks_from_facts(derived, final_plan)
        if len(tasks) < 2:
            continue
        multi += 1
        csp = build_csp(tasks, start)
        used, naive = set(), {}
        for t in tasks:                                  # first free slot, ignores constraints
            for slot in csp.domains[t.id]:
                if slot not in used:
                    naive[t.id] = slot
                    used.add(slot)
                    break
        bad_naive += bool(check_solution(naive, tasks, start))
        sol, kept, dropped, _ = plan_week(tasks, start)
        bad_csp += bool(check_solution(sol, kept, start))
        dropped_total += len(dropped)
    print("\nSCHEDULER ABLATION (patients with 2 or more tasks)")
    print(f"  patients scheduled: {multi}")
    print(f"  naive first-free-slot schedules breaking a constraint: {bad_naive}/{multi}")
    print(f"  CSP schedules breaking a constraint:                   {bad_csp}/{multi}")
    print(f"  tasks the CSP had to drop as infeasible: {dropped_total}")


if __name__ == "__main__":
    main()