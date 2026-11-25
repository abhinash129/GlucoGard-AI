import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from src.config import MODELS_DIR, HYPO_THRESHOLD, HYPER_THRESHOLD
from src.dl.aptos_data import load_image
from src.rules.knowledge_base import RULES
from src.rules.engine import RuleEngine, show
from src.rules.safety import check_plan
from src.csp.scheduler import tasks_from_facts, plan_week, print_timetable

# probability -> level cut points (tunable choices, not clinical thresholds)
DIABETES_CUTS = (0.35, 0.65)
COMPLICATION_CUTS = (0.40, 0.60)
DR_LEVEL = {0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "severe"}
DR_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
AGE_GROUP_LABEL = {"young": "young adult, 18-39", "adult": "adult, 40-64",
                   "older": "older adult, 65+"}


def level(p, cuts):
    lo, hi = cuts
    return "low" if p < lo else ("medium" if p < hi else "high")


def age_group(age):
    return "young" if age < 40 else "adult" if age < 65 else "older"


def age_band(age):
    """BRFSS age category 1..13 (1 = 18-24, 2 = 25-29, ... 13 = 80+)."""
    return 1 if age < 25 else min(13, int((age - 25) // 5) + 2)


class GlucoGuard:
    def __init__(self):
        self.diabetes = joblib.load(MODELS_DIR / "pima_best.joblib")
        self.phenotype = joblib.load(MODELS_DIR / "pima_kmeans.joblib")
        self.mlp = keras.models.load_model(MODELS_DIR / "complication_mlp.keras", compile=False)
        self.mlp_meta = joblib.load(MODELS_DIR / "complication_scaler.joblib")
        self.cnn = keras.models.load_model(MODELS_DIR / "dr_cnn.keras", compile=False)
        self.lstm = keras.models.load_model(MODELS_DIR / "cgm_lstm.keras", compile=False)
        self.lstm_meta = joblib.load(MODELS_DIR / "cgm_lstm_meta.joblib")
        self.engine = RuleEngine(RULES)

    def forecast_batch(self, windows):
        """windows: [N, 12, 7] raw values -> forecast glucose [N, 6] in mg/dL."""
        m = self.lstm_meta
        w = np.asarray(windows, dtype=float)
        out = self.lstm.predict((w - m["mean"]) / m["std"], verbose=0)
        return w[:, -1, 0][:, None] + out * m["scale"]

    def assess(self, patient, start_weekday=0):
        """patient may contain: sex ('female'/'male'), age (years), pregnant (bool),
        clinical (dict of Pima features), survey (dict of BRFSS features),
        fundus_path, cgm_window (12x7 array), on_insulin (bool)."""
        facts, nums = set(), {}

        if patient.get("clinical") is not None:
            row = pd.DataFrame([patient["clinical"]])[self.diabetes["features"]]
            p = float(self.diabetes["pipeline"].predict_proba(row)[0, 1])
            nums["diabetes_probability"] = p
            facts.add(("diabetes_risk", level(p, DIABETES_CUTS)))
            ph = self.phenotype
            nums["phenotype_cluster"] = int(
                ph["kmeans"].predict(ph["prep"].transform(row[ph["features"]]))[0])

        if patient.get("survey") is not None:
            row = pd.DataFrame([patient["survey"]])[self.mlp_meta["features"]]
            score = float(self.mlp.predict(self.mlp_meta["scaler"].transform(row),
                                           verbose=0)[0, 0])
            nums["complication_score"] = score
            facts.add(("complication_risk", level(score, COMPLICATION_CUTS)))

        if patient.get("fundus_path") is not None:
            img = load_image(patient["fundus_path"]).astype("float32")[None]
            probs = self.cnn.predict(img, verbose=0)[0]
            grade = int(probs.argmax())
            nums["dr_grade"] = grade
            nums["dr_confidence"] = float(probs[grade])
            facts.add(("retinopathy", DR_LEVEL[grade]))

        if patient.get("cgm_window") is not None:
            fc = self.forecast_batch(np.asarray(patient["cgm_window"])[None])[0]
            nums["glucose_forecast"] = fc
            end = fc[-1]
            state = "hypo" if end < HYPO_THRESHOLD else "hyper" if end > HYPER_THRESHOLD else "normal"
            facts.add(("glucose_forecast", state))

        if "on_insulin" in patient:
            facts.add(("on_insulin", "yes" if patient["on_insulin"] else "no"))

        # --- patient profile: sex, age group, pregnancy ---
        sex, age = patient.get("sex"), patient.get("age")
        pregnant = bool(patient.get("pregnant")) if sex == "female" else False
        group = age_group(age) if age is not None else None
        if group:
            facts.add(("age_group", group))
        if sex in ("female", "male"):
            facts.add(("sex", sex))
            if sex == "female":
                facts.add(("pregnant", "yes" if pregnant else "no"))
        profile = {"sex": sex, "age": age, "age_group": group, "pregnant": pregnant}

        # --- reliability warnings ---
        warnings = []
        if patient.get("clinical") is not None:
            if sex == "male":
                warnings.append("The diabetes-risk model was trained only on women (Pima data), "
                                "so its estimate is less reliable for men.")
            if age is not None and not 21 <= age <= 81:
                warnings.append("Age is outside the range the diabetes-risk model was trained on (21-81).")
            if pregnant:
                warnings.append("The diabetes-risk estimate does not apply during pregnancy "
                                "(gestational diabetes is diagnosed differently).")
        if patient.get("survey") is not None and ("diabetes_risk", "low") in facts:
            warnings.append("The complication score was built for people who already have diabetes; "
                            "with low diabetes risk it is not meaningful.")
        if nums.get("dr_confidence", 1.0) < 0.5:
            warnings.append(f"Retinopathy grade has low confidence ({nums['dr_confidence']:.0%}); "
                            "ask a clinician to review the image.")

        # --- reasoning ---
        derived, _ = self.engine.forward_chain(facts)
        explain = {f: self.engine.why(f, facts) for f in derived
                   if f[0] in ("priority", "alert", "refer", "advise", "test")}
        plan = [f[1] for f in sorted(derived) if f[0] == "suggest"]
        safety = check_plan(facts, plan)

        # --- scheduling ---
        tasks = tasks_from_facts(derived, safety["final_plan"])
        solution, kept, dropped, _ = plan_week(tasks, start_weekday)
        return {"numbers": nums, "facts": facts, "derived": derived, "explain": explain,
                "plan": plan, "safety": safety, "tasks": kept, "dropped": dropped,
                "schedule": solution, "start_weekday": start_weekday,
                "profile": profile, "warnings": warnings}


def print_report(r):
    p = r["profile"]
    if p["sex"] or p["age"] is not None:
        print(f"\n-- PATIENT -- {p['sex']}, age {p['age']}"
              f" ({AGE_GROUP_LABEL.get(p['age_group'], '')})" + (", pregnant" if p["pregnant"] else ""))
    for w in r["warnings"]:
        print("  WARNING:", w)

    n = r["numbers"]
    print("\n-- MODEL OUTPUTS --")
    if "diabetes_probability" in n:
        print(f"  diabetes probability: {n['diabetes_probability']:.2f}"
              f" (phenotype cluster {n['phenotype_cluster']})")
    if "complication_score" in n:
        print(f"  complication score:   {n['complication_score']:.2f}")
    if "dr_grade" in n:
        print(f"  retinopathy:          grade {n['dr_grade']} ({DR_NAMES[n['dr_grade']]}),"
              f" confidence {n['dr_confidence']:.2f}")
    if "glucose_forecast" in n:
        steps = "  ".join(f"+{5 * (i + 1)}min {v:.0f}" for i, v in enumerate(n["glucose_forecast"]))
        print(f"  glucose forecast:     {steps}")

    print("\n-- FACTS GIVEN TO THE RULE ENGINE --")
    print("  " + ", ".join(sorted(show(f) for f in r["facts"])))

    print("\n-- RECOMMENDATIONS (forward chaining) --")
    for kind in ("priority", "alert", "advise", "refer", "test", "lifestyle", "monitor", "suggest"):
        for f in sorted(x for x in r["derived"] if x[0] == kind):
            print(f"  {show(f)}")

    print("\n-- WHY (backward chaining) --")
    for f, lines in r["explain"].items():
        print(f"  {show(f)}")
        print("\n".join("    " + l for l in lines[1:]) or "    (given)")

    s = r["safety"]
    print("\n-- SAFETY CHECK (resolution refutation) --")
    print("  proposed exercise:", r["plan"])
    for c in s["conflicts"]:
        print(f"  CONFLICT: {c['text']}  (refutation proof: {len(c['proof'])} steps)")
    print("  final plan:", s["final_plan"])

    print("\n-- WEEKLY SCHEDULE (CSP) --")
    print_timetable(r["schedule"], r["tasks"], r["start_weekday"])
    if r["dropped"]:
        print("  could not be fitted:", [t.label for t in r["dropped"]])