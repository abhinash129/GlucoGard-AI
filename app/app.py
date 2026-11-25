import sys
import tempfile
import datetime
from pathlib import Path

# make "import src..." work when Streamlit runs this file from the app folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
from src.config import HYPO_THRESHOLD, HYPER_THRESHOLD
from src.integration.pipeline import GlucoGuard, DR_NAMES, AGE_GROUP_LABEL, age_group, age_band
from src.integration.demo_patients import build_demo_patients
from src.rules.engine import show

st.set_page_config(page_title="GlucoGuard-AI", page_icon="🩺", layout="wide")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="stToolbar"] {display: none;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models ...")
def get_system():
    return GlucoGuard()


@st.cache_resource(show_spinner="Preparing example patients (first start takes 1-3 minutes) ...")
def get_demo():
    return build_demo_patients(get_system())


def pretty(f):
    return f[1].replace("_", " ")


def forecast_chart(window, forecast):
    hist = [float(v) for v in window[:, 0]]
    minutes = list(range(-55, 35, 5))                      # 12 past points + 6 future
    df = pd.DataFrame({
        "minutes": minutes,
        "history": hist + [np.nan] * 6,
        "forecast": [np.nan] * 11 + [hist[-1]] + [float(v) for v in forecast],
        "hypo limit (70)": HYPO_THRESHOLD,
        "hyper limit (180)": HYPER_THRESHOLD,
    }).set_index("minutes")
    st.line_chart(df)


def schedule_table(r):
    by_id = {t.id: t for t in r["tasks"]}
    today = datetime.date.today()
    rows = []
    for d in range(7):
        day = today + datetime.timedelta(days=d)
        items = sorted((slot[1], by_id[v].label)
                       for v, slot in r["schedule"].items() if slot[0] == d)
        rows.append({"Day": day.strftime("%a %d %b"),
                     "Plan": "; ".join(f"{h:02d}:00 {label}" for h, label in items) or "-"})
    return pd.DataFrame(rows)


def show_results(r, patient):
    n, facts = r["numbers"], dict(r["facts"])

    p = r["profile"]
    bits = []
    if p["sex"]:
        bits.append(p["sex"].capitalize())
    if p["age"] is not None:
        bits.append(f"{p['age']} years ({AGE_GROUP_LABEL[p['age_group']]})")
    if p["pregnant"]:
        bits.append("pregnant")
    if bits:
        st.markdown("**Patient:** " + ", ".join(bits))
    for w in r["warnings"]:
        st.warning(w)

    if n:
        st.subheader("Model outputs")
        cols = st.columns(4)
        if "diabetes_probability" in n:
            cols[0].metric("Diabetes probability", f"{n['diabetes_probability']:.0%}",
                           facts.get("diabetes_risk", "") + " risk", delta_color="off")
        if "complication_score" in n:
            cols[1].metric("Complication score", f"{n['complication_score']:.2f}",
                           facts.get("complication_risk", "") + " risk", delta_color="off")
            cols[1].caption("meaningful only for people who have diabetes")
        if "dr_grade" in n:
            cols[2].metric("Retinopathy", DR_NAMES[n["dr_grade"]],
                           f"confidence {n['dr_confidence']:.0%}", delta_color="off")
        if "glucose_forecast" in n:
            end = n["glucose_forecast"][-1]
            cols[3].metric("Glucose in 30 min", f"{end:.0f} mg/dL",
                           facts.get("glucose_forecast", ""), delta_color="off")
        if "glucose_forecast" in n and patient.get("cgm_window") is not None:
            forecast_chart(np.asarray(patient["cgm_window"]), n["glucose_forecast"])

    with st.expander("Facts given to the rule engine"):
        st.code(", ".join(sorted(show(f) for f in r["facts"])), language="text")

    st.subheader("Recommendations")
    derived = r["derived"]
    for f in sorted(x for x in derived if x[0] == "priority"):
        st.error(f"Priority: {f[1]}")
    for f in sorted(x for x in derived if x[0] == "alert"):
        st.error(f"Alert: {pretty(f)}")
    advice = sorted(x for x in derived if x[0] == "advise")
    if advice:
        st.info("**Advice**\n\n" + "\n".join(f"- {pretty(f)}" for f in advice))
    for f in sorted(x for x in derived if x[0] in ("refer", "test")):
        st.warning(f"{f[0].capitalize()}: {pretty(f)}")
    others = sorted(x for x in derived if x[0] in ("lifestyle", "monitor"))
    if others:
        st.markdown("\n".join(f"- {f[0].capitalize()}: {pretty(f)}" for f in others))
    if not any(f[0] in ("priority", "alert", "advise", "refer", "test", "lifestyle", "monitor")
               for f in derived):
        st.success("No special recommendations from the rule base.")

    if r["explain"]:
        st.subheader("Why? (backward chaining)")
        for f, lines in sorted(r["explain"].items()):
            with st.expander(f"Why {show(f)}?"):
                st.code("\n".join(lines), language="text")

    st.subheader("Safety check (resolution refutation)")
    s = r["safety"]
    if r["plan"]:
        st.write("Proposed exercise:", ", ".join(x.replace("_", " ") for x in r["plan"]))
    if s["conflicts"]:
        for c in s["conflicts"]:
            st.warning(f"Unsafe: {c['text']}")
            with st.expander("Show the refutation proof"):
                st.code("\n".join(c["proof"]), language="text")
        st.write("Corrected plan:", ", ".join(x.replace("_", " ") for x in s["final_plan"]))
    else:
        st.success("No safety conflicts found.")

    st.subheader("Weekly schedule (CSP)")
    st.dataframe(schedule_table(r), use_container_width=True, hide_index=True)
    if r["dropped"]:
        st.warning("Could not fit: " + ", ".join(t.label for t in r["dropped"]))


def manual_patient(gg):
    patient = {}

    c1, c2, c3, c4 = st.columns(4)
    sex = c1.selectbox("Sex", ["Female", "Male"]).lower()
    age = int(c2.number_input("Age (years)", 18, 100, 45))
    patient["sex"], patient["age"] = sex, age
    patient["on_insulin"] = c3.checkbox("Patient is on insulin")
    if sex == "female":
        patient["pregnant"] = c4.checkbox("Currently pregnant")
    st.caption(f"Age group: {AGE_GROUP_LABEL[age_group(age)]}")

    with st.expander("1. Clinical measurements (diabetes risk)", expanded=True):
        if st.checkbox("Include clinical measurements", value=True):
            st.caption("Enter 0 for glucose, blood pressure, skin thickness, insulin or BMI if unknown.")
            d1, d2, d3, d4 = st.columns(4)
            clin = {"Age": age}
            if sex == "female":
                clin["Pregnancies"] = d1.number_input("Previous pregnancies", 0, 20, 1)
            else:
                clin["Pregnancies"] = np.nan
                d1.caption("Pregnancy count is not used for men.")
            clin["Glucose"] = d2.number_input("Glucose (mg/dL)", 0, 300, 120)
            clin["BloodPressure"] = d3.number_input("Blood pressure (diastolic)", 0, 200, 70)
            clin["SkinThickness"] = d4.number_input("Skin thickness (mm)", 0, 100, 20)
            clin["Insulin"] = d1.number_input("Insulin (mu U/ml)", 0, 900, 80)
            clin["BMI"] = d2.number_input("BMI", 0.0, 70.0, 28.0)
            clin["DiabetesPedigreeFunction"] = d3.number_input("Pedigree function", 0.0, 3.0, 0.4)
            for k in ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"):
                if clin[k] == 0:
                    clin[k] = np.nan
            patient["clinical"] = clin

    with st.expander("2. Health survey (complication score, for people with diabetes)"):
        if st.checkbox("Include health survey"):
            yes_no = {"HighBP": "High blood pressure", "HighChol": "High cholesterol",
                      "CholCheck": "Cholesterol checked in last 5 years", "Smoker": "Smoked 100+ cigarettes",
                      "PhysActivity": "Physical activity in last 30 days", "Fruits": "Eats fruit daily",
                      "Veggies": "Eats vegetables daily", "HvyAlcoholConsump": "Heavy alcohol use",
                      "AnyHealthcare": "Has health coverage", "NoDocbcCost": "Skipped a doctor due to cost",
                      "DiffWalk": "Difficulty walking"}
            survey = {}
            cols = st.columns(3)
            for i, (key, label) in enumerate(yes_no.items()):
                survey[key] = int(cols[i % 3].checkbox(label, key="sv_" + key))
            s1, s2, s3 = st.columns(3)
            survey["BMI"] = s1.number_input("BMI ", 10.0, 70.0, 30.0, key="sv_bmi")
            survey["GenHlth"] = s2.slider("General health (1 excellent - 5 poor)", 1, 5, 3)
            survey["MentHlth"] = s3.slider("Bad mental-health days (of 30)", 0, 30, 0)
            survey["PhysHlth"] = s1.slider("Bad physical-health days (of 30)", 0, 30, 0)
            survey["Education"] = s2.slider("Education level (1-6)", 1, 6, 4)
            survey["Income"] = s3.slider("Income level (1-8)", 1, 8, 5)
            survey["Sex"] = 1 if sex == "male" else 0          # taken from the top of the form
            survey["Age"] = age_band(age)                      # BRFSS age category from the age above
            missing = set(gg.mlp_meta["features"]) - set(survey)
            if missing:
                st.error(f"Survey form is missing: {sorted(missing)}")
            else:
                patient["survey"] = survey

    with st.expander("3. Retinal (fundus) photo"):
        up = st.file_uploader("Upload a fundus image", type=["png", "jpg", "jpeg"])
        if up is not None:
            st.image(up, width=240)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix or ".png") as tmp:
                tmp.write(up.getvalue())
            patient["fundus_path"] = tmp.name

    with st.expander("4. Recent glucose readings (forecast)"):
        text = st.text_area("Last 12 CGM readings, oldest first, one every 5 minutes (mg/dL)", "")
        if text.strip():
            try:
                vals = [float(x) for x in text.replace("\n", ",").split(",") if x.strip()]
                if len(vals) != 12:
                    st.error(f"Need exactly 12 readings, got {len(vals)}.")
                else:
                    window = np.tile(gg.lstm_meta["mean"], (12, 1))   # other signals: training averages
                    window[:, 0] = vals
                    patient["cgm_window"] = window
            except ValueError:
                st.error("Readings must be numbers separated by commas.")
    return patient


def main():
    st.title("GlucoGuard-AI")
    st.caption("Diabetes decision support: risk models, rule-based advice with explanations, "
               "a safety check and a weekly care schedule. A student project, not medical advice.")
    gg = get_system()
    start = datetime.date.today().weekday()

    mode = st.sidebar.radio("Mode", ["Example patients", "Enter a patient manually"])
    with st.sidebar.expander("About this prototype"):
        st.write("Models were trained on public datasets (Pima, BRFSS, APTOS 2019, HUPA-UCM). "
                 "The diabetes-risk model comes from women only. The complication score is a proxy "
                 "built from self-reported stroke and heart disease. Risk levels use tunable cut "
                 "points, not clinical thresholds. Advice varies by sex, age group and pregnancy "
                 "and is meant to be discussed with a clinician.")

    if mode == "Example patients":
        demo = get_demo()
        if "removed" not in st.session_state:
            st.session_state.removed = set()
        names = [n for n in demo if n not in st.session_state.removed]

        if st.sidebar.button("Restore all patients"):
            st.session_state.removed = set()
            st.rerun()

        if not names:
            st.info("All example patients were removed. Use 'Restore all patients' in the sidebar.")
            return

        choice = st.selectbox("Choose an example patient", names)
        if st.sidebar.button("Remove this patient from the list"):
            st.session_state.removed.add(choice)
            st.rerun()

        patient = demo[choice]
        show_results(gg.assess(patient, start_weekday=start), patient)
    else:
        patient = manual_patient(gg)
        if st.button("Analyse", type="primary"):
            show_results(gg.assess(patient, start_weekday=start), patient)


main()