"""Rule base. A fact is a tuple like ("diabetes_risk", "high").
Variables start with '?'. A condition ("not", pattern) means "pattern cannot be proven".
Profile facts: ("sex", "female"|"male"), ("age_group", "young"|"adult"|"older"),
("pregnant", "yes"|"no").  Educational decision support only, not medical advice."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    if_: tuple      # conditions
    then: tuple     # conclusion
    text: str       # plain-English meaning, used in explanations


def R(name, if_, then, text):
    return Rule(name, tuple(if_), then, text)


RULES = [
    # ---- retinopathy (from the CNN) ----
    R("R01", [("retinopathy", "severe")], ("refer", "urgent_eye_exam"),
      "severe or proliferative retinopathy needs urgent ophthalmology review"),
    R("R02", [("retinopathy", "moderate")], ("refer", "eye_exam_within_30_days"),
      "moderate retinopathy needs an eye exam soon"),
    R("R03", [("retinopathy", "mild")], ("monitor", "annual_eye_screening"),
      "mild retinopathy is monitored with yearly screening"),

    # ---- diabetes and complication risk (from the ML models and MLP) ----
    R("R04", [("diabetes_risk", "high")], ("test", "hba1c_and_fasting_glucose"),
      "high diabetes risk should be confirmed with lab tests"),
    R("R05", [("diabetes_risk", "high")], ("lifestyle", "dietitian_review"),
      "high diabetes risk benefits from a dietitian review"),
    R("R06", [("diabetes_risk", "medium")], ("lifestyle", "diet_and_activity_counselling"),
      "medium diabetes risk calls for lifestyle counselling"),
    R("R07", [("complication_risk", "high")], ("refer", "cardiology_checkup"),
      "high cardiovascular-complication risk needs a cardiology check"),
    R("R08", [("complication_risk", "high"), ("diabetes_risk", "high")], ("priority", "high"),
      "high diabetes risk together with high complication risk is high priority"),
    R("R09", [("retinopathy", "severe")], ("priority", "high"),
      "severe retinopathy is high priority"),

    # ---- glucose forecast (from the LSTM) ----
    R("R10", [("glucose_forecast", "hypo")], ("alert", "hypoglycemia_risk"),
      "forecast glucose below 70 mg/dL raises a hypoglycaemia alert"),
    R("R11", [("glucose_forecast", "hypo")], ("advise", "check_glucose_now"),
      "a forecast low means check glucose now"),
    R("R12", [("glucose_forecast", "hypo"), ("on_insulin", "yes")],
      ("advise", "take_fast_acting_carbs"),
      "a forecast low while on insulin means take fast-acting carbohydrate"),
    R("R13", [("glucose_forecast", "hyper")], ("alert", "hyperglycemia_risk"),
      "forecast glucose above 180 mg/dL raises a hyperglycaemia alert"),
    R("R14", [("glucose_forecast", "hyper")], ("advise", "drink_water_and_recheck"),
      "a forecast high means hydrate and recheck"),
    R("R15", [("glucose_forecast", "hyper"), ("on_insulin", "yes")],
      ("advise", "review_correction_dose_with_clinician"),
      "a forecast high while on insulin means review the correction dose with a clinician"),

    # ---- activity suggestions (these are checked by the safety module) ----
    R("R16", [("complication_risk", "low")], ("suggest", "intense_exercise"),
      "low complication risk allows intense exercise"),
    R("R17", [("complication_risk", "medium")], ("suggest", "moderate_exercise"),
      "medium complication risk suggests moderate exercise"),
    R("R18", [("complication_risk", "high")], ("suggest", "light_exercise"),
      "high complication risk suggests light exercise"),

    # ---- rules with a variable: anything referred or tested needs an appointment ----
    R("R19", [("refer", "?x")], ("needs_appointment", "?x"),
      "every referral becomes an appointment to schedule"),
    R("R20", [("test", "?x")], ("needs_appointment", "?x"),
      "every lab test becomes an appointment to schedule"),

    # ---- rule with negation as failure ----
    R("R21", [("diabetes_risk", "medium"), ("not", ("priority", "high"))],
      ("monitor", "recheck_in_6_months"),
      "medium risk that is not high priority is re-assessed in 6 months"),

    # ================= pregnancy =================
    R("R22", [("pregnant", "yes"), ("diabetes_risk", "high")],
      ("refer", "obstetric_diabetes_team"),
      "pregnancy with high diabetes risk needs a joint obstetric and diabetes team"),
    R("R23", [("pregnant", "yes")], ("advise", "check_all_medicines_for_pregnancy_safety"),
      "some diabetes, blood-pressure and cholesterol medicines are unsafe in pregnancy"),
    R("R24", [("pregnant", "yes")], ("advise", "use_pregnancy_specific_glucose_targets"),
      "pregnancy uses tighter glucose targets set with the care team"),
    R("R25", [("pregnant", "yes"), ("diabetes_risk", "high")],
      ("refer", "eye_exam_within_30_days"),
      "retinopathy can progress during pregnancy, so an eye exam is needed"),

    # ================= sex =================
    R("R26", [("sex", "female"), ("pregnant", "no"), ("age_group", "young"),
              ("diabetes_risk", "high")],
      ("refer", "preconception_counselling"),
      "women with diabetes and childbearing potential should have preconception counselling"),
    R("R27", [("sex", "male"), ("diabetes_risk", "high")],
      ("advise", "ask_about_sexual_health_low_libido_or_erectile_problems"),
      "men with diabetes should be asked about sexual health; low testosterone is checked if symptoms appear"),
    R("R28", [("sex", "female"), ("diabetes_risk", "high")],
      ("advise", "ask_about_sexual_health_concerns"),
      "women with diabetes should be asked about sexual health concerns"),

    # ================= age groups =================
    R("R29", [("age_group", "young"), ("diabetes_risk", "high")],
      ("advise", "start_lifestyle_changes_and_diabetes_education_early"),
      "young adults with high risk benefit from early lifestyle change and education"),
    R("R30", [("age_group", "adult"), ("diabetes_risk", "high")],
      ("advise", "discuss_cardiovascular_risk_prevention_with_clinician"),
      "adults from 40 with high diabetes risk should discuss cardiovascular prevention"),
    R("R31", [("age_group", "adult"), ("diabetes_risk", "high")],
      ("test", "kidney_and_cholesterol_check"),
      "adults from 40 with high diabetes risk need kidney and cholesterol checks"),
    R("R32", [("age_group", "older"), ("diabetes_risk", "high")],
      ("advise", "discuss_cardiovascular_risk_prevention_with_clinician"),
      "older adults with high diabetes risk should discuss cardiovascular prevention"),
    R("R33", [("age_group", "older"), ("diabetes_risk", "high")],
      ("test", "kidney_and_cholesterol_check"),
      "older adults with high diabetes risk need kidney and cholesterol checks"),
    R("R34", [("age_group", "older"), ("diabetes_risk", "high")],
      ("advise", "agree_individual_glucose_targets_that_avoid_low_glucose"),
      "older adults need individual glucose targets that avoid hypoglycaemia"),
    R("R35", [("age_group", "older"), ("diabetes_risk", "high"), ("on_insulin", "yes")],
      ("advise", "review_medicines_that_can_cause_low_glucose"),
      "older adults on insulin have a higher risk of hypoglycaemia, so medicines need review"),
    R("R36", [("age_group", "older"), ("diabetes_risk", "high"), ("on_insulin", "yes")],
      ("advise", "consider_continuous_glucose_monitoring"),
      "older adults on insulin may benefit from continuous glucose monitoring"),
    R("R37", [("age_group", "older")],
      ("advise", "add_balance_and_strength_exercises_to_prevent_falls"),
      "falls are a common problem in older adults with diabetes"),
    R("R38", [("age_group", "older"), ("diabetes_risk", "high")],
      ("refer", "geriatric_syndrome_screening"),
      "older adults with diabetes are screened yearly for falls, memory, mood and frailty"),
]