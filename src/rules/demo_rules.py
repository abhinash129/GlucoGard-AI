from src.rules.knowledge_base import RULES
from src.rules.engine import RuleEngine, show
from src.rules.safety import check_plan

SHOW_ORDER = ["priority", "alert", "advise", "refer", "test", "monitor",
              "lifestyle", "suggest", "needs_appointment"]

SCENARIOS = [
    ("SCENARIO A: high diabetes risk, severe retinopathy, forecast low, on insulin",
     {("diabetes_risk", "high"), ("complication_risk", "low"), ("retinopathy", "severe"),
      ("glucose_forecast", "hypo"), ("on_insulin", "yes")},
     [("priority", "high"), ("advise", "take_fast_acting_carbs"),
      ("needs_appointment", "urgent_eye_exam")]),
    ("SCENARIO B: low-risk patient with a normal forecast",
     {("diabetes_risk", "low"), ("complication_risk", "low"), ("retinopathy", "none"),
      ("glucose_forecast", "normal"), ("on_insulin", "no")},
     [("alert", "hypoglycemia_risk"), ("suggest", "intense_exercise")]),
]


def run(title, facts, goals):
    engine = RuleEngine(RULES)
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)
    print("input facts:", ", ".join(sorted(show(f) for f in facts)))

    derived, why = engine.forward_chain(facts)
    print("\n-- FORWARD CHAINING: derived facts --")
    for pred in SHOW_ORDER:
        for f in sorted(x for x in derived if x[0] == pred):
            print(f"  {show(f):55s} <- {why[f][0]}")

    print("\n-- BACKWARD CHAINING: why? --")
    for g in goals:
        lines = engine.why(g, facts)
        print(f"\nWhy {show(g)}?")
        print("\n".join("  " + l for l in lines) if lines else "  no proof: it cannot be derived")

    plan = [f[1] for f in sorted(derived) if f[0] == "suggest"]
    result = check_plan(facts, plan)
    print("\n-- SAFETY CHECK (resolution refutation) --")
    print("proposed plan:", plan)
    print("safe:", result["safe"])
    for c in result["conflicts"]:
        print(f"\nCONFLICT {c['name']}: {c['text']}")
        print("\n".join("  " + l for l in c["proof"]))
    print("\nfinal plan:", result["final_plan"])


if __name__ == "__main__":
    for title, facts, goals in SCENARIOS:
        run(title, facts, goals)