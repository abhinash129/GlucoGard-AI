from src.rules.knowledge_base import RULES
from src.rules.engine import RuleEngine
from src.rules.safety import check_plan
from src.csp.solver import solve
from src.csp.scheduler import (Task, tasks_from_facts, build_csp, plan_week,
                               print_timetable, check_solution)

START_WEEKDAY = 0      # 0 = the planning week starts on a Monday

# The same patient as Scenario A in the rule-engine demo
FACTS = {("diabetes_risk", "high"), ("complication_risk", "low"), ("retinopathy", "severe"),
         ("glucose_forecast", "hypo"), ("on_insulin", "yes")}


def patient_tasks():
    derived, _ = RuleEngine(RULES).forward_chain(FACTS)
    plan = [f[1] for f in sorted(derived) if f[0] == "suggest"]
    final_plan = check_plan(FACTS, plan)["final_plan"]
    print("safe exercise plan from the rule engine:", final_plan)
    return tasks_from_facts(derived, final_plan)


def main():
    tasks = patient_tasks()
    print(f"\ntasks to schedule ({len(tasks)}):")
    for t in tasks:
        print(f"  [{t.priority}] {t.label}")

    print("\n" + "=" * 64 + "\nWEEKLY TIMETABLE\n" + "=" * 64)
    solution, kept, dropped, stats = plan_week(tasks, START_WEEKDAY)
    print_timetable(solution, kept, START_WEEKDAY)
    problems = check_solution(solution, kept, START_WEEKDAY)
    print("\nindependent constraint check:", "all constraints satisfied" if not problems else problems)

    print("\n" + "=" * 64 + "\nSEARCH STRATEGY COMPARISON (same problem)\n" + "=" * 64)
    print(f"{'strategy':28s}{'nodes':>9s}{'backtracks':>12s}{'ms':>9s}")
    csp = build_csp(tasks, START_WEEKDAY)
    for name, mrv, fc in [("plain backtracking", False, False), ("+ MRV", True, False),
                          ("+ forward checking", False, True), ("+ FC + MRV", True, True)]:
        sol, s = solve(csp, use_mrv=mrv, use_fc=fc)
        ok = "" if sol else "  (no solution / limit)"
        print(f"{name:28s}{s.nodes:9d}{s.backtracks:12d}{s.seconds * 1000:9.1f}{ok}")

    print("\n" + "=" * 64 + "\nOVER-CONSTRAINED WEEK (three urgent visits that must all be on day 1)\n" + "=" * 64)
    crowded = [Task(f"urgent{i}", f"Urgent review {i}", "clinic", "custom", i, (0,), (9, 10, 11))
               for i in (1, 2, 3)] + [t for t in tasks if t.kind == "lab"]
    sol, kept, dropped, _ = plan_week(crowded, START_WEEKDAY)
    print_timetable(sol, kept, START_WEEKDAY)
    print("dropped (could not be fitted):", [t.label for t in dropped])


if __name__ == "__main__":
    main()