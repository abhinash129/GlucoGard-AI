from collections import Counter
from dataclasses import dataclass
from src.csp.solver import CSP, solve

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MAX_PER_DAY = 2
CLINIC_HOURS = tuple(range(9, 17))
EXERCISE_HOURS = (7, 17, 18)

# key: (label, kind, priority 1=most urgent, latest day index allowed, hours)
CATALOG = {
    "urgent_eye_exam": ("Urgent eye exam (ophthalmology)", "clinic", 1, 2, CLINIC_HOURS),
    "obstetric_diabetes_team": ("Obstetric + diabetes team visit", "clinic", 1, 3, CLINIC_HOURS),
    "eye_exam_within_30_days": ("Eye exam", "clinic", 2, 6, CLINIC_HOURS),
    "hba1c_and_fasting_glucose": ("HbA1c + fasting glucose test", "lab", 2, 4, (8, 9)),
    "cardiology_checkup": ("Cardiology check-up", "clinic", 3, 6, CLINIC_HOURS),
    "kidney_and_cholesterol_check": ("Kidney + cholesterol blood test", "lab", 3, 6, (8, 9, 10)),
    "dietitian_review": ("Dietitian review", "clinic", 4, 6, tuple(range(10, 17))),
    "preconception_counselling": ("Preconception counselling", "clinic", 4, 6, tuple(range(10, 17))),
    "geriatric_syndrome_screening": ("Geriatric health screening", "clinic", 4, 6, CLINIC_HOURS),
    "diet_and_activity_counselling": ("Diet and activity counselling", "clinic", 5, 6,
                                      tuple(range(10, 17))),
}
EXERCISE = {                 # plan item: (label, sessions per week)
    "light_exercise": ("Light exercise (walk)", 3),
    "moderate_exercise": ("Moderate exercise", 3),
    "intense_exercise": ("Intense exercise", 4),
}
PRECEDENCE = {("hba1c_and_fasting_glucose", "dietitian_review"),
              ("hba1c_and_fasting_glucose", "diet_and_activity_counselling")}


@dataclass(frozen=True)
class Task:
    id: str
    label: str
    kind: str            # "clinic", "lab" or "exercise"
    key: str
    priority: int
    days: tuple          # allowed day indices (0 = first day of the planning week)
    hours: tuple


def tasks_from_facts(facts, final_plan):
    """Turn derived rule-engine facts and the safe plan into schedulable tasks."""
    names = sorted({f[1] for f in facts if f[0] in ("needs_appointment", "lifestyle")})
    tasks = []
    for key in names:
        label, kind, prio, max_day, hours = CATALOG.get(
            key, (key.replace("_", " ").capitalize(), "clinic", 5, 6, CLINIC_HOURS))
        tasks.append(Task(key, label, kind, key, prio, tuple(range(min(max_day, 6) + 1)), hours))
    for item in final_plan:
        if item in EXERCISE:
            label, sessions = EXERCISE[item]
            for i in range(1, sessions + 1):
                tasks.append(Task(f"{item}#{i}", f"{label} #{i}", "exercise", item, 6,
                                  tuple(range(7)), EXERCISE_HOURS))
    return sorted(tasks, key=lambda t: t.priority)


def build_csp(tasks, start_weekday=0, max_per_day=MAX_PER_DAY):
    """start_weekday: weekday of day 0 (0 = Monday). Clinics and labs are closed at weekends."""
    by_id = {t.id: t for t in tasks}

    def open_day(t, d):
        return t.kind == "exercise" or (start_weekday + d) % 7 < 5

    domains = {t.id: [(d, h) for d in t.days if open_day(t, d) for h in t.hours]
               for t in tasks}

    def compatible(a, va, b, vb):
        if va == vb:                                    # one task per hourly slot
            return False
        ta, tb = by_id[a], by_id[b]
        da, db = va[0], vb[0]
        if ta.kind == "exercise" and tb.kind == "exercise" and abs(da - db) < 2:
            return False                                # rest day between sessions
        if {ta.kind, tb.kind} == {"exercise", "lab"} and da == db:
            return False                                # no exercise on the fasting-test day
        if (ta.key, tb.key) in PRECEDENCE and not da < db:
            return False                                # test before dietitian visit
        if (tb.key, ta.key) in PRECEDENCE and not db < da:
            return False
        return True

    return CSP([t.id for t in tasks], domains, compatible, lambda v: v[0], max_per_day)


def check_solution(solution, tasks, start_weekday=0, max_per_day=MAX_PER_DAY):
    """Independent re-check of a finished schedule. Returns a list of problems (empty = valid)."""
    csp = build_csp(tasks, start_weekday, max_per_day)
    problems = []
    for v, val in solution.items():
        if val not in csp.domains[v]:
            problems.append(f"{v}: slot outside its allowed domain")
    items = list(solution.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if not csp.compatible(items[i][0], items[i][1], items[j][0], items[j][1]):
                problems.append(f"{items[i][0]} conflicts with {items[j][0]}")
    for d, c in Counter(val[0] for val in solution.values()).items():
        if c > max_per_day:
            problems.append(f"day {d}: {c} tasks (max {max_per_day})")
    return problems


def plan_week(tasks, start_weekday=0, use_mrv=True, use_fc=True):
    """Solve; if the week is over-constrained, drop the least urgent task and retry.
    Returns (solution, kept tasks, dropped tasks, stats)."""
    active, dropped = list(tasks), []
    stats = None
    while active:
        solution, stats = solve(build_csp(active, start_weekday), use_mrv, use_fc)
        if solution is not None:
            return solution, active, dropped, stats
        worst = max(active, key=lambda t: (t.priority, active.index(t)))
        active.remove(worst)
        dropped.append(worst)
    return {}, [], dropped, stats


def print_timetable(solution, tasks, start_weekday=0):
    by_id = {t.id: t for t in tasks}
    for d in range(7):
        items = sorted((slot[1], by_id[v].label) for v, slot in solution.items() if slot[0] == d)
        day = DAY_NAMES[(start_weekday + d) % 7]
        text = "; ".join(f"{h:02d}:00 {label}" for h, label in items) if items else "free"
        print(f"  Day {d + 1} {day}: {text}")