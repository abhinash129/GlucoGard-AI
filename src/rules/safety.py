"""Plan safety check by propositional resolution refutation."""

# conflict name: (atoms that cause it, plan item it vetoes, description)
CONFLICTS = {
    "conflict_hypo_exercise": (
        ["hypo_forecast", "plan_intense_exercise"], "intense_exercise",
        "intense exercise while a glucose drop is forecast"),
    "conflict_retinopathy_exercise": (
        ["retinopathy_severe", "plan_intense_exercise"], "intense_exercise",
        "intense exercise with severe retinopathy (risk of retinal bleeding)"),
    "conflict_pregnancy_exercise": (
        ["pregnant", "plan_intense_exercise"], "intense_exercise",
        "intense exercise during pregnancy without obstetric clearance"),
    "conflict_older_exercise": (
        ["older_adult", "plan_intense_exercise"], "intense_exercise",
        "intense exercise for an older adult without clinician clearance"),
    "conflict_insulin_skip_meal": (
        ["on_insulin", "plan_skip_meal"], "skip_meal",
        "skipping a meal while on insulin"),
    "conflict_hypo_skip_meal": (
        ["hypo_forecast", "plan_skip_meal"], "skip_meal",
        "skipping a meal while a glucose drop is forecast"),
}
FALLBACK = {"intense_exercise": "light_exercise", "skip_meal": "regular_meal"}


def neg(lit):
    return lit[1:] if lit.startswith("~") else "~" + lit


def implies(conds, concl):
    """a AND b -> c   becomes the clause   ~a OR ~b OR c."""
    return frozenset([neg(c) for c in conds] + [concl])


KB_CLAUSES = [implies(conds, name) for name, (conds, _, _) in CONFLICTS.items()]


def _resolvents(c1, c2):
    out = []
    for lit in c1:
        if neg(lit) in c2:
            r = (c1 - {lit}) | (c2 - {neg(lit)})
            if not any(neg(x) in r for x in r):      # drop tautologies
                out.append(frozenset(r))
    return out


def _fmt(clause):
    return "(empty clause: contradiction)" if not clause else " OR ".join(sorted(clause))


def _trace(known):
    order, seen = [], set()

    def visit(c):
        if c in seen:
            return
        seen.add(c)
        if known[c] is not None:
            visit(known[c][0])
            visit(known[c][1])
        order.append(c)

    visit(frozenset())
    ids = {c: i + 1 for i, c in enumerate(order)}
    lines = []
    for c in order:
        if known[c] is None:
            lines.append(f"{ids[c]}. {_fmt(c)}   [given]")
        else:
            a, b = known[c]
            lines.append(f"{ids[c]}. {_fmt(c)}   [resolve {ids[a]} and {ids[b]}]")
    return lines


def refute(clauses, goal, max_clauses=5000):
    """Prove `goal` by adding its negation and deriving the empty clause."""
    known = {frozenset(c): None for c in clauses}
    known.setdefault(frozenset([neg(goal)]), None)
    changed = True
    while changed and len(known) < max_clauses:
        changed = False
        current = list(known)
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                for r in _resolvents(current[i], current[j]):
                    if r not in known:
                        known[r] = (current[i], current[j])
                        changed = True
                        if not r:
                            return True, _trace(known)
    return False, []


def patient_atoms(facts):
    atoms = set()
    if ("glucose_forecast", "hypo") in facts:
        atoms.add("hypo_forecast")
    if ("on_insulin", "yes") in facts:
        atoms.add("on_insulin")
    if ("retinopathy", "severe") in facts:
        atoms.add("retinopathy_severe")
    if ("pregnant", "yes") in facts:
        atoms.add("pregnant")
    if ("age_group", "older") in facts:
        atoms.add("older_adult")
    return atoms


def check_plan(facts, plan_items):
    """facts: the patient's input facts. plan_items: e.g. ["intense_exercise"].
    Returns which conflicts exist, with a resolution proof for each, and a corrected plan."""
    units = [frozenset([a]) for a in patient_atoms(facts) | {"plan_" + p for p in plan_items}]
    kb = KB_CLAUSES + units
    conflicts = []
    for name, (_, item, text) in CONFLICTS.items():
        proved, trace = refute(kb, name)
        if proved:
            conflicts.append({"name": name, "item": item, "text": text, "proof": trace})
    vetoed = {c["item"] for c in conflicts}
    final, seen = [], set()
    for p in plan_items:
        q = FALLBACK.get(p, p) if p in vetoed else p
        if q not in seen:
            final.append(q)
            seen.add(q)
    return {"safe": not conflicts, "conflicts": conflicts,
            "vetoed": sorted(vetoed), "final_plan": final}