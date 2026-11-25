from itertools import count


# ---------- unification (the core of Prolog-style matching) ----------
def is_var(x):
    return isinstance(x, str) and x.startswith("?")


def walk(t, s):
    while is_var(t) and t in s:
        t = s[t]
    return t


def unify(a, b, s):
    """Make a and b equal by binding variables. Returns the new bindings or None."""
    if s is None:
        return None
    a, b = walk(a, s), walk(b, s)
    if a == b:
        return s
    if is_var(a):
        return {**s, a: b}
    if is_var(b):
        return {**s, b: a}
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        for x, y in zip(a, b):
            s = unify(x, y, s)
            if s is None:
                return None
        return s
    return None


def resolve(t, s):
    """Replace every bound variable inside t by its value."""
    t = walk(t, s)
    if isinstance(t, tuple):
        return tuple(resolve(x, s) for x in t)
    return t


def rename(t, tag):
    """Give a rule's variables fresh names so two uses of a rule never clash."""
    if is_var(t):
        return f"{t}_{tag}"
    if isinstance(t, tuple):
        return tuple(rename(x, tag) for x in t)
    return t


def show(t):
    return f"{t[0]}({', '.join(str(x) for x in t[1:])})"


class RuleEngine:
    def __init__(self, rules):
        self.rules = list(rules)
        self.by_name = {r.name: r for r in self.rules}
        self._tag = count(1)

    @staticmethod
    def _has_negation(rule):
        return any(c[0] == "not" for c in rule.if_)

    # ---------- forward chaining ----------
    def _match(self, conds, facts, s, used):
        """Yield every way to satisfy all conditions against the fact list."""
        if not conds:
            yield s, used
            return
        first, rest = conds[0], conds[1:]
        if first[0] == "not":                       # negation as failure
            if not any(unify(first[1], f, s) is not None for f in facts):
                yield from self._match(rest, facts, s, used)
            return
        for f in facts:
            s2 = unify(first, f, s)
            if s2 is not None:
                yield from self._match(rest, facts, s2, used + [f])

    def forward_chain(self, base_facts):
        """Apply rules until nothing new appears.
        Returns (all facts, why) where why[fact] = (rule name, supporting facts).
        Rules without 'not' run first; rules with 'not' run afterwards, so a
        negated condition is only tested once everything else is derived."""
        facts = set(base_facts)
        why = {f: ("given", []) for f in facts}
        strata = [[r for r in self.rules if not self._has_negation(r)],
                  [r for r in self.rules if self._has_negation(r)]]
        for stratum in strata:
            changed = True
            while changed:
                changed = False
                for rule in stratum:
                    snapshot = sorted(facts, key=str)
                    for s, used in self._match(rule.if_, snapshot, {}, []):
                        new = resolve(rule.then, s)
                        if new not in facts:
                            facts.add(new)
                            why[new] = (rule.name, used)
                            changed = True
        return facts, why

    # ---------- backward chaining ----------
    def _prove(self, goal, s, facts, depth):
        if depth > 12:
            return
        for f in facts:                              # 1) is the goal a known fact?
            s2 = unify(goal, f, s)
            if s2 is not None:
                yield s2, ("fact", resolve(goal, s2))
        for rule in self.rules:                      # 2) can a rule conclude it?
            tag = next(self._tag)
            s2 = unify(goal, rename(rule.then, tag), s)
            if s2 is None:
                continue
            conds = [rename(c, tag) for c in rule.if_]
            for s3, subs in self._prove_all(conds, s2, facts, depth + 1):
                yield s3, ("rule", rule.name, resolve(goal, s3), subs)

    def _prove_all(self, conds, s, facts, depth):
        if not conds:
            yield s, []
            return
        first, rest = conds[0], conds[1:]
        if first[0] == "not":
            if next(self._prove(first[1], s, facts, depth), None) is None:
                for s2, subs in self._prove_all(rest, s, facts, depth):
                    yield s2, [("naf", resolve(first[1], s))] + subs
            return
        for s2, p in self._prove(first, s, facts, depth):
            for s3, subs in self._prove_all(rest, s2, facts, depth):
                yield s3, [p] + subs

    def prove(self, goal, facts):
        return next(self._prove(goal, {}, list(facts), 0), None) is not None

    def why(self, goal, facts):
        """Return the explanation (list of text lines) for a goal, or None if unprovable."""
        found = next(self._prove(goal, {}, list(facts), 0), None)
        return None if found is None else self._format(found[1])

    def _format(self, proof, indent=0):
        pad = "  " * indent
        kind = proof[0]
        if kind == "fact":
            return [f"{pad}{show(proof[1])}   [given]"]
        if kind == "naf":
            return [f"{pad}not {show(proof[1])}   [cannot be proven]"]
        _, name, goal, subs = proof
        lines = [f"{pad}{show(goal)}   [{name}: {self.by_name[name].text}]"]
        for p in subs:
            lines += self._format(p, indent + 1)
        return lines