"""Generic CSP solver: backtracking search with optional MRV and forward checking."""
import time
from collections import Counter
from dataclasses import dataclass


@dataclass
class Stats:
    nodes: int = 0          # value assignments tried
    backtracks: int = 0     # assignments that had to be undone
    seconds: float = 0.0
    hit_limit: bool = False


class NodeLimit(Exception):
    pass


class CSP:
    """variables : list of ids
    domains    : {variable: [values]}
    compatible : f(a, va, b, vb) -> bool, the binary constraints (must be symmetric)
    day_of     : f(value) -> day; max_per_day is a global 'at most k per day' constraint"""

    def __init__(self, variables, domains, compatible, day_of, max_per_day):
        self.variables = list(variables)
        self.domains = domains
        self.compatible = compatible
        self.day_of = day_of
        self.max_per_day = max_per_day


def solve(csp, use_mrv=True, use_fc=True, node_limit=500_000):
    """Return (solution dict or None, Stats)."""
    stats = Stats()
    n = len(csp.variables)
    order = {v: i for i, v in enumerate(csp.variables)}
    t0 = time.perf_counter()

    def day_counts(assign):
        return Counter(csp.day_of(val) for val in assign.values())

    def consistent(var, val, assign):
        if day_counts(assign)[csp.day_of(val)] >= csp.max_per_day:
            return False
        return all(csp.compatible(var, val, o, ov) for o, ov in assign.items())

    def select_var(assign, domains):
        free = [v for v in csp.variables if v not in assign]
        if use_mrv:   # most constrained first; ties broken by the original (priority) order
            return min(free, key=lambda v: (len(domains[v]), order[v]))
        return free[0]

    def forward_check(assign, domains):
        """After an assignment, remove values from every unassigned variable
        that can no longer be chosen. Return None if some domain becomes empty."""
        counts = day_counts(assign)
        var, val = list(assign.items())[-1]          # the assignment just made
        new = {}
        for v in csp.variables:
            if v in assign:
                new[v] = domains[v]
                continue
            kept = [x for x in domains[v]
                    if counts[csp.day_of(x)] < csp.max_per_day
                    and csp.compatible(var, val, v, x)]
            if not kept:
                return None
            new[v] = kept
        return new

    def backtrack(assign, domains):
        if len(assign) == n:
            return dict(assign)
        var = select_var(assign, domains)
        for val in domains[var]:
            stats.nodes += 1
            if stats.nodes > node_limit:
                raise NodeLimit
            if not consistent(var, val, assign):
                continue
            assign[var] = val
            nd = forward_check(assign, domains) if use_fc else domains
            if nd is not None:
                result = backtrack(assign, nd)
                if result is not None:
                    return result
            del assign[var]
            stats.backtracks += 1
        return None

    try:
        solution = backtrack({}, {v: list(d) for v, d in csp.domains.items()})
    except NodeLimit:
        solution, stats.hit_limit = None, True
    stats.seconds = time.perf_counter() - t0
    return solution, stats