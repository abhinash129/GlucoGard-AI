from src.csp.solver import CSP, solve

NODE_LIMIT = 200_000


def make_problem(n_flex, n_narrow=2, max_per_day=2):
    """Flexible tasks (any weekday, 09-16h) are listed FIRST; two tight tasks that can
    only go on day 1 are listed LAST. Day 1 holds at most 2 tasks, so a search that
    places flexible tasks early fills day 1 and only discovers the trouble much later."""
    variables, domains = [], {}
    for i in range(n_flex):
        v = f"flex{i}"
        variables.append(v)
        domains[v] = [(d, h) for d in range(7) for h in range(9, 17)]
    for i in range(n_narrow):
        v = f"tight{i}"
        variables.append(v)
        domains[v] = [(0, 9 + i), (0, 10 + i)]
    return CSP(variables, domains, lambda a, va, b, vb: va != vb,
               lambda value: value[0], max_per_day)


def main():
    strategies = [("plain backtracking", False, False), ("+ MRV", True, False),
                  ("+ forward checking", False, True), ("+ FC + MRV", True, True)]
    for n_flex in (6, 8, 10):
        csp = make_problem(n_flex)
        print(f"\n{n_flex} flexible + 2 tight tasks")
        print(f"{'strategy':22s}{'nodes':>10s}{'backtracks':>12s}{'ms':>9s}  result")
        for name, mrv, fc in strategies:
            sol, s = solve(csp, use_mrv=mrv, use_fc=fc, node_limit=NODE_LIMIT)
            result = "solved" if sol else f"gave up at {NODE_LIMIT:,} nodes"
            print(f"{name:22s}{s.nodes:10d}{s.backtracks:12d}{s.seconds * 1000:9.1f}  {result}")


if __name__ == "__main__":
    main()