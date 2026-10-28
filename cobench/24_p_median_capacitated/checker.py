"""Capacitated p-median location problem: feasibility check.

check(instance, solution) returns a list of violation descriptions; an empty list means the solution is feasible.
Solution format: {"status": "feasible"|"infeasible", "medians": list[int], "assignments": list[int]}
"""


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def check(instance: dict, solution: dict) -> list:
    violations = []

    # ---------- Basic instance reading ----------
    try:
        n = instance["n"]
        p = instance["p"]
        capacity = instance["capacity"]
        demands = instance["demands"]
        dist = instance["distance_matrix"]
    except (KeyError, TypeError) as e:
        return ["Instance is missing required fields or is malformed: %r" % (e,)]
    if not (_is_int(n) and _is_int(p) and _is_int(capacity)):
        return ["Instance fields n / p / capacity must be integers"]
    if not isinstance(demands, list) or len(demands) != n:
        return ["Instance field demands must be a list of length n"]
    if not isinstance(dist, list) or len(dist) != n or any(
        not isinstance(row, list) or len(row) != n for row in dist
    ):
        return ["Instance field distance_matrix must be an n×n matrix"]

    # ---------- Structural completeness of the solution ----------
    if not isinstance(solution, dict):
        return ["Solution must be a dict"]
    for key in ("status", "medians", "assignments"):
        if key not in solution:
            violations.append("Solution is missing field '%s'" % key)
    if violations:
        return violations

    status = solution["status"]
    medians = solution["medians"]
    assignments = solution["assignments"]

    if status not in ("feasible", "infeasible"):
        return ["status must be 'feasible' or 'infeasible', got %r" % (status,)]
    if not isinstance(medians, list):
        violations.append("medians must be a list")
    if not isinstance(assignments, list):
        violations.append("assignments must be a list")
    if violations:
        return violations

    if status == "infeasible":
        violations.append("status is 'infeasible': no feasible plan satisfying all constraints was given")
        if medians != []:
            violations.append("medians must be an empty list when status is 'infeasible'")
        if assignments != []:
            violations.append("assignments must be an empty list when status is 'infeasible'")
        return violations

    # medians elements: integers, in range, no duplicates
    median_ok = True
    for k, j in enumerate(medians):
        if not _is_int(j):
            violations.append("medians[%d] is not an integer: %r" % (k, j))
            median_ok = False
        elif not (0 <= j < n):
            violations.append("medians[%d]=%d is outside the index range 0..%d" % (k, j, n - 1))
            median_ok = False
    if median_ok and len(set(medians)) != len(medians):
        violations.append("medians contains duplicate indices")
    median_set = set(j for j in medians if _is_int(j) and 0 <= j < n)

    # assignments shape: length n
    assign_len_ok = len(assignments) == n
    if not assign_len_ok:
        violations.append("assignments should have length n=%d, got %d" % (n, len(assignments)))

    # C1 exactly p medians selected: the number of distinct valid medians must equal p
    if len(median_set) != p or len(medians) != p:
        violations.append(
            "C1 violated: exactly p=%d distinct medians should be selected, got %d listed, %d valid and distinct"
            % (p, len(medians), len(median_set))
        )

    # C2 each customer assigned exactly once: assignments must cover all n customers, each mapped to a valid median index
    valid_assign = {}
    for i, j in enumerate(assignments):
        if i >= n:
            break
        if not _is_int(j):
            violations.append("C2 violated: assignment value of customer %d is not an integer: %r" % (i, j))
        elif not (0 <= j < n):
            violations.append("C2 violated: customer %d is assigned to index %d outside the range 0..%d" % (i, j, n - 1))
        else:
            valid_assign[i] = j
    if not assign_len_ok:
        missing = n - len(assignments)
        if missing > 0:
            violations.append("C2 violated: %d customers are not assigned (customers %d..%d)" % (missing, len(assignments), n - 1))
        else:
            violations.append("C2 violated: assignments has %d extra entries; there are only %d customers" % (-missing, n))

    # C3 assign only to selected medians: assignments[i] must belong to medians
    for i, j in valid_assign.items():
        if j not in median_set:
            violations.append("C3 violated: customer %d is assigned to unselected median %d" % (i, j))

    # C4 median capacity limit: total demand served by each selected median <= capacity
    load = {}
    for i, j in valid_assign.items():
        load[j] = load.get(j, 0) + demands[i]
    for j in sorted(load):
        if j in median_set and load[j] > capacity:
            violations.append(
                "C4 violated: median %d serves demand %s, exceeding capacity %s" % (j, load[j], capacity)
            )

    return violations
