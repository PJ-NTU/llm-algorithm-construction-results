"""Checker for the Set Partitioning problem.

check(instance, solution) returns a list of violation descriptions; an empty list means the solution is feasible.
Convention (see solution_form.json):
  solution = {"feasible": bool, "x": list[int] | None, "total_cost": float | None}
  feasible=false reports "no feasible solution"; in that case x and total_cost must be null.
  Such a report is not itself a feasible scheme selection, so check returns one explanatory violation.
"""
import math

TOL = 1e-6


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_instance(instance):
    errs = []
    if not isinstance(instance, dict):
        return ["instance: instance must be a dict"]
    for k in ("n", "m", "costs", "covers"):
        if k not in instance:
            errs.append(f"instance: missing field '{k}'")
    if errs:
        return errs
    n, m, costs, covers = instance["n"], instance["m"], instance["costs"], instance["covers"]
    if not _is_int(n) or n < 0:
        errs.append("instance: n must be a non-negative integer")
    if not _is_int(m) or m < 0:
        errs.append("instance: m must be a non-negative integer")
    if errs:
        return errs
    if not isinstance(costs, list) or len(costs) != n:
        errs.append("instance: costs must be a list of length n")
    else:
        for j, c in enumerate(costs):
            if not _is_num(c) or not math.isfinite(c) or c < 0:
                errs.append(f"instance: costs[{j}]={c!r} must be a finite non-negative real")
    if not isinstance(covers, list) or len(covers) != n:
        errs.append("instance: covers must be a list of length n")
    else:
        for j, cov in enumerate(covers):
            if not isinstance(cov, list):
                errs.append(f"instance: covers[{j}] must be a list")
                continue
            seen = set()
            for r in cov:
                if not _is_int(r) or r < 0 or r >= m:
                    errs.append(f"instance: covers[{j}] contains invalid task index {r!r} (must be in 0..{m - 1})")
                elif r in seen:
                    errs.append(f"instance: covers[{j}] contains duplicate task index {r}")
                else:
                    seen.add(r)
    return errs


def check(instance: dict, solution: dict) -> list:
    violations = _check_instance(instance)
    if violations:
        return violations
    n, m = instance["n"], instance["m"]
    costs, covers = instance["costs"], instance["covers"]

    # ---- Structural integrity ----
    if not isinstance(solution, dict):
        return ["structure: solution must be a dict"]
    for k in ("feasible", "x", "total_cost"):
        if k not in solution:
            violations.append(f"structure: missing field '{k}'")
    if violations:
        return violations
    feasible = solution["feasible"]
    x = solution["x"]
    total_cost = solution["total_cost"]
    if not isinstance(feasible, bool):
        return ["structure: feasible must be a bool"]

    if feasible is False:
        if x is not None:
            violations.append("structure: x must be null when feasible=false")
        if total_cost is not None:
            violations.append("structure: total_cost must be null when feasible=false")
        violations.append("infeasible_report: the solution reports no feasible solution and gives no scheme selection satisfying C1")
        return violations

    if not isinstance(x, list):
        return ["structure: x must be a list when feasible=true"]
    if len(x) != n:
        return [f"structure: x has length {len(x)}, expected n={n}"]
    bad = False
    for j, v in enumerate(x):
        if not isinstance(v, (int, float)) or isinstance(v, float) and not math.isfinite(v) or v not in (0, 1):
            violations.append(f"structure: x[{j}]={v!r} must be 0 or 1")
            bad = True
    if bad:
        return violations
    xs = [int(v) for v in x]

    # C1: each task r must be covered by exactly one selected scheme: sum_{j: r in covers[j]} x[j] == 1
    count = [0] * m
    for j in range(n):
        if xs[j] == 1:
            for r in covers[j]:
                count[r] += 1
    for r in range(m):
        if count[r] == 0:
            violations.append(f"C1: task {r} is not covered by any selected scheme")
        elif count[r] > 1:
            violations.append(f"C1: task {r} is covered {count[r]} times by selected schemes")

    # ---- total_cost must match the total cost recomputed from x (absolute tolerance 1e-6) ----
    if not _is_num(total_cost) or not math.isfinite(total_cost):
        violations.append("structure: total_cost must be a finite number when feasible=true")
    else:
        recomputed = math.fsum(costs[j] for j in range(n) if xs[j] == 1)
        if abs(total_cost - recomputed) > TOL:
            violations.append(
                f"structure: total_cost={total_cost} differs from the recomputed value {recomputed} by more than {TOL}")
    return violations
