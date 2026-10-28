"""Feasibility check for the multi-demand multidimensional knapsack problem (MDMKP).

check(instance, solution) -> list[str]: returns a list of violation descriptions; an empty list means the solution is feasible.
Constraint numbering follows MODEL.md:
  C1 resource capacity upper bound  sum_j A_leq[i][j]*x[j] <= b_leq[i]
  C2 demand lower bound            sum_j A_geq[k][j]*x[j] >= b_geq[k]
  C3 0-1 selection                 x[j] ∈ {0,1}
Convention: x being null (None) means the solver declares "no feasible solution". It is not a feasible selection,
and the checker cannot verify it against C1-C3, so a single violation message is returned.
"""


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _matrix_ok(mat, rows, cols, name, errs):
    if not isinstance(mat, list) or len(mat) != rows:
        errs.append(f"Instance structure error: {name} should be a list of {rows} rows")
        return False
    for r, row in enumerate(mat):
        if not isinstance(row, list) or len(row) != cols:
            errs.append(f"Instance structure error: {name}[{r}] should have length {cols}")
            return False
        if not all(_is_int(v) for v in row):
            errs.append(f"Instance structure error: {name}[{r}] contains non-integer elements")
            return False
    return True


def _vector_ok(vec, length, name, errs):
    if not isinstance(vec, list) or len(vec) != length:
        errs.append(f"Instance structure error: {name} should have length {length}")
        return False
    if not all(_is_int(v) for v in vec):
        errs.append(f"Instance structure error: {name} contains non-integer elements")
        return False
    return True


def check(instance: dict, solution: dict) -> list:
    errs = []

    # ---- instance structure ----
    if not isinstance(instance, dict):
        return ["Instance structure error: instance is not a dict"]
    for key in ("n", "m", "q", "cost_vector", "A_leq", "b_leq", "A_geq", "b_geq"):
        if key not in instance:
            errs.append(f"Instance structure error: missing field {key}")
    if errs:
        return errs
    n, m, q = instance["n"], instance["m"], instance["q"]
    if not (_is_int(n) and _is_int(m) and _is_int(q)) or n < 1 or m < 0 or q < 0:
        return ["Instance structure error: integers n>=1, m>=0, q>=0 required"]
    ok = _vector_ok(instance["cost_vector"], n, "cost_vector", errs)
    ok = _matrix_ok(instance["A_leq"], m, n, "A_leq", errs) and ok
    ok = _vector_ok(instance["b_leq"], m, "b_leq", errs) and ok
    ok = _matrix_ok(instance["A_geq"], q, n, "A_geq", errs) and ok
    ok = _vector_ok(instance["b_geq"], q, "b_geq", errs) and ok
    if not ok:
        return errs
    A_leq, b_leq = instance["A_leq"], instance["b_leq"]
    A_geq, b_geq = instance["A_geq"], instance["b_geq"]

    # ---- solution structural integrity ----
    if not isinstance(solution, dict):
        return ["Solution structure error: solution is not a dict"]
    if "x" not in solution:
        return ["Solution structure error: missing field x"]
    x = solution["x"]
    if x is None:
        return ["x is null: the solver declares no feasible solution and gives no selection satisfying C1-C3"]
    if not isinstance(x, list):
        return ["Solution structure error: x should be a list"]
    if len(x) != n:
        return [f"Solution structure error: x has length {len(x)}, should be n={n}"]

    # C3 each item is either selected whole (1) or not selected (0)
    for j, v in enumerate(x):
        if not (_is_int(v) and v in (0, 1)):
            errs.append(f"C3 violated: x[{j}]={v!r} is not the integer 0 or 1")
    if errs:
        return errs  # sums for C1/C2 are meaningless when values are invalid

    chosen = [j for j in range(n) if x[j] == 1]

    # C1 total consumption of each resource i does not exceed capacity b_leq[i]
    for i in range(m):
        lhs = sum(A_leq[i][j] for j in chosen)
        if lhs > b_leq[i]:
            errs.append(f"C1 violated: resource {i} total consumption {lhs} > capacity {b_leq[i]}")

    # C2 total contribution to each demand k is at least the demand b_geq[k]
    for k in range(q):
        lhs = sum(A_geq[k][j] for j in chosen)
        if lhs < b_geq[k]:
            errs.append(f"C2 violated: demand {k} total contribution {lhs} < demand {b_geq[k]}")

    return errs
