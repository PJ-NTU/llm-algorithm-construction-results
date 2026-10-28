"""Problem configuration: description, data loader, objective and feasibility check."""
import json

DESCRIPTION = 'The problem involves packing a subset of unequal circles into a fixed circular container of radius R centered at (cx, cy). Given N circles with radii R_i (sorted non-decreasingly), the goal is to select and position as many circles as possible such that each selected circle is entirely within the container and no two selected circles overlap. All indices start from 0.\n\n== DECISION VARIABLES (authoritative; model with exactly these) ==\n- alpha[i]: Binary indicator whether circle i is packed into the container  |  alpha[i] ∈ {0, 1}\n- x[i]: The x-coordinate of the center of circle i  |  x[i] ∈ ℝ\n- y[i]: The y-coordinate of the center of circle i  |  y[i] ∈ ℝ\n\n== CONSTRAINTS (authoritative; ALL must be modeled) ==\n- [hard] Each selected circle must lie entirely inside the container, i.e. the distance from the circle center to the container center plus the circle\'s radius must not exceed the container radius.  |  alpha[i] \\cdot \\sqrt{(x[i] - cx)^2 + (y[i] - cy)^2} \\le alpha[i] \\cdot (R - R_i), \\forall i \\in \\{0, \\dots, N-1\\}\n- [hard] No two selected circles may overlap, i.e. the distance between their centers must be greater than or equal to the sum of their radii.  |  \\sqrt{(x[i] - x[j])^2 + (y[i] - y[j])^2} \\ge alpha[i] \\cdot alpha[j] \\cdot (R_i + R_j), \\forall i, j \\in \\{0, \\dots, N-1\\}, i < j\n- [hard] Circles must be packed in order of increasing radius: if circle i is selected, all circles with smaller radius (smaller index) must also be selected.  |  alpha[i] \\ge alpha[i+1], \\forall i \\in \\{0, \\dots, N-2\\}\n\n== DATA FIELD SEMANTICS (key meanings and axis conventions) ==\n- n: N\n- cx: cx\n- cy: cy\n- R: R\n- radii: R_i; axis: circle_index\n\n== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==\nSense: maximize\nFormula: max \\sum_{i=0}^{N-1} alpha[i]\nPlain: Maximize the total number of circles successfully packed into the container.\nIf any earlier sentence conflicts with this section, this section wins.\n\n== IO SCHEMA ==\n{"n": int, "cx": float, "cy": float, "R": float, "radii": [float, ...]}\n\nOUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):\n- "packing_status": list[int] — Whether each circle is selected and packed into the container. example: [1, 1, 1, 0, 0]. List the selection status of all n circles (1 = selected, 0 = not selected)\n- "centers": list[list[float]] — Coordinates of each circle\'s center. example: [[0.5, 0.5], [1.2, -0.3], [-0.8, 0.1], [-1.0, -1.0], [-1.0, -1.0]]. List the (x, y) coordinates of all n circles; unselected circles default to (-1.0, -1.0)\nFill every value for every entity; keep the keys EXACTLY as above.'
_DIRECTION = 'max'

def load_data(input_filename):
    with open(input_filename) as f:
        return json.load(f)

# --- generated compute_objective ---
def objective(instance, solution):
    return float(sum(solution["packing_status"]))
compute_objective = objective

# --- generated check_solution ---
def check_solution(instance, solution):
    """Return a list of short violation strings; empty list means feasible."""
    violations = []
    if not solution:
        return ["No solution provided"]

    n = instance.get('n', 0)
    cx = instance.get('cx', 0.0)
    cy = instance.get('cy', 0.0)
    R = instance.get('R', 0.0)
    radii = instance.get('radii', [])

    packing_status = solution.get('packing_status', [])
    centers = solution.get('centers', [])

    if len(packing_status) != n:
        violations.append(f"packing_status length {len(packing_status)} != n {n}")
    if len(centers) != n:
        violations.append(f"centers length {len(centers)} != n {n}")

    EPS = 1e-5

    # Check each circle
    for i in range(n):
        alpha_i = packing_status[i] if i < len(packing_status) else 0
        if alpha_i is None: alpha_i = 0
        
        if alpha_i == 1:
            if i >= len(centers) or centers[i] is None or len(centers[i]) < 2:
                violations.append(f"Circle {i} selected but center missing or invalid")
                continue
            
            xi, yi = centers[i][0], centers[i][1]
            ri = radii[i]
            
            # Within container constraint
            dist_sq = (xi - cx)**2 + (yi - cy)**2
            dist = dist_sq**0.5
            limit = R - ri
            if dist > limit + EPS + EPS * abs(limit):
                violations.append(f"Circle {i} center ({xi}, {yi}) with radius {ri} exceeds container R={R} (dist={dist:.6f} > {limit:.6f})")

            # Overlap constraint
            for j in range(i + 1, n):
                alpha_j = packing_status[j] if j < len(packing_status) else 0
                if alpha_j == 1:
                    if j >= len(centers) or centers[j] is None or len(centers[j]) < 2:
                        continue
                    xj, yj = centers[j][0], centers[j][1]
                    rj = radii[j]
                    dist_ij_sq = (xi - xj)**2 + (yi - yj)**2
                    dist_ij = dist_ij_sq**0.5
                    min_dist = ri + rj
                    if dist_ij < min_dist - EPS - EPS * abs(min_dist):
                        violations.append(f"Circle {i} and {j} overlap: dist={dist_ij:.6f} < sum of radii={min_dist:.6f}")

    # Radius order constraint: alpha[i] >= alpha[i+1]
    for i in range(n - 1):
        a_i = packing_status[i] if packing_status[i] is not None else 0
        a_next = packing_status[i+1] if packing_status[i+1] is not None else 0
        if a_i < a_next:
            violations.append(f"Order violation: circle {i} (alpha={a_i}) not selected but circle {i+1} (alpha={a_next}) is")

    # Objective consistency check
    if solution.get('objective_value') is not None:
        claimed_obj = solution['objective_value']
        actual_obj = sum(1 for a in packing_status if a == 1)
        if abs(claimed_obj - actual_obj) > 1e-7:
            violations.append(f"Objective consistency mismatch: claimed {claimed_obj}, recomputed {actual_obj}")

    return violations

def objective(instance, solution):
    val = float(compute_objective(instance, solution))
    return -val if _DIRECTION == "max" else val

def is_feasible(instance, solution):
    v = check_solution(instance, solution)
    return (len(v) == 0, None if not v else "; ".join(str(x) for x in v))

HELPERS = []
TOOLS_DESCRIPTION = [{'name': 'is_feasible', 'input': 'solution: dict', 'output': '(bool, message)', 'purpose': 'Check feasibility; returns (True, None) iff the solution satisfies all constraints.'}, {'name': 'objective', 'input': 'solution: dict', 'output': 'float', 'purpose': 'Objective value (LOWER IS BETTER; max negated).'}]
