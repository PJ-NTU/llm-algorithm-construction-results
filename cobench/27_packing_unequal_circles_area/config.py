"""Problem configuration: description, data loader, objective and feasibility check."""
import json

DESCRIPTION = 'The problem involves selecting a subset of unequal circles from a given set and placing them into a fixed circular container of radius R centered at (cx, cy) such that no two circles overlap and all selected circles are contained within the container\'s boundary. The goal is to maximize the total area of the packed circles. Indices for circles range from 0 to N-1.\n\n== DECISION VARIABLES (authoritative; model with exactly these) ==\n- alpha[i]: Binary indicator whether circle i is selected to be packed  |  alpha[i] ∈ {0, 1}\n- x[i]: The x-coordinate of the center of circle i  |  x[i] ∈ ℝ\n- y[i]: The y-coordinate of the center of circle i  |  y[i] ∈ ℝ\n\n== CONSTRAINTS (authoritative; ALL must be modeled) ==\n- [hard] Each packed circle must lie entirely within the circular container. If a circle is not packed, its center is forced to the container\'s center.  |  (x[i] - cx)^2 + (y[i] - cy)^2 <= (alpha[i] * (R - radii[i]))^2 + (1 - alpha[i]) * 0\n- [hard] Any two packed circles must not overlap with each other.  |  (x[i] - x[j])^2 + (y[i] - y[j])^2 >= (alpha[i] * alpha[j] * (radii[i] + radii[j]))^2\n- [hard] If a circle is not packed, its coordinates must be set to the container\'s center (cx, cy).  |  x[i] = alpha[i] * x[i] + (1 - alpha[i]) * cx, y[i] = alpha[i] * y[i] + (1 - alpha[i]) * cy\n\n== DATA FIELD SEMANTICS (key meanings and axis conventions) ==\n- n: N\n- cx: cx\n- cy: cy\n- R: R\n- radii: radii; axis: circle index\n\n== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==\nSense: maximize\nFormula: max ∑_{i=0}^{N-1} alpha[i] * π * radii[i]^2\nPlain: Maximize the total area of all circles successfully packed into the container.\nIf any earlier sentence conflicts with this section, this section wins.\n\n== IO SCHEMA ==\n{"n": int, "cx": float, "cy": float, "R": float, "radii": [float, ...]}\n\nOUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):\n- "packing_status": list[int] — Whether each circle is placed in the container. example: [1, 0, 1, 1]. List the selection status of all n circles; 1 = placed, 0 = not placed\n- "center_x_coordinates": list[float] — x-coordinate of each circle\'s center. example: [0.5, 0.0, -1.2, 2.1]. List the x-coordinates of all n circles; if not placed, fill in the container center coordinate cx\n- "center_y_coordinates": list[float] — y-coordinate of each circle\'s center. example: [0.5, 0.0, 1.5, -0.8]. List the y-coordinates of all n circles; if not placed, fill in the container center coordinate cy\nFill every value for every entity; keep the keys EXACTLY as above.'
_DIRECTION = 'max'

def load_data(input_filename):
    with open(input_filename) as f:
        return json.load(f)

# --- generated compute_objective ---
def objective(instance, solution):
    import math
    radii = instance['radii']
    packing_status = solution['packing_status']
    total_area = 0.0
    for i in range(len(radii)):
        if packing_status[i] == 1:
            total_area += math.pi * (radii[i] ** 2)
    return float(total_area)
compute_objective = objective

# --- generated check_solution ---
def check_solution(instance, solution):
    import math
    violations = []
    
    n = instance['n']
    cx = instance['cx']
    cy = instance['cy']
    R = instance['R']
    radii = instance['radii']
    
    packing_status = solution.get('packing_status', [])
    x_coords = solution.get('center_x_coordinates', [])
    y_coords = solution.get('center_y_coordinates', [])
    
    if len(packing_status) != n:
        violations.append(f"packing_status length {len(packing_status)} != {n}")
    if len(x_coords) != n:
        violations.append(f"center_x_coordinates length {len(x_coords)} != {n}")
    if len(y_coords) != n:
        violations.append(f"center_y_coordinates length {len(y_coords)} != {n}")
    
    if violations:
        return violations

    EPS = 1e-5

    for i in range(n):
        alpha = packing_status[i]
        xi = x_coords[i]
        yi = y_coords[i]
        ri = radii[i]
        
        if alpha not in [0, 1]:
            violations.append(f"Circle {i} packing_status {alpha} is not binary")
            continue

        # Constraint: If not packed, center must be (cx, cy)
        if alpha == 0:
            if abs(xi - cx) > EPS + EPS * abs(cx) or abs(yi - cy) > EPS + EPS * abs(cy):
                violations.append(f"Circle {i} not packed but center ({xi}, {yi}) != container center ({cx}, {cy})")
        
        # Constraint: Packed circle must lie entirely within the container
        if alpha == 1:
            dist_sq = (xi - cx)**2 + (yi - cy)**2
            max_dist = R - ri
            if max_dist < -EPS: # Radius larger than container
                violations.append(f"Circle {i} radius {ri} is larger than container radius {R}")
            else:
                max_dist_sq = max_dist**2
                if dist_sq > max_dist_sq + EPS + EPS * max_dist_sq:
                    violations.append(f"Circle {i} center ({xi}, {yi}) is too far from container center; exceeds boundary")

    # Constraint: No two packed circles overlap
    for i in range(n):
        if packing_status[i] == 0: continue
        for j in range(i + 1, n):
            if packing_status[j] == 0: continue
            
            dist_sq = (x_coords[i] - x_coords[j])**2 + (y_coords[i] - y_coords[j])**2
            min_dist = radii[i] + radii[j]
            min_dist_sq = min_dist**2
            
            if dist_sq < min_dist_sq - (EPS + EPS * min_dist_sq):
                violations.append(f"Circle {i} and {j} overlap: distance^2 {dist_sq} < required {min_dist_sq}")

    # Objective consistency check
    if solution.get('objective_value') is not None:
        recomputed_obj = sum(packing_status[i] * math.pi * (radii[i]**2) for i in range(n))
        claimed_obj = solution['objective_value']
        if abs(recomputed_obj - claimed_obj) > EPS + EPS * abs(claimed_obj):
            violations.append(f"Claimed objective {claimed_obj} != recomputed {recomputed_obj}")

    return violations

def objective(instance, solution):
    val = float(compute_objective(instance, solution))
    return -val if _DIRECTION == "max" else val

def is_feasible(instance, solution):
    v = check_solution(instance, solution)
    return (len(v) == 0, None if not v else "; ".join(str(x) for x in v))

HELPERS = []
TOOLS_DESCRIPTION = [{'name': 'is_feasible', 'input': 'solution: dict', 'output': '(bool, message)', 'purpose': 'Check feasibility; returns (True, None) iff the solution satisfies all constraints.'}, {'name': 'objective', 'input': 'solution: dict', 'output': 'float', 'purpose': 'Objective value (LOWER IS BETTER; max negated).'}]
