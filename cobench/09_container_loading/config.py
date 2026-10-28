"""Problem configuration: description, data loader, objective and feasibility check."""
import json

DESCRIPTION = 'Given a single container of dimensions (L, W, H) and N types of boxes, where each type i has dimensions (l_i, w_i, h_i), a total quantity Q_i, and orientation constraints, the goal is to determine the selection, orientation, and 3D coordinates of boxes to maximize space utilization. All indices start from 0. Let M be the total number of potential box instances (sum of Q_i).\n\n== DECISION VARIABLES (authoritative; model with exactly these) ==\n- x[j], y[j], z[j]: The coordinates of the front-bottom-left corner of box instance j in the container.  |  x[j] ≥ 0, y[j] ≥ 0, z[j] ≥ 0, continuous\n- s[j]: Binary indicator whether box instance j is selected to be placed in the container.  |  s[j] ∈ {0, 1}\n- o[j][k]: Binary indicator for the orientation of box instance j, where k represents one of the 6 possible rotation states (respecting vertical constraints).  |  o[j][k] ∈ {0, 1}\n\n== CONSTRAINTS (authoritative; ALL must be modeled) ==\n- [hard] Each selected box must lie entirely within the container boundary.  |  s[j] * (x[j] + \\sum_{k=0}^5 o[j][k] \\cdot l\'_{j,k}) \\le L, \\\\ s[j] * (y[j] + \\sum_{k=0}^5 o[j][k] \\cdot w\'_{j,k}) \\le W, \\\\ s[j] * (z[j] + \\sum_{k=0}^5 o[j][k] \\cdot h\'_{j,k}) \\le H, \\quad \\forall j \\in \\{0..M-1\\}\n- [hard] No two selected boxes may overlap in space.  |  \\forall j, i \\in \\{0..M-1\\}, j \\neq i: \\\\ (s[j] \\cdot s[i] = 1) \\implies (x[j] + dx_j \\le x[i]) \\lor (x[i] + dx_i \\le x[j]) \\lor \\\\ (y[j] + dy_j \\le y[i]) \\lor (y[i] + dy_i \\le y[j]) \\lor \\\\ (z[j] + dz_j \\le z[i]) \\lor (z[i] + dz_i \\le z[j]) \\\\ \\text{where } dx_j = \\sum_{k=0}^5 o[j][k] \\cdot l\'_{j,k} \\text{ (and similarly for } dy, dz\\text{)}\n- [hard] If a box is selected, exactly one valid rotation orientation must be chosen for it.  |  \\sum_{k=0}^5 o[j][k] = s[j], \\quad \\forall j \\in \\{0..M-1\\}\n- [hard] Box rotation must respect its vertical orientation constraint (i.e. certain axes must stay vertical).  |  o[j][k] = 0 \\quad \\forall k \\in \\text{InvalidOrientations}_j\n\n== DATA FIELD SEMANTICS (key meanings and axis conventions) ==\n- container_length: L; unit=mm\n- container_width: W; unit=mm\n- container_height: H; unit=mm\n- total_box_instances: M\n- boxes.volume: v[j]; unit=mm^3; axis: box instance\n- boxes.rotated_lengths: l\'_{j,k}; unit=mm; axis: box instance × rotation state\n- boxes.rotated_widths: w\'_{j,k}; unit=mm; axis: box instance × rotation state\n- boxes.rotated_heights: h\'_{j,k}; unit=mm; axis: box instance × rotation state\n- boxes.forbidden_orientations: InvalidOrientations_j; axis: box instance\n\n== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==\nSense: maximize\nFormula: max (sum_{j=0}^{M-1} s[j] * v[j]) / (L * W * H)\nPlain: The total volume of all successfully placed boxes divided by the total volume of the container.\nIf any earlier sentence conflicts with this section, this section wins.\n\n== IO SCHEMA ==\n{"container_length": float, "container_width": float, "container_height": float, "total_box_instances": int, "boxes.volume": [float, ...], "boxes.rotated_lengths": [[float, ...], ...], "boxes.rotated_widths": [[float, ...], ...], "boxes.rotated_heights": [[float, ...], ...], "boxes.forbidden_orientations": [[any, ...]|[int, ...], ...]}\n\nOUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):\n- "placement_status": list[int] — whether each box instance is placed into the container. example: [1, 1, 0, 1]. Lists the selection status of all M candidate box instances (1 = selected, 0 = not selected)\n- "coordinates": list[list[float]] — 3D coordinates of each box in the container (front-left-bottom corner). example: [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 40.0, 10.0]]. Lists the [x, y, z] coordinates of all M box instances; unselected boxes are filled with [0.0, 0.0, 0.0]\n- "orientations": list[int] — rotation orientation index (0-5) chosen for each box. example: [0, 2, -1, 5]. Lists the orientation index of all M box instances; unselected boxes may use -1 or 0\nFill every value for every entity; keep the keys EXACTLY as above.'
_DIRECTION = 'max'

def load_data(input_filename):
    with open(input_filename) as f:
        return json.load(f)

# --- generated compute_objective ---
def objective(instance, solution):
    L = instance['container_length']
    W = instance['container_width']
    H = instance['container_height']
    M = instance['total_box_instances']
    v = instance['boxes.volume']
    placement_status = solution['placement_status']
    
    total_placed_volume = 0.0
    for j in range(M):
        if placement_status[j] == 1:
            total_placed_volume += v[j]
            
    container_volume = L * W * H
    if container_volume == 0:
        return 0.0
    return total_placed_volume / container_volume
compute_objective = objective

# --- generated check_solution ---
def check_solution(instance, solution):
    violations = []
    EPS = 1e-5

    L = instance.get('container_length', 0.0)
    W = instance.get('container_width', 0.0)
    H = instance.get('container_height', 0.0)
    M = instance.get('total_box_instances', 0)
    
    v = instance.get('boxes.volume', [])
    l_rot = instance.get('boxes.rotated_lengths', [])
    w_rot = instance.get('boxes.rotated_widths', [])
    h_rot = instance.get('boxes.rotated_heights', [])
    forbidden = instance.get('boxes.forbidden_orientations', [])

    s = solution.get('placement_status', [])
    coords = solution.get('coordinates', [])
    orientations = solution.get('orientations', [])

    placed_boxes = []

    for j in range(M):
        if s[j] == 1:
            # Check orientation validity
            k = orientations[j]
            if k < 0 or k > 5:
                violations.append(f"Box {j} has invalid orientation index {k}")
                continue
            
            if k in (forbidden[j] if forbidden[j] is not None else []):
                violations.append(f"Box {j} uses forbidden orientation {k}")
            
            # Get dimensions
            dx = l_rot[j][k]
            dy = w_rot[j][k]
            dz = h_rot[j][k]
            
            x, y, z = coords[j]
            
            # Boundary check
            if x < -EPS or (x + dx) > L + EPS + EPS * L:
                violations.append(f"Box {j} exceeds container length: x={x}, dx={dx}, L={L}")
            if y < -EPS or (y + dy) > W + EPS + EPS * W:
                violations.append(f"Box {j} exceeds container width: y={y}, dy={dy}, W={W}")
            if z < -EPS or (z + dz) > H + EPS + EPS * H:
                violations.append(f"Box {j} exceeds container height: z={z}, dz={dz}, H={H}")
            
            # Overlap check
            for other in placed_boxes:
                oi, ox, oy, oz, odx, ody, odz = other
                # Two boxes do NOT overlap if they are separated in at least one dimension
                overlap_x = not (x + dx <= ox + EPS or ox + odx <= x + EPS)
                overlap_y = not (y + dy <= oy + EPS or oy + ody <= y + EPS)
                overlap_z = not (z + dz <= oz + EPS or oz + odz <= z + EPS)
                
                if overlap_x and overlap_y and overlap_z:
                    violations.append(f"Box {j} overlaps with box {oi}")
            
            placed_boxes.append((j, x, y, z, dx, dy, dz))

    # Objective consistency check
    total_placed_volume = sum(v[j] for j in range(M) if s[j] == 1)
    container_volume = L * W * H
    calculated_objective = total_placed_volume / container_volume if container_volume > 0 else 0.0
    
    claimed_objective = solution.get('objective_value')
    if claimed_objective is not None:
        if abs(claimed_objective - calculated_objective) > EPS + EPS * abs(calculated_objective):
            violations.append(f"Objective value mismatch: claimed {claimed_objective}, calculated {calculated_objective}")

    return violations

def objective(instance, solution):
    val = float(compute_objective(instance, solution))
    return -val if _DIRECTION == "max" else val

def is_feasible(instance, solution):
    v = check_solution(instance, solution)
    return (len(v) == 0, None if not v else "; ".join(str(x) for x in v))

HELPERS = []
TOOLS_DESCRIPTION = [{'name': 'is_feasible', 'input': 'solution: dict', 'output': '(bool, message)', 'purpose': 'Check feasibility; returns (True, None) iff the solution satisfies all constraints.'}, {'name': 'objective', 'input': 'solution: dict', 'output': 'float', 'purpose': 'Objective value (LOWER IS BETTER; max negated).'}]
