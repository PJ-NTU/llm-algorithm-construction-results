"""Problem configuration: description, data loader, objective and feasibility check."""
import json

DESCRIPTION = 'The problem involves placing a subset of rectangular pieces from N available types onto a single rectangular stock of dimensions (L, W). Each piece type i has dimensions (l_i, w_i), a value v_i, and usage bounds [min_i, max_i]. Pieces can be rotated 90 degrees and must not overlap or exceed stock boundaries. The goal is to maximize the total value of placed pieces. All indices start from 0. Let K be the maximum possible number of pieces that could fit (an upper bound on total items).\n\n== DECISION VARIABLES (authoritative; model with exactly these) ==\n- x[k]: Binary variable indicating if the k-th placement slot is used  |  x[k] ∈ {0,1}\n- t[k][i]: Binary variable indicating if the k-th placement slot contains piece type i  |  t[k][i] ∈ {0,1}\n- pos_x[k], pos_y[k]: Continuous coordinates of the bottom-left corner of the piece in slot k  |  pos_x[k] ≥ 0, pos_y[k] ≥ 0\n- r[k]: Binary variable indicating if the piece in slot k is rotated 90 degrees  |  r[k] ∈ {0,1}\n\n== CONSTRAINTS (authoritative; ALL must be modeled) ==\n- [hard] Each placement slot can be associated with at most one piece type, and only when that slot is activated.  |  ∑_{i=0}^{N-1} t[k][i] = x[k], ∀k\n- [hard] Pieces must lie entirely within the stock sheet boundary (accounting for rotation).  |  pos_x[k] + ∑_{i=0}^{N-1} t[k][i] * (l_i * (1 - r[k]) + w_i * r[k]) ≤ L, ∀k\npos_y[k] + ∑_{i=0}^{N-1} t[k][i] * (w_i * (1 - r[k]) + l_i * r[k]) ≤ W, ∀k\n- [hard] No two placed pieces may overlap.  |  Let W_k = ∑_{i=0}^{N-1} t[k][i] * (l_i * (1 - r[k]) + w_i * r[k])\nLet H_k = ∑_{i=0}^{N-1} t[k][i] * (w_i * (1 - r[k]) + l_i * r[k])\nFor k < m: \npos_x[k] + W_k ≤ pos_x[m] + M(3 - x[k] - x[m] - b1_{km})\npos_x[m] + W_m ≤ pos_x[k] + M(3 - x[k] - x[m] - b2_{km})\npos_y[k] + H_k ≤ pos_y[m] + M(3 - x[k] - x[m] - b3_{km})\npos_y[m] + H_m ≤ pos_y[k] + M(3 - x[k] - x[m] - b4_{km})\nb1_{km} + b2_{km} + b3_{km} + b4_{km} ≥ 1\n- [hard] The number of pieces used of each type must be within the specified minimum and maximum.  |  min_q_i ≤ ∑_{k=0}^{K-1} t[k][i] ≤ max_q_i, ∀i\n\n== DATA FIELD SEMANTICS (key meanings and axis conventions) ==\n- stock_length: L; unit=mm\n- stock_width: W; unit=mm\n- piece_values: v_i; axis: piece_type\n- piece_lengths: l_i; unit=mm; axis: piece_type\n- piece_widths: w_i; unit=mm; axis: piece_type\n- min_usage: min_q_i; unit=pcs; axis: piece_type\n- max_usage: max_q_i; unit=pcs; axis: piece_type\n- max_slots: K\n- num_piece_types: N\n- big_m: M\n\n== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==\nSense: maximize\nFormula: max ∑_{k=0}^{K-1} ∑_{i=0}^{N-1} (v_i * t[k][i])\nPlain: Maximize the sum of values of all pieces successfully placed on the stock.\nIf any earlier sentence conflicts with this section, this section wins.\n\n== IO SCHEMA ==\n{"stock_length": float, "stock_width": float, "num_piece_types": int, "max_slots": int, "piece_lengths": [float, ...], "piece_widths": [float, ...], "piece_values": [float, ...], "min_usage": [int, ...], "max_usage": [int, ...], "big_m": float}\n\nOUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):\n- "placements": list[dict] — status of each potential placement slot (whether used, piece type, coordinates and rotation). example: [{"is_used": 1, "piece_type_id": 0, "x": 0.0, "y": 0.0, "is_rotated": 0}, {"is_used": 1, "piece_type_id": 1, "x": 50.5, "y": 20.0, "is_rotated": 1}, {"is_used": 0, "piece_type_id": -1, "x": 0.0, "y": 0.0, "is_rotated": 0}]. Lists all K potential placement slots; for unused slots set is_used to 0 and piece_type_id to -1\nFill every value for every entity; keep the keys EXACTLY as above.'
_DIRECTION = 'max'

def load_data(input_filename):
    with open(input_filename) as f:
        return json.load(f)

# --- generated compute_objective ---
def objective(instance, solution):
    piece_values = instance['piece_values']
    placements = solution['placements']
    
    total_value = 0.0
    for placement in placements:
        if placement['is_used'] == 1:
            piece_type_id = placement['piece_type_id']
            if 0 <= piece_type_id < len(piece_values):
                total_value += float(piece_values[piece_type_id])
                
    return total_value
compute_objective = objective

# --- generated check_solution ---
def check_solution(instance, solution):
    violations = []
    EPS = 1e-5

    stock_length = instance.get('stock_length', 0.0)
    stock_width = instance.get('stock_width', 0.0)
    num_piece_types = instance.get('num_piece_types', 0)
    max_slots = instance.get('max_slots', 0)
    piece_lengths = instance.get('piece_lengths', [])
    piece_widths = instance.get('piece_widths', [])
    piece_values = instance.get('piece_values', [])
    min_usage = instance.get('min_usage', [])
    max_usage = instance.get('max_usage', [])

    placements = solution.get('placements', [])
    if len(placements) != max_slots:
        violations.append(f"Expected {max_slots} placements, found {len(placements)}")

    usage_counts = [0] * num_piece_types
    active_placements = []

    for k, p in enumerate(placements):
        is_used = int(round(p.get('is_used', 0)))
        piece_type_id = p.get('piece_type_id', -1)
        
        if is_used == 1:
            if piece_type_id < 0 or piece_type_id >= num_piece_types:
                violations.append(f"Slot {k} is used but has invalid piece_type_id {piece_type_id}")
                continue
            
            usage_counts[piece_type_id] += 1
            
            x = p.get('x', 0.0)
            y = p.get('y', 0.0)
            is_rotated = int(round(p.get('is_rotated', 0)))
            
            l_i = piece_lengths[piece_type_id]
            w_i = piece_widths[piece_type_id]
            
            # Calculate actual dimensions based on rotation
            if is_rotated == 1:
                w_k, h_k = w_i, l_i
            else:
                w_k, h_k = l_i, w_i
            
            # Boundary check
            if x < -EPS or x + w_k > stock_length + EPS + EPS * stock_length:
                violations.append(f"Slot {k} piece exceeds stock length: x={x}, width={w_k}, stock={stock_length}")
            if y < -EPS or y + h_k > stock_width + EPS + EPS * stock_width:
                violations.append(f"Slot {k} piece exceeds stock width: y={y}, height={h_k}, stock={stock_width}")
            
            active_placements.append({'k': k, 'x': x, 'y': y, 'w': w_k, 'h': h_k})
        else:
            if piece_type_id != -1:
                violations.append(f"Slot {k} is not used but piece_type_id is not -1")

    # Overlap check
    for i in range(len(active_placements)):
        for j in range(i + 1, len(active_placements)):
            p1 = active_placements[i]
            p2 = active_placements[j]
            
            # Check if rectangles (x1, y1, w1, h1) and (x2, y2, w2, h2) overlap
            # They do NOT overlap if one is to the left, right, above, or below the other
            no_overlap = (p1['x'] + p1['w'] <= p2['x'] + EPS + EPS * max(p1['x']+p1['w'], p2['x'])) or \
                         (p2['x'] + p2['w'] <= p1['x'] + EPS + EPS * max(p2['x']+p2['w'], p1['x'])) or \
                         (p1['y'] + p1['h'] <= p2['y'] + EPS + EPS * max(p1['y']+p1['h'], p2['y'])) or \
                         (p2['y'] + p2['h'] <= p1['y'] + EPS + EPS * max(p2['y']+p2['h'], p1['y']))
            
            if not no_overlap:
                violations.append(f"Overlap between slot {p1['k']} and slot {p2['k']}")

    # Usage bounds check
    for i in range(num_piece_types):
        if usage_counts[i] < min_usage[i]:
            violations.append(f"Piece type {i} usage {usage_counts[i]} below min {min_usage[i]}")
        if usage_counts[i] > max_usage[i]:
            violations.append(f"Piece type {i} usage {usage_counts[i]} above max {max_usage[i]}")

    # Objective consistency check
    recomputed_objective = sum(piece_values[p.get('piece_type_id')] 
                               for p in placements 
                               if int(round(p.get('is_used', 0))) == 1 
                               and 0 <= p.get('piece_type_id', -1) < num_piece_types)
    
    claimed_objective = solution.get('objective_value')
    if claimed_objective is not None:
        if abs(claimed_objective - recomputed_objective) > EPS + EPS * abs(recomputed_objective):
            violations.append(f"Claimed objective {claimed_objective} does not match recomputed {recomputed_objective}")

    return violations

def objective(instance, solution):
    val = float(compute_objective(instance, solution))
    return -val if _DIRECTION == "max" else val

def is_feasible(instance, solution):
    v = check_solution(instance, solution)
    return (len(v) == 0, None if not v else "; ".join(str(x) for x in v))

HELPERS = []
TOOLS_DESCRIPTION = [{'name': 'is_feasible', 'input': 'solution: dict', 'output': '(bool, message)', 'purpose': 'Check feasibility; returns (True, None) iff the solution satisfies all constraints.'}, {'name': 'objective', 'input': 'solution: dict', 'output': 'float', 'purpose': 'Objective value (LOWER IS BETTER; max negated).'}]
