# Library slot 6/10  id=alg_c924bb94002e
# score=-11995.367  feasible_rate=1.0
# score_vec=[-2843.52, -2434.98, -6775.889999999999, -6473.94, -2426.6899999999996, -6743.29, -6878.5999999999985, -15790.609999999999, -37148.36000000001, -32437.789999999997]

import random
import time


def solve(instance, time_limit_s):
    """
    Pack pieces with a randomized density-based recursive guillotine strategy.

    Every emitted cut is strictly inside its current region.  When a piece edge
    coincides with a region edge, the corresponding no-op cut is omitted.
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(time_limit_s))

    # Input Parsing
    L = instance["sheet_length"]
    W = instance["sheet_width"]
    N = instance["num_piece_types"]
    lengths = instance["piece_lengths"]
    widths = instance["piece_widths"]
    demands = instance["max_quantities"]
    values = instance["piece_values"]

    # Prepare piece data for the strategy
    pieces = []
    for i in range(N):
        pieces.append({
            "id": i,
            "l": lengths[i],
            "w": widths[i],
            "v": values[i],
            "d": demands[i],
            "density": values[i] / (lengths[i] * widths[i] + 1e-9),
        })

    # Sort pieces by density (the original core strategy)
    pieces.sort(key=lambda x: x["density"], reverse=True)

    def get_empty_sol():
        return {
            "selection_status": [[0] * demands[i] for i in range(N)],
            "placements": [[[-1.0, -1.0]] * demands[i] for i in range(N)],
            "guillotine_cut_tree": {"type": "empty"},
        }

    best_total_value = 0.0
    best_sol_dict = get_empty_sol()

    def pack_recursive(rect_l, rect_w, x0, y0, local_counts, sorted_pieces, noise):
        """Recursively build a packing and its matching guillotine cut tree."""
        if time.time() >= deadline or rect_l < 1e-6 or rect_w < 1e-6:
            return 0, [], {"type": "empty"}

        # A piece must really fit; no tolerance may enlarge the current region.
        candidates = []
        for p in sorted_pieces:
            if (
                local_counts[p["id"]] > 0
                and p["l"] <= rect_l
                and p["w"] <= rect_w
            ):
                candidates.append(p)

        if not candidates:
            return 0, [], {"type": "empty"}

        # Select a piece: top-k randomized greedy (the original core strategy)
        k = min(len(candidates), 1 + int(noise * 4))
        p = random.choice(candidates[:k])

        p_id = p["id"]
        # Instance index is total demand - remaining demand
        p_inst = demands[p_id] - local_counts[p_id]
        local_counts[p_id] -= 1

        cut_x = float(x0 + p["l"])
        cut_y = float(y0 + p["w"])
        region_right = float(x0 + rect_l)
        region_top = float(y0 + rect_w)
        has_right_region = x0 < cut_x < region_right
        has_top_region = y0 < cut_y < region_top

        # Decide split orientation
        split_vertical = random.random() < 0.5

        leaf = {"type": "leaf", "piece_index": [p_id, p_inst]}
        if split_vertical:
            # First isolate a left strip, then split the area above the piece.
            if has_top_region:
                v_top, pl_top, t_top = pack_recursive(
                    p["l"], rect_w - p["w"], x0, cut_y,
                    local_counts, sorted_pieces, noise,
                )
                left_branch = {
                    "type": "node",
                    "cut_direction": "horizontal",
                    "cut_coordinate": cut_y,
                    "bottom": leaf,
                    "top": t_top,
                }
            else:
                v_top, pl_top = 0, []
                left_branch = leaf

            if has_right_region:
                v_right, pl_right, t_right = pack_recursive(
                    rect_l - p["l"], rect_w, cut_x, y0,
                    local_counts, sorted_pieces, noise,
                )
                tree = {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": cut_x,
                    "left_child": left_branch,
                    "right_child": t_right,
                }
            else:
                v_right, pl_right = 0, []
                tree = left_branch

            total_v = p["v"] + v_top + v_right
            placements = [(p_id, p_inst, x0, y0)] + pl_top + pl_right
        else:
            # First isolate a bottom strip, then split the area right of the piece.
            if has_right_region:
                v_right, pl_right, t_right = pack_recursive(
                    rect_l - p["l"], p["w"], cut_x, y0,
                    local_counts, sorted_pieces, noise,
                )
                bottom_branch = {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": cut_x,
                    "left_child": leaf,
                    "right_child": t_right,
                }
            else:
                v_right, pl_right = 0, []
                bottom_branch = leaf

            if has_top_region:
                v_top, pl_top, t_top = pack_recursive(
                    rect_l, rect_w - p["w"], x0, cut_y,
                    local_counts, sorted_pieces, noise,
                )
                tree = {
                    "type": "node",
                    "cut_direction": "horizontal",
                    "cut_coordinate": cut_y,
                    "bottom": bottom_branch,
                    "top": t_top,
                }
            else:
                v_top, pl_top = 0, []
                tree = bottom_branch

            total_v = p["v"] + v_right + v_top
            placements = [(p_id, p_inst, x0, y0)] + pl_right + pl_top

        return total_v, placements, tree

    # Main search loop
    iteration = 0
    while time.time() < deadline - 0.2:
        noise = (iteration % 5) / 4.0

        current_pieces = list(pieces)
        if iteration > 0:
            if random.random() < 0.3:
                random.shuffle(current_pieces)
            else:
                for _ in range(int(N * 0.1) + 1):
                    idx1 = random.randint(0, N - 1)
                    idx2 = random.randint(0, N - 1)
                    current_pieces[idx1], current_pieces[idx2] = (
                        current_pieces[idx2], current_pieces[idx1]
                    )

        local_demands = list(demands)
        val, placements, tree = pack_recursive(
            L, W, 0.0, 0.0, local_demands, current_pieces, noise
        )

        if val > best_total_value:
            # Convert to target schema
            sel = [[0] * d for d in demands]
            coords = [[[-1.0, -1.0]] * d for d in demands]
            for p_id, p_inst, px, py in placements:
                sel[p_id][p_inst] = 1
                coords[p_id][p_inst] = [float(px), float(py)]

            if tree["type"] == "node":
                tree["type"] = "root"

            best_sol_dict = {
                "selection_status": sel,
                "placements": coords,
                "guillotine_cut_tree": tree,
            }
            best_total_value = val

        iteration += 1

    return best_sol_dict
