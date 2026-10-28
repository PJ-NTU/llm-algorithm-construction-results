# Library slot 7/10  id=alg_5169f4a6abe1
# score=-11974.579  feasible_rate=1.0
# score_vec=[-2856.7899999999995, -2434.98, -6775.889999999999, -6473.94, -2426.6899999999996, -6743.29, -6878.5999999999985, -15790.609999999999, -36829.05, -32535.95]

import time
import random
import copy


def solve(instance, time_limit_s):
    t0 = time.time()
    deadline = t0 + max(0.0, float(time_limit_s))

    L = instance["sheet_length"]
    W = instance["sheet_width"]
    N = instance["num_piece_types"]
    lengths = instance["piece_lengths"]
    widths = instance["piece_widths"]
    demands = instance["max_quantities"]
    values = instance["piece_values"]

    # Pre-calculate densities and prepare piece data
    pieces = []
    for i in range(N):
        pieces.append({
            "id": i,
            "l": lengths[i],
            "w": widths[i],
            "v": values[i],
            "d": demands[i],
            "density": values[i] / (lengths[i] * widths[i] + 1e-9)
        })

    # Sort pieces by density for the greedy base
    pieces.sort(key=lambda x: x["density"], reverse=True)

    def get_empty_sol():
        return {
            "selection_status": [[0] * demands[i] for i in range(N)],
            "placements": [[[-1.0, -1.0]] * demands[i] for i in range(N)],
            "guillotine_cut_tree": {"type": "empty"}
        }

    best_total_value = -1.0
    best_sol_dict = get_empty_sol()

    def pack_recursive(x0, y0, x1, y1, local_counts, depth, sorted_pieces, noise):
        """
        Recursive guillotine packer.
        Uses a randomized greedy choice to explore the search space.
        """
        rect_l = x1 - x0
        rect_w = y1 - y0
        if time.time() >= deadline or rect_l < 1e-6 or rect_w < 1e-6:
            return 0, [], {"type": "empty"}

        # Filter pieces using the same absolute coordinates that are returned.
        candidates = []
        for p in sorted_pieces:
            piece_right = x0 + p["l"]
            piece_top = y0 + p["w"]
            if (
                local_counts[p["id"]] > 0
                and piece_right <= x1
                and piece_top <= y1
            ):
                candidates.append(p)

        if not candidates:
            return 0, [], {"type": "empty"}

        # Select a piece: top-k randomized greedy
        k = min(len(candidates), 1 + int(noise * 4))
        p = random.choice(candidates[:k])

        p_id = p["id"]
        p_inst = demands[p_id] - local_counts[p_id]
        local_counts[p_id] -= 1

        piece_right = x0 + p["l"]
        piece_top = y0 + p["w"]
        can_cut_vertical = x0 < piece_right < x1
        can_cut_horizontal = y0 < piece_top < y1

        # Decide split orientation: vertical vs horizontal
        # We try to balance based on the remaining aspect ratio or just random
        # h_a style: vertical cut first vs h_b style
        split_vertical = random.random() < 0.5

        if split_vertical:
            # Cut vertically at the piece's right edge, then horizontally above it.
            if can_cut_horizontal:
                v_top, pl_top, t_top = pack_recursive(
                    x0, piece_top, piece_right, y1,
                    local_counts, depth + 1, sorted_pieces, noise
                )
            else:
                v_top, pl_top, t_top = 0, [], {"type": "empty"}

            if can_cut_vertical:
                v_right, pl_right, t_right = pack_recursive(
                    piece_right, y0, x1, y1,
                    local_counts, depth + 1, sorted_pieces, noise
                )
            else:
                v_right, pl_right, t_right = 0, [], {"type": "empty"}

            total_v = p["v"] + v_top + v_right
            placements = [(p_id, p_inst, x0, y0)] + pl_top + pl_right

            leaf = {"type": "leaf", "piece_index": [p_id, p_inst]}
            if can_cut_horizontal:
                left_branch = {
                    "type": "node",
                    "cut_direction": "horizontal",
                    "cut_coordinate": float(piece_top),
                    "bottom": leaf,
                    "top": t_top
                }
            else:
                left_branch = leaf

            if can_cut_vertical:
                tree = {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": float(piece_right),
                    "left_child": left_branch,
                    "right_child": t_right
                }
            else:
                tree = left_branch
        else:
            # Cut horizontally at the piece's top edge, then vertically beside it.
            if can_cut_vertical:
                v_right, pl_right, t_right = pack_recursive(
                    piece_right, y0, x1, piece_top,
                    local_counts, depth + 1, sorted_pieces, noise
                )
            else:
                v_right, pl_right, t_right = 0, [], {"type": "empty"}

            if can_cut_horizontal:
                v_top, pl_top, t_top = pack_recursive(
                    x0, piece_top, x1, y1,
                    local_counts, depth + 1, sorted_pieces, noise
                )
            else:
                v_top, pl_top, t_top = 0, [], {"type": "empty"}

            total_v = p["v"] + v_right + v_top
            placements = [(p_id, p_inst, x0, y0)] + pl_right + pl_top

            leaf = {"type": "leaf", "piece_index": [p_id, p_inst]}
            if can_cut_vertical:
                bottom_branch = {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": float(piece_right),
                    "left_child": leaf,
                    "right_child": t_right
                }
            else:
                bottom_branch = leaf

            if can_cut_horizontal:
                tree = {
                    "type": "node",
                    "cut_direction": "horizontal",
                    "cut_coordinate": float(piece_top),
                    "bottom": bottom_branch,
                    "top": t_top
                }
            else:
                tree = bottom_branch

        return total_v, placements, tree

    # Main search loop
    iteration = 0
    while time.time() < deadline - 0.1:
        # Vary the noise level to alternate between greedy and exploration
        noise = (iteration % 5) / 4.0

        # Perturb the piece priorities slightly
        current_pieces = list(pieces)
        if iteration > 0:
            # Random shuffle a small portion or swap
            if random.random() < 0.3:
                random.shuffle(current_pieces)
            else:
                for _ in range(int(N * 0.1) + 1):
                    idx1 = random.randint(0, N - 1)
                    idx2 = random.randint(0, N - 1)
                    current_pieces[idx1], current_pieces[idx2] = current_pieces[idx2], current_pieces[idx1]

        local_demands = list(demands)
        val, placements, tree = pack_recursive(
            0.0, 0.0, L, W,
            local_demands, 0, current_pieces, noise
        )

        if val > best_total_value:
            best_total_value = val

            # Convert to required output format
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
                "guillotine_cut_tree": tree
            }

        iteration += 1

    # Final check for empty tree case
    if best_sol_dict["guillotine_cut_tree"]["type"] == "node":
        best_sol_dict["guillotine_cut_tree"]["type"] = "root"

    return best_sol_dict
