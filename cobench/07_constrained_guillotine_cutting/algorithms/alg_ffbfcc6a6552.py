# Library slot 9/10  id=alg_ffbfcc6a6552
# score=-11779.699  feasible_rate=1.0
# score_vec=[-2816.96, -2434.98, -6775.889999999999, -6473.94, -2426.6899999999996, -6732.37, -6878.5999999999985, -15402.5, -35790.630000000005, -32064.429999999997]

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

    # Pre-sort piece types by value density
    piece_types = []
    for i in range(N):
        piece_types.append({
            "id": i,
            "l": lengths[i],
            "w": widths[i],
            "v": values[i],
            "d": demands[i]
        })
    piece_types.sort(key=lambda x: x["v"] / (x["l"] * x["w"] + 1e-9), reverse=True)

    def get_empty_solution():
        status = [[0] * demands[j] for j in range(N)]
        coords = [[[-1.0, -1.0]] * demands[j] for j in range(N)]
        return {
            "selection_status": status,
            "placements": coords,
            "guillotine_cut_tree": {"type": "empty"}
        }

    def recursive_packer(rect_l, rect_w, x0, y0, available_counts):
        """
        Greedy recursive guillotine packer.
        Returns (value, placements, cut_tree, updated_counts)
        """
        if time.time() >= deadline:
            return 0, [], {"type": "empty"}, available_counts

        # Try to find the best piece type that fits.  The strict arithmetic
        # comparison prevents a physically oversized piece from being accepted.
        best_p_idx = -1
        for idx, p in enumerate(piece_types):
            if available_counts[p["id"]] > 0 and p["l"] <= rect_l and p["w"] <= rect_w:
                best_p_idx = idx
                break

        if best_p_idx == -1:
            return 0, [], {"type": "empty"}, available_counts

        p = piece_types[best_p_idx]

        # Heuristic: Decide whether to cut or just place the piece
        # For simplicity in this greedy version, we place the piece and split the remaining area.
        # We can split either horizontally or vertically.

        # Option 1: Vertical cut first (split along L)
        # [Piece | Remainder L]
        # [Remainder W        ]

        # Placement of piece p at (x0, y0)
        p_id = p["id"]
        p_inst = demands[p_id] - available_counts[p_id]
        current_placement = [(p_id, p_inst, x0, y0)]
        new_counts = list(available_counts)
        new_counts[p_id] -= 1
        cut_x = x0 + p["l"]
        cut_y = y0 + p["w"]
        has_right_region = x0 < cut_x < x0 + rect_l
        has_top_region = y0 < cut_y < y0 + rect_w

        # Choice: Vertical split or horizontal split for the rest of the rectangle
        # We'll try to fit more pieces in the remaining two rectangles.

        # Vertical split:
        # Left part: (p['l'], rect_w) -> containing piece p at bottom and a top-rect
        # Right part: (rect_l - p['l'], rect_w)

        if has_top_region:
            v1, pl1, t1, counts1 = recursive_packer(
                p["l"], rect_w - p["w"], x0, y0 + p["w"], new_counts
            )
        else:
            v1, pl1, t1, counts1 = 0, [], {"type": "empty"}, new_counts

        if has_right_region:
            v2, pl2, t2, counts2 = recursive_packer(
                rect_l - p["l"], rect_w, x0 + p["l"], y0, counts1
            )
        else:
            v2, pl2, t2, counts2 = 0, [], {"type": "empty"}, counts1

        total_v = p["v"] + v1 + v2
        all_placements = current_placement + pl1 + pl2

        # A cut node is emitted only when it creates two non-zero regions.
        leaf_node = {"type": "leaf", "piece_index": [p_id, p_inst]}
        if has_top_region:
            left_branch = {
                "type": "node",
                "cut_direction": "horizontal",
                "cut_coordinate": float(y0 + p["w"]),
                "bottom": leaf_node,
                "top": t1
            }
        else:
            left_branch = leaf_node

        if has_right_region:
            root_node = {
                "type": "node",
                "cut_direction": "vertical",
                "cut_coordinate": float(x0 + p["l"]),
                "left_child": left_branch,
                "right_child": t2
            }
        else:
            root_node = left_branch

        return total_v, all_placements, root_node, counts2

    best_val = -1
    best_sol = get_empty_solution()

    # Initial Greedy Pass
    val, placements, tree, _ = recursive_packer(L, W, 0, 0, demands)

    def format_solution(val, placements, tree):
        status = [[0] * demands[j] for j in range(N)]
        coords = [[[-1.0, -1.0]] * demands[j] for j in range(N)]
        for p_id, p_inst, px, py in placements:
            status[p_id][p_inst] = 1
            coords[p_id][p_inst] = [float(px), float(py)]

        # Ensure root type
        if tree["type"] == "node":
            tree["type"] = "root"

        return {
            "selection_status": status,
            "placements": coords,
            "guillotine_cut_tree": tree,
            "objective_value": float(val)
        }

    if val > best_val:
        best_val = val
        best_sol = format_solution(val, placements, tree)

    # Iterated search with randomized greedy
    while time.time() < deadline - 0.2:
        # Perturb the piece priority slightly
        shuffled_types = list(piece_types)
        # Randomly swap a few elements or add noise to sorting
        if N > 0 and random.random() < 0.5:
            idx1 = random.randint(0, N-1)
            idx2 = random.randint(0, N-1)
            shuffled_types[idx1], shuffled_types[idx2] = shuffled_types[idx2], shuffled_types[idx1]

        # Randomized recursive packer
        def random_packer(rect_l, rect_w, x0, y0, available_counts, depth):
            if depth > 200 or time.time() >= deadline:  # Safety break
                return 0, [], {"type": "empty"}, available_counts

            fitting = [
                p for p in shuffled_types
                if available_counts[p["id"]] > 0
                and p["l"] <= rect_l
                and p["w"] <= rect_w
            ]

            if not fitting:
                return 0, [], {"type": "empty"}, available_counts

            # Pick from top-k instead of just the best
            k = min(3, len(fitting))
            p = random.choice(fitting[:k])

            p_id = p["id"]
            p_inst = demands[p_id] - available_counts[p_id]
            current_placement = [(p_id, p_inst, x0, y0)]
            new_counts = list(available_counts)
            new_counts[p_id] -= 1
            cut_x = x0 + p["l"]
            cut_y = y0 + p["w"]
            has_right_region = x0 < cut_x < x0 + rect_l
            has_top_region = y0 < cut_y < y0 + rect_w

            # Randomize split direction
            if random.random() < 0.5:
                if has_top_region:
                    v1, pl1, t1, counts1 = random_packer(
                        p["l"], rect_w - p["w"], x0, y0 + p["w"], new_counts, depth + 1
                    )
                else:
                    v1, pl1, t1, counts1 = 0, [], {"type": "empty"}, new_counts

                if has_right_region:
                    v2, pl2, t2, counts2 = random_packer(
                        rect_l - p["l"], rect_w, x0 + p["l"], y0, counts1, depth + 1
                    )
                else:
                    v2, pl2, t2, counts2 = 0, [], {"type": "empty"}, counts1

                total_v = p["v"] + v1 + v2
                all_placements = current_placement + pl1 + pl2

                leaf_node = {"type": "leaf", "piece_index": [p_id, p_inst]}
                if has_top_region:
                    left_branch = {
                        "type": "node",
                        "cut_direction": "horizontal",
                        "cut_coordinate": float(y0 + p["w"]),
                        "bottom": leaf_node,
                        "top": t1
                    }
                else:
                    left_branch = leaf_node

                if has_right_region:
                    root_node = {
                        "type": "node",
                        "cut_direction": "vertical",
                        "cut_coordinate": float(x0 + p["l"]),
                        "left_child": left_branch,
                        "right_child": t2
                    }
                else:
                    root_node = left_branch
            else:
                if has_right_region:
                    v1, pl1, t1, counts1 = random_packer(
                        rect_l - p["l"], p["w"], x0 + p["l"], y0, new_counts, depth + 1
                    )
                else:
                    v1, pl1, t1, counts1 = 0, [], {"type": "empty"}, new_counts

                if has_top_region:
                    v2, pl2, t2, counts2 = random_packer(
                        rect_l, rect_w - p["w"], x0, y0 + p["w"], counts1, depth + 1
                    )
                else:
                    v2, pl2, t2, counts2 = 0, [], {"type": "empty"}, counts1

                total_v = p["v"] + v1 + v2
                all_placements = current_placement + pl1 + pl2

                leaf_node = {"type": "leaf", "piece_index": [p_id, p_inst]}
                if has_right_region:
                    bottom_branch = {
                        "type": "node",
                        "cut_direction": "vertical",
                        "cut_coordinate": float(x0 + p["l"]),
                        "left_child": leaf_node,
                        "right_child": t1
                    }
                else:
                    bottom_branch = leaf_node

                if has_top_region:
                    root_node = {
                        "type": "node",
                        "cut_direction": "horizontal",
                        "cut_coordinate": float(y0 + p["w"]),
                        "bottom": bottom_branch,
                        "top": t2
                    }
                else:
                    root_node = bottom_branch

            return total_v, all_placements, root_node, counts2

        val, placements, tree, _ = random_packer(L, W, 0, 0, demands, 0)
        if val > best_val:
            best_val = val
            best_sol = format_solution(val, placements, tree)

    return best_sol
