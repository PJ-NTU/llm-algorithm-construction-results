# Library slot 2/10  id=alg_c8cff3a01885
# score=-12112.226  feasible_rate=1.0
# score_vec=[-2928.5699999999997, -2434.98, -6591.209999999999, -6473.94, -2426.6899999999996, -6661.34, -6878.5999999999985, -16165.789999999999, -37378.94000000001, -33182.2]

import time
import random
import math


def solve(instance, time_limit_s):
    """
    SA-based guillotine packing using piece-order swaps and recursive splits.
    """
    t0 = time.time()

    # 1. Input Parsing
    L = instance["sheet_length"]
    W = instance["sheet_width"]
    N_types = instance["num_piece_types"]
    lengths_types = instance["piece_lengths"]
    widths_types = instance["piece_widths"]
    demands_types = instance["max_quantities"]
    values_types = instance["piece_values"]

    # Flatten piece types into individual instances for the SA logic
    flat_pieces = []
    for i in range(N_types):
        for j in range(demands_types[i]):
            flat_pieces.append({
                "type_idx": i,
                "inst_idx": j,
                "l": lengths_types[i],
                "w": widths_types[i],
                "v": values_types[i]
            })

    N = len(flat_pieces)

    def pack_guillotine(order, L_stock, W_stock):
        """
        Core logic: recursive guillotine packer represented by free rectangles.
        Returns: total_v, placed_list, cut_tree
        """
        placed = []
        total_v = 0.0

        root_tree = {"type": "empty"}
        rects = [(0.0, 0.0, L_stock, W_stock, root_tree)]

        for p_idx in order:
            p = flat_pieces[p_idx]
            p_l, p_w = p["l"], p["w"]

            best_rect_idx = -1
            for r_idx, (rx, ry, rw, rh, _) in enumerate(rects):
                # Do not admit a piece that is actually larger than the free region.
                if p_l <= rw and p_w <= rh:
                    best_rect_idx = r_idx
                    break

            if best_rect_idx == -1:
                continue

            rx, ry, rw, rh, node_to_fill = rects.pop(best_rect_idx)
            placed.append((p["type_idx"], p["inst_idx"], rx, ry))
            total_v += p["v"]

            leaf = {"type": "leaf", "piece_index": [p["type_idx"], p["inst_idx"]]}
            x1 = rx + rw
            y1 = ry + rh
            cut_x = rx + p_l
            cut_y = ry + p_w

            # A cut is emitted only when its coordinate is strictly inside the
            # current region.  When a piece touches an edge, that no-op cut is
            # folded away and the valid branch directly replaces it.
            has_right = rx < cut_x < x1
            has_top = ry < cut_y < y1

            if rw - p_l > rh - p_w:
                # Vertical split first, retaining the original split preference.
                subtree = leaf
                top_node = {"type": "empty"}
                right_node = {"type": "empty"}

                if has_top:
                    subtree = {
                        "type": "node",
                        "cut_direction": "horizontal",
                        "cut_coordinate": float(cut_y),
                        "bottom": leaf,
                        "top": top_node
                    }
                if has_right:
                    subtree = {
                        "type": "node",
                        "cut_direction": "vertical",
                        "cut_coordinate": float(cut_x),
                        "left_child": subtree,
                        "right_child": right_node
                    }

                node_to_fill.clear()
                node_to_fill.update(subtree)

                if has_right:
                    rects.append((cut_x, ry, x1 - cut_x, rh, right_node))
                if has_top:
                    rects.append((rx, cut_y, p_l, y1 - cut_y, top_node))
            else:
                # Horizontal split first, retaining the original split preference.
                subtree = leaf
                top_node = {"type": "empty"}
                right_node = {"type": "empty"}

                if has_right:
                    subtree = {
                        "type": "node",
                        "cut_direction": "vertical",
                        "cut_coordinate": float(cut_x),
                        "left_child": leaf,
                        "right_child": right_node
                    }
                if has_top:
                    subtree = {
                        "type": "node",
                        "cut_direction": "horizontal",
                        "cut_coordinate": float(cut_y),
                        "bottom": subtree,
                        "top": top_node
                    }

                node_to_fill.clear()
                node_to_fill.update(subtree)

                if has_top:
                    rects.append((rx, cut_y, rw, y1 - cut_y, top_node))
                if has_right:
                    rects.append((cut_x, ry, x1 - cut_x, p_w, right_node))

        return total_v, placed, root_tree

    # 2. Simulated Annealing
    current_order = list(range(N))
    random.shuffle(current_order)

    current_val, current_placed, current_tree = pack_guillotine(current_order, L, W)

    best_val = current_val
    best_placed = current_placed
    best_tree = current_tree

    temp = 100.0
    cooling_rate = 0.999

    while N > 0 and time.time() - t0 < time_limit_s - 0.2:
        idx1 = random.randint(0, N - 1)
        idx2 = random.randint(0, N - 1)

        new_order = list(current_order)
        new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]

        new_val, new_placed, new_tree = pack_guillotine(new_order, L, W)

        if new_val >= current_val or math.exp((new_val - current_val) / (temp + 1e-9)) > random.random():
            current_order = new_order
            current_val = new_val
            current_placed = new_placed
            current_tree = new_tree
            if current_val > best_val:
                best_val = current_val
                best_placed = new_placed
                best_tree = new_tree

        temp *= cooling_rate
        if temp < 0.01:
            temp = 100.0

    # 3. Output Building
    selection_status = [[0] * demands_types[i] for i in range(N_types)]
    placements = [[[-1.0, -1.0]] * demands_types[i] for i in range(N_types)]

    for t_idx, i_idx, x, y in best_placed:
        selection_status[t_idx][i_idx] = 1
        placements[t_idx][i_idx] = [float(x), float(y)]

    if best_tree.get("type") == "node":
        best_tree["type"] = "root"

    return {
        "selection_status": selection_status,
        "placements": placements,
        "guillotine_cut_tree": best_tree
    }
