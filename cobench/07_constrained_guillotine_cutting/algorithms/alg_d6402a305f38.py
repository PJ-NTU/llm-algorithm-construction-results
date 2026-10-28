# Library slot 1/10  id=alg_d6402a305f38
# score=-12141.223999999998  feasible_rate=1.0
# score_vec=[-2881.06, -2434.98, -6775.889999999999, -6473.94, -2426.6899999999996, -6707.29, -6878.5999999999985, -16144.33, -37356.049999999996, -33333.409999999996]

import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()

    try:
        lengths = [float(x) for x in instance["piece_lengths"]]
        widths = [float(x) for x in instance["piece_widths"]]
        demands = [int(x) for x in instance["max_quantities"]]
        values = [float(x) for x in instance["piece_values"]]
        L = float(instance["sheet_length"])
        W = float(instance["sheet_width"])
        n = int(instance["num_piece_types"])
    except Exception:
        return {
            "selection_status": [],
            "placements": [],
            "guillotine_cut_tree": {"type": "empty"},
        }

    def empty_solution():
        return {
            "selection_status": [[0 for _ in range(demands[i])] for i in range(n)],
            "placements": [
                [[-1.0, -1.0] for _ in range(demands[i])] for i in range(n)
            ],
            "guillotine_cut_tree": {"type": "empty"},
        }

    best = empty_solution()
    best_value = 0.0
    if time_limit_s <= 0.0 or n <= 0 or L <= 0.0 or W <= 0.0:
        return best

    deadline = t0 + max(0.0, time_limit_s) * 0.85
    eps = 1e-8
    valid_types = [
        i
        for i in range(n)
        if demands[i] > 0
        and values[i] > 0.0
        and lengths[i] > 0.0
        and widths[i] > 0.0
        and lengths[i] <= L + eps
        and widths[i] <= W + eps
    ]
    density_order = sorted(
        valid_types,
        key=lambda i: values[i] / (lengths[i] * widths[i]),
        reverse=True,
    )

    def accept(placed, tree, total):
        nonlocal best, best_value
        if total <= best_value + 1e-9:
            return
        selection = [[0 for _ in range(demands[i])] for i in range(n)]
        coords = [
            [[-1.0, -1.0] for _ in range(demands[i])] for i in range(n)
        ]
        for i, p, x, y in placed:
            selection[i][p] = 1
            coords[i][p] = [float(x), float(y)]
        if tree.get("type") == "node":
            tree["type"] = "root"
        best = {
            "selection_status": selection,
            "placements": coords,
            "guillotine_cut_tree": tree,
        }
        best_value = float(total)

    def make_piece_leaf(i, p):
        return {"type": "leaf", "piece_index": [i, p]}

    def quality(rw, rh, remaining, removed=-1):
        if rw <= eps or rh <= eps:
            return 0.0
        area = rw * rh
        max_density = 0.0
        possible_value = 0.0
        for i in density_order:
            count = remaining[i] - (1 if i == removed else 0)
            if count <= 0 or lengths[i] > rw + eps or widths[i] > rh + eps:
                continue
            den = values[i] / (lengths[i] * widths[i])
            if den > max_density:
                max_density = den
            possible_value += values[i] * count
        return min(possible_value, area * max_density)

    def place_in_slot(slot, x, y, rw, rh, i, p, vertical_first):
        a = lengths[i]
        b = widths[i]
        leaf = make_piece_leaf(i, p)
        has_right = rw - a > eps
        has_top = rh - b > eps
        new_free = []

        if not has_right and not has_top:
            slot.clear()
            slot.update(leaf)
            return new_free

        if not has_right:
            top = {"type": "empty"}
            slot.clear()
            slot.update(
                {
                    "type": "node",
                    "cut_direction": "horizontal",
                    "cut_coordinate": float(y + b),
                    "bottom": leaf,
                    "top": top,
                }
            )
            new_free.append((x, y + b, rw, rh - b, top))
            return new_free

        if not has_top:
            right = {"type": "empty"}
            slot.clear()
            slot.update(
                {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": float(x + a),
                    "left_child": leaf,
                    "right_child": right,
                }
            )
            new_free.append((x + a, y, rw - a, rh, right))
            return new_free

        if vertical_first:
            top = {"type": "empty"}
            right = {"type": "empty"}
            left = {
                "type": "node",
                "cut_direction": "horizontal",
                "cut_coordinate": float(y + b),
                "bottom": leaf,
                "top": top,
            }
            slot.clear()
            slot.update(
                {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": float(x + a),
                    "left_child": left,
                    "right_child": right,
                }
            )
            new_free.append((x, y + b, a, rh - b, top))
            new_free.append((x + a, y, rw - a, rh, right))
        else:
            right = {"type": "empty"}
            top = {"type": "empty"}
            bottom = {
                "type": "node",
                "cut_direction": "vertical",
                "cut_coordinate": float(x + a),
                "left_child": leaf,
                "right_child": right,
            }
            slot.clear()
            slot.update(
                {
                    "type": "node",
                    "cut_direction": "horizontal",
                    "cut_coordinate": float(y + b),
                    "bottom": bottom,
                    "top": top,
                }
            )
            new_free.append((x + a, y, rw - a, b, right))
            new_free.append((x, y + b, rw, rh - b, top))
        return new_free

    def guillotine_construct(rng, mode, randomness, lookahead):
        remaining = list(demands)
        root = {"type": "empty"}
        free = [(0.0, 0.0, L, W, root)]
        placed = []
        total = 0.0

        while free and time.time() < deadline:
            usable = []
            for k, rect in enumerate(free):
                rw, rh = rect[2], rect[3]
                fits = False
                for i in valid_types:
                    if (
                        remaining[i] > 0
                        and lengths[i] <= rw + eps
                        and widths[i] <= rh + eps
                    ):
                        fits = True
                        break
                if fits:
                    usable.append(k)
            if not usable:
                break

            if mode == 0:
                chosen_free = max(usable, key=lambda k: free[k][2] * free[k][3])
            elif mode == 1:
                chosen_free = min(usable, key=lambda k: free[k][2] * free[k][3])
            elif mode == 2:
                chosen_free = max(
                    usable,
                    key=lambda k: quality(free[k][2], free[k][3], remaining),
                )
            elif mode == 3:
                chosen_free = usable[-1]
            else:
                chosen_free = usable[int(rng.random() * len(usable))]

            x, y, rw, rh, slot = free.pop(chosen_free)
            parent_q = quality(rw, rh, remaining)
            actions = []
            max_v = 0.0
            for i in valid_types:
                if (
                    remaining[i] <= 0
                    or lengths[i] > rw + eps
                    or widths[i] > rh + eps
                ):
                    continue
                if values[i] > max_v:
                    max_v = values[i]
                a, b = lengths[i], widths[i]
                if rw - a <= eps or rh - b <= eps:
                    orientations = (True,)
                else:
                    orientations = (True, False)
                for vertical_first in orientations:
                    if vertical_first:
                        children = ((a, rh - b), (rw - a, rh))
                    else:
                        children = ((rw - a, b), (rw, rh - b))
                    future = quality(
                        children[0][0], children[0][1], remaining, i
                    ) + quality(children[1][0], children[1][1], remaining, i)
                    exact_edges = (1 if abs(rw - a) <= eps else 0) + (
                        1 if abs(rh - b) <= eps else 0
                    )
                    density = values[i] / (a * b)
                    preservation = future - parent_q
                    score = values[i] + lookahead * preservation
                    score += 0.025 * values[i] * exact_edges
                    if mode == 1:
                        score += 0.08 * values[i] * (a * b) / (rw * rh)
                    elif mode == 3:
                        score += 0.04 * density * min(rw * rh, a * b * 4.0)
                    actions.append((score, future, density, i, vertical_first))

            if not actions:
                continue
            actions.sort(key=lambda z: (z[0], z[1], z[2]), reverse=True)
            if randomness <= 0.0 or len(actions) == 1:
                action = actions[0]
            else:
                rcl = min(len(actions), 2 + int(randomness * 7.0))
                if rng.random() < 0.70:
                    pick = int((rng.random() ** (2.0 + 3.0 * (1.0 - randomness))) * rcl)
                else:
                    pick = int(rng.random() * rcl)
                action = actions[pick]

            i = action[3]
            vertical_first = action[4]
            p = demands[i] - remaining[i]
            remaining[i] -= 1
            placed.append((i, p, x, y))
            total += values[i]
            free.extend(
                place_in_slot(slot, x, y, rw, rh, i, p, vertical_first)
            )

        return total, placed, root

    def state_to_layout(state):
        total, remaining, free, nodes = state
        placed = []

        def emit(node_id, x, y, rw, rh):
            op = nodes[node_id]
            if op is None:
                return {"type": "empty"}
            i, p, vertical_first, top_id, right_id = op
            a, b = lengths[i], widths[i]
            leaf = make_piece_leaf(i, p)
            placed.append((i, p, x, y))
            if top_id is None and right_id is None:
                return leaf
            if right_id is None:
                return {
                    "type": "node",
                    "cut_direction": "horizontal",
                    "cut_coordinate": float(y + b),
                    "bottom": leaf,
                    "top": emit(top_id, x, y + b, rw, rh - b),
                }
            if top_id is None:
                return {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": float(x + a),
                    "left_child": leaf,
                    "right_child": emit(right_id, x + a, y, rw - a, rh),
                }
            if vertical_first:
                left = {
                    "type": "node",
                    "cut_direction": "horizontal",
                    "cut_coordinate": float(y + b),
                    "bottom": leaf,
                    "top": emit(top_id, x, y + b, a, rh - b),
                }
                return {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": float(x + a),
                    "left_child": left,
                    "right_child": emit(right_id, x + a, y, rw - a, rh),
                }
            bottom = {
                "type": "node",
                "cut_direction": "vertical",
                "cut_coordinate": float(x + a),
                "left_child": leaf,
                "right_child": emit(right_id, x + a, y, rw - a, b),
            }
            return {
                "type": "node",
                "cut_direction": "horizontal",
                "cut_coordinate": float(y + b),
                "bottom": bottom,
                "top": emit(top_id, x, y + b, rw, rh - b),
            }

        return placed, emit(0, 0.0, 0.0, L, W)

    def beam_upper(state):
        total, remaining, free, nodes = state
        capacity = sum(rect[2] * rect[3] for rect in free)
        extra = 0.0
        for i in density_order:
            if remaining[i] <= 0:
                continue
            fits = False
            for rect in free:
                if lengths[i] <= rect[2] + eps and widths[i] <= rect[3] + eps:
                    fits = True
                    break
            if not fits:
                continue
            area = lengths[i] * widths[i]
            take = min(float(remaining[i]), capacity / area)
            extra += take * values[i]
            capacity -= take * area
            if capacity <= eps:
                break
        return total + extra

    def beam_search(width_limit, branch_limit, rect_mode, future_weight, stop_time):
        initial = (
            0.0,
            tuple(demands),
            ((0.0, 0.0, L, W, 0),),
            (None,),
        )
        beam = [initial]
        depth = 0
        while beam and time.time() < stop_time:
            generated = []
            for state in beam:
                if time.time() >= stop_time:
                    break
                total, remaining, free, nodes = state
                usable = []
                for k, rect in enumerate(free):
                    for i in valid_types:
                        if (
                            remaining[i] > 0
                            and lengths[i] <= rect[2] + eps
                            and widths[i] <= rect[3] + eps
                        ):
                            usable.append(k)
                            break
                if not usable:
                    if total > best_value + 1e-9:
                        placed, tree = state_to_layout(state)
                        accept(placed, tree, total)
                    continue
                if rect_mode == 0:
                    k = max(usable, key=lambda q: free[q][2] * free[q][3])
                elif rect_mode == 1:
                    k = min(usable, key=lambda q: free[q][2] * free[q][3])
                else:
                    k = max(
                        usable,
                        key=lambda q: quality(
                            free[q][2], free[q][3], remaining
                        ),
                    )
                x, y, rw, rh, node_id = free[k]
                parent_q = quality(rw, rh, remaining)
                actions = []
                for i in valid_types:
                    if (
                        remaining[i] <= 0
                        or lengths[i] > rw + eps
                        or widths[i] > rh + eps
                    ):
                        continue
                    a, b = lengths[i], widths[i]
                    orientations = (
                        (True,)
                        if rw - a <= eps or rh - b <= eps
                        else (True, False)
                    )
                    for vertical_first in orientations:
                        if vertical_first:
                            children = ((a, rh - b), (rw - a, rh))
                        else:
                            children = ((rw - a, b), (rw, rh - b))
                        future = quality(
                            children[0][0], children[0][1], remaining, i
                        ) + quality(
                            children[1][0], children[1][1], remaining, i
                        )
                        score = values[i] + future_weight * (future - parent_q)
                        actions.append((score, values[i], future, i, vertical_first))
                actions.sort(key=lambda z: (z[0], z[1], z[2]), reverse=True)
                chosen_actions = actions[:branch_limit]
                by_value = sorted(
                    actions, key=lambda z: (z[1], z[2]), reverse=True
                )[: max(2, branch_limit // 3)]
                seen_actions = {(z[3], z[4]) for z in chosen_actions}
                for action in by_value:
                    if (action[3], action[4]) not in seen_actions:
                        chosen_actions.append(action)
                        seen_actions.add((action[3], action[4]))

                for action in chosen_actions:
                    i, vertical_first = action[3], action[4]
                    a, b = lengths[i], widths[i]
                    new_remaining = list(remaining)
                    p = demands[i] - new_remaining[i]
                    new_remaining[i] -= 1
                    new_nodes = list(nodes)
                    top_id = None
                    right_id = None
                    new_free = list(free)
                    new_free.pop(k)
                    if rh - b > eps:
                        top_id = len(new_nodes)
                        new_nodes.append(None)
                    if rw - a > eps:
                        right_id = len(new_nodes)
                        new_nodes.append(None)
                    new_nodes[node_id] = (i, p, vertical_first, top_id, right_id)
                    if top_id is not None:
                        if vertical_first and right_id is not None:
                            new_free.append((x, y + b, a, rh - b, top_id))
                        else:
                            new_free.append((x, y + b, rw, rh - b, top_id))
                    if right_id is not None:
                        if not vertical_first and top_id is not None:
                            new_free.append((x + a, y, rw - a, b, right_id))
                        else:
                            new_free.append((x + a, y, rw - a, rh, right_id))
                    child = (
                        total + values[i],
                        tuple(new_remaining),
                        tuple(new_free),
                        tuple(new_nodes),
                    )
                    generated.append(child)
                    if child[0] > best_value + 1e-9:
                        placed, tree = state_to_layout(child)
                        accept(placed, tree, child[0])

            if not generated:
                break
            ranked = []
            signatures = set()
            for state in generated:
                remaining = state[1]
                free = state[2]
                signature = remaining + tuple(
                    sorted((round(r[2], 6), round(r[3], 6)) for r in free)
                )
                if signature in signatures:
                    continue
                signatures.add(signature)
                upper = beam_upper(state)
                shape = sum(
                    quality(r[2], r[3], remaining) for r in state[2]
                )
                optimism = min(max(0.0, upper - state[0]), shape)
                rank = state[0] + 0.62 * optimism
                ranked.append((rank, state[0], state))
            ranked.sort(key=lambda z: (z[0], z[1]), reverse=True)
            beam = [z[2] for z in ranked[:width_limit]]
            depth += 1

    def block_construct(rng, alpha, noise, choice_mode):
        ordered = []
        for i in valid_types:
            area = lengths[i] * widths[i]
            for p in range(demands[i]):
                priority = values[i] / (area ** alpha)
                priority *= math.exp(noise * (rng.random() - 0.5))
                ordered.append((priority, values[i], i, p))
        ordered.sort(reverse=True)
        block = None
        total = 0.0
        for priority, value, i, p in ordered:
            if time.time() >= deadline:
                break
            a, b = lengths[i], widths[i]
            leaf = (a, b, i, p)
            if block is None:
                block = leaf
                total += value
                continue
            bw, bh = block[0], block[1]
            choices = []
            vw, vh = bw + a, max(bh, b)
            if vw <= L + eps and vh <= W + eps:
                growth = vw * vh - bw * bh
                choices.append((growth, abs(bh - b), "vertical", vw, vh))
            hw, hh = max(bw, a), bh + b
            if hw <= L + eps and hh <= W + eps:
                growth = hw * hh - bw * bh
                choices.append((growth, abs(bw - a), "horizontal", hw, hh))
            if not choices:
                continue
            choices.sort()
            if choice_mode == 1:
                choice = min(choices, key=lambda z: z[1])
            elif choice_mode == 2 and len(choices) > 1 and rng.random() < 0.45:
                choice = choices[1]
            else:
                choice = choices[0]
            block = (choice[3], choice[4], choice[2], block, leaf)
            total += value

        placed = []

        def emit(current, x, y):
            if len(current) == 4:
                i, p = current[2], current[3]
                placed.append((i, p, x, y))
                return make_piece_leaf(i, p)
            direction, first, second = current[2], current[3], current[4]
            if direction == "vertical":
                cut = x + first[0]
                return {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": float(cut),
                    "left_child": emit(first, x, y),
                    "right_child": emit(second, cut, y),
                }
            cut = y + first[1]
            return {
                "type": "node",
                "cut_direction": "horizontal",
                "cut_coordinate": float(cut),
                "bottom": emit(first, x, y),
                "top": emit(second, x, cut),
            }

        if block is None:
            return 0.0, [], {"type": "empty"}
        return total, placed, emit(block, 0.0, 0.0)

    def make_line_tree(items, horizontal_axis):
        if not items:
            return {"type": "empty"}
        first = items[0]
        leaf = make_piece_leaf(first[0], first[1])
        if len(items) == 1:
            return leaf
        rest = make_line_tree(items[1:], horizontal_axis)
        if horizontal_axis:
            return {
                "type": "node",
                "cut_direction": "vertical",
                "cut_coordinate": float(first[2] + lengths[first[0]]),
                "left_child": leaf,
                "right_child": rest,
            }
        return {
            "type": "node",
            "cut_direction": "horizontal",
            "cut_coordinate": float(first[3] + widths[first[0]]),
            "bottom": leaf,
            "top": rest,
        }

    def make_band_tree(bands, horizontal_bands, k=0):
        if k >= len(bands):
            return {"type": "empty"}
        band = bands[k]
        line = make_line_tree(band["items"], horizontal_bands)
        if k + 1 == len(bands):
            return line
        rest = make_band_tree(bands, horizontal_bands, k + 1)
        if horizontal_bands:
            return {
                "type": "node",
                "cut_direction": "horizontal",
                "cut_coordinate": float(band["offset"] + band["thick"]),
                "bottom": line,
                "top": rest,
            }
        return {
            "type": "node",
            "cut_direction": "vertical",
            "cut_coordinate": float(band["offset"] + band["thick"]),
            "left_child": line,
            "right_child": rest,
        }

    def shelf_construct(rng, horizontal_bands, alpha, noise, fit_mode):
        if horizontal_bands:
            capacity, max_thickness = L, W
        else:
            capacity, max_thickness = W, L
        items = []
        for i in valid_types:
            along = lengths[i] if horizontal_bands else widths[i]
            thick = widths[i] if horizontal_bands else lengths[i]
            for p in range(demands[i]):
                jitter = math.exp(noise * (rng.random() - 0.5))
                base = values[i] / ((lengths[i] * widths[i]) ** alpha)
                shape = 1.0 + 0.04 * along / capacity
                items.append((base * jitter * shape, i, p, along, thick))
        items.sort(key=lambda z: z[0], reverse=True)

        bands = []
        total_thickness = 0.0
        total = 0.0
        for _, i, p, along, thick in items:
            if time.time() >= deadline:
                break
            options = []
            for k, band in enumerate(bands):
                if band["used"] + along > capacity + eps:
                    continue
                raised = max(0.0, thick - band["thick"])
                if total_thickness + raised > max_thickness + eps:
                    continue
                leftover = capacity - band["used"] - along
                cost = raised * capacity + leftover * 0.001
                if fit_mode == 1:
                    cost += abs(band["thick"] - thick) * along * 0.05
                options.append((cost, k, raised))
            if total_thickness + thick <= max_thickness + eps:
                new_cost = thick * capacity
                if fit_mode == 2:
                    new_cost *= 0.75
                options.append((new_cost, len(bands), thick))
            if not options:
                continue
            options.sort()
            if noise > 0.5 and len(options) > 1 and rng.random() < 0.12:
                choice = options[1]
            else:
                choice = options[0]
            k, raised = choice[1], choice[2]
            if k == len(bands):
                bands.append({"used": 0.0, "thick": thick, "items": []})
            else:
                bands[k]["thick"] += raised
            total_thickness += raised
            band = bands[k]
            band["items"].append((i, p, 0.0, 0.0))
            band["used"] += along
            total += values[i]

        offset = 0.0
        placed = []
        for band in bands:
            band["offset"] = offset
            along_pos = 0.0
            positioned = []
            for i, p, _, _ in band["items"]:
                if horizontal_bands:
                    x, y = along_pos, offset
                    along_pos += lengths[i]
                else:
                    x, y = offset, along_pos
                    along_pos += widths[i]
                positioned.append((i, p, x, y))
                placed.append((i, p, x, y))
            band["items"] = positioned
            offset += band["thick"]
        tree = make_band_tree(bands, horizontal_bands)
        return total, placed, tree

    def knapsack_shelves(horizontal_bands, height_power, fill_bias):
        if horizontal_bands:
            capacity, max_thickness = L, W
        else:
            capacity, max_thickness = W, L
        available = []
        for i in valid_types:
            along = lengths[i] if horizontal_bands else widths[i]
            thick = widths[i] if horizontal_bands else lengths[i]
            for p in range(demands[i]):
                available.append((i, p, along, thick, values[i]))

        bands = []
        thickness_used = 0.0
        bins = 720
        unit = capacity / bins
        while available and time.time() < deadline:
            room = max_thickness - thickness_used
            eligible = [
                j for j, item in enumerate(available) if item[3] <= room + eps
            ]
            if not eligible:
                break
            eligible.sort(key=lambda j: available[j][3])
            dp = [0.0] * (bins + 1)
            masks = [0] * (bins + 1)
            candidates = []
            q = 0
            while q < len(eligible):
                thick = available[eligible[q]][3]
                group_end = q
                while (
                    group_end < len(eligible)
                    and abs(available[eligible[group_end]][3] - thick) <= eps
                ):
                    j = eligible[group_end]
                    along = available[j][2]
                    weight = max(1, int(math.ceil(along / unit - 1e-12)))
                    value = available[j][4]
                    if weight <= bins:
                        bit = 1 << j
                        for c in range(bins, weight - 1, -1):
                            proposal = dp[c - weight] + value
                            if proposal > dp[c] + 1e-10:
                                dp[c] = proposal
                                masks[c] = masks[c - weight] | bit
                    group_end += 1
                best_c = max(range(bins + 1), key=lambda c: dp[c])
                mask = masks[best_c]
                if mask:
                    actual_h = 0.0
                    actual_along = 0.0
                    for j in eligible[:group_end]:
                        if mask & (1 << j):
                            actual_h = max(actual_h, available[j][3])
                            actual_along += available[j][2]
                    score = dp[best_c] / (actual_h ** height_power)
                    score *= 1.0 + fill_bias * actual_along / capacity
                    candidates.append((score, dp[best_c], mask, actual_h))
                q = group_end
                if time.time() >= deadline:
                    break
            if not candidates:
                break
            _, band_value, chosen_mask, band_height = max(
                candidates, key=lambda z: (z[0], z[1])
            )
            chosen = [j for j in range(len(available)) if chosen_mask & (1 << j)]
            chosen.sort(key=lambda j: (available[j][2], available[j][3]), reverse=True)
            band_items = [
                (available[j][0], available[j][1], 0.0, 0.0) for j in chosen
            ]
            bands.append(
                {
                    "used": sum(available[j][2] for j in chosen),
                    "thick": band_height,
                    "items": band_items,
                }
            )
            thickness_used += band_height
            chosen_set = set(chosen)
            available = [
                item for j, item in enumerate(available) if j not in chosen_set
            ]

        offset = 0.0
        placed = []
        total = 0.0
        for band in bands:
            band["offset"] = offset
            along_pos = 0.0
            positioned = []
            for i, p, _, _ in band["items"]:
                if horizontal_bands:
                    x, y = along_pos, offset
                    along_pos += lengths[i]
                else:
                    x, y = offset, along_pos
                    along_pos += widths[i]
                positioned.append((i, p, x, y))
                placed.append((i, p, x, y))
                total += values[i]
            band["items"] = positioned
            offset += band["thick"]
        return total, placed, make_band_tree(bands, horizontal_bands)

    try:
        # Deterministic incumbents come first, so even tiny budgets return useful layouts.
        for mode, lookahead in ((0, 0.75), (2, 0.55), (3, 0.35), (1, 0.20)):
            if time.time() >= deadline:
                break
            total, placed, tree = guillotine_construct(
                random.Random(9173 + mode), mode, 0.0, lookahead
            )
            accept(placed, tree, total)

        for horizontal in (True, False):
            for alpha in (1.0, 0.72, 0.45, 0.15):
                if time.time() >= deadline:
                    break
                total, placed, tree = shelf_construct(
                    random.Random(19001 + int(horizontal) * 101 + int(alpha * 100)),
                    horizontal,
                    alpha,
                    0.0,
                    0,
                )
                accept(placed, tree, total)

        for horizontal in (True, False):
            for height_power, fill_bias in (
                (1.15, 0.0),
                (0.95, 0.08),
                (0.75, 0.16),
                (0.52, 0.22),
            ):
                if time.time() >= deadline:
                    break
                total, placed, tree = knapsack_shelves(
                    horizontal, height_power, fill_bias
                )
                accept(placed, tree, total)

        beam_slice = max(0.0, time_limit_s) * 0.065
        for rect_mode, future_weight in ((0, 0.55), (2, 0.75), (1, 0.30)):
            if time.time() >= deadline:
                break
            beam_search(
                72,
                8,
                rect_mode,
                future_weight,
                min(deadline, time.time() + beam_slice),
            )

        iteration = 0
        seed_base = int(L * 17.0 + W * 31.0 + n * 1009.0) & 0xFFFFFFFF
        while time.time() < deadline:
            rng = random.Random(seed_base + iteration * 104729)
            phase = iteration % 10
            if phase < 6:
                mode = phase % 5
                randomness = (0.18, 0.32, 0.52, 0.75)[(iteration // 10) % 4]
                lookahead = (0.15, 0.35, 0.60, 0.90)[(iteration // 40) % 4]
                total, placed, tree = guillotine_construct(
                    rng, mode, randomness, lookahead
                )
            elif phase < 9:
                horizontal = phase != 8
                alpha = (0.15, 0.35, 0.55, 0.75, 0.95)[(iteration // 10) % 5]
                noise = (0.35, 0.70, 1.10)[(iteration // 50) % 3]
                fit_mode = (iteration // 150) % 3
                total, placed, tree = shelf_construct(
                    rng, horizontal, alpha, noise, fit_mode
                )
            else:
                alpha = (0.15, 0.35, 0.55, 0.75, 0.95)[(iteration // 10) % 5]
                noise = (0.35, 0.75, 1.20)[(iteration // 50) % 3]
                total, placed, tree = block_construct(
                    rng, alpha, noise, (iteration // 150) % 3
                )
            accept(placed, tree, total)
            iteration += 1
    except Exception:
        pass

    return best
