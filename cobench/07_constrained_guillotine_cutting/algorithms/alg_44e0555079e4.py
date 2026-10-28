# Library slot 10/10  id=alg_44e0555079e4
# score=-11770.532  feasible_rate=1.0
# score_vec=[-2833.4299999999994, -2434.98, -6775.889999999999, -6473.94, -2426.6899999999996, -6743.29, -6878.5999999999985, -15501.249999999998, -36049.66, -31587.589999999997]

import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime hybrid strip/BSP heuristic for fixed-orientation guillotine packing."""
    t0 = time.time()
    deadline = t0 + max(0.0, float(time_limit_s)) * 0.85

    # The empty packing is the emergency incumbent, so every exit is feasible.
    try:
        n = int(instance["num_piece_types"])
        sheet_l = float(instance["sheet_length"])
        sheet_w = float(instance["sheet_width"])
        lengths = [float(x) for x in instance["piece_lengths"][:n]]
        widths = [float(x) for x in instance["piece_widths"][:n]]
        demands = [max(0, int(x)) for x in instance["max_quantities"][:n]]
        values = [float(x) for x in instance["piece_values"][:n]]
    except Exception:
        return {
            "selection_status": [],
            "placements": [],
            "guillotine_cut_tree": {"type": "empty"},
        }

    empty_solution = {
        "selection_status": [[0 for _ in range(demands[i])] for i in range(n)],
        "placements": [
            [[-1.0, -1.0] for _ in range(demands[i])] for i in range(n)
        ],
        "guillotine_cut_tree": {"type": "empty"},
    }
    if (
        n <= 0
        or len(lengths) != n
        or len(widths) != n
        or len(demands) != n
        or len(values) != n
        or sheet_l <= 0.0
        or sheet_w <= 0.0
        or time.time() >= deadline
    ):
        return empty_solution

    eps = 1e-8
    areas = [lengths[i] * widths[i] for i in range(n)]
    densities = [
        values[i] / areas[i] if areas[i] > eps else -1e100 for i in range(n)
    ]

    # A state is (remaining counts, free rectangles, actions, accumulated value).
    # Rectangles are (tree-slot, x, y, length, width).  Tree slots use heap ids.
    initial_state = (
        tuple(demands),
        ((0, 0.0, 0.0, sheet_l, sheet_w),),
        (),
        0.0,
    )
    best_state = initial_state
    best_value = 0.0

    def fitting_types(rect, counts):
        _, _, _, rw, rh = rect
        return [
            i
            for i in range(n)
            if counts[i] > 0
            and values[i] > 0.0
            and lengths[i] <= rw + eps
            and widths[i] <= rh + eps
            and lengths[i] > eps
            and widths[i] > eps
        ]

    def choose_rectangle(free_rects, counts, jitter, rng):
        chosen_pos = -1
        chosen_fit = []
        chosen_score = -1e300
        for pos, rect in enumerate(free_rects):
            if time.time() >= deadline:
                break
            fit = fitting_types(rect, counts)
            if not fit:
                continue
            rw, rh = rect[3], rect[4]
            top_density = max(densities[i] for i in fit)
            top_value = max(values[i] for i in fit)
            # Best-first filling preserves useful narrow strips instead of
            # blindly recursing down one side of the partition tree.
            score = min(rw * rh * top_density, sum(values[i] * counts[i] for i in fit))
            score += 0.12 * top_value
            if jitter:
                score *= 0.82 + 0.36 * rng.random()
            if score > chosen_score:
                chosen_score = score
                chosen_pos = pos
                chosen_fit = fit
        return chosen_pos, chosen_fit

    def strip_sequence(fit, counts, cap, band_limit, horizontal, mode, rng):
        remaining = list(counts)
        sequence = []
        used = 0.0
        while time.time() < deadline:
            chosen = -1
            chosen_score = -1e300
            for i in fit:
                size = lengths[i] if horizontal else widths[i]
                cross = widths[i] if horizontal else lengths[i]
                if remaining[i] <= 0 or cross > band_limit + eps or used + size > cap + eps:
                    continue
                if mode == 0:
                    score = values[i] / max(size, eps)
                elif mode == 1:
                    score = densities[i]
                else:
                    # Raw value matters when demands are scarce; the small
                    # fill term breaks ties toward a tighter strip.
                    score = values[i] * (0.88 + 0.24 * rng.random())
                    score += 0.04 * values[i] * size / max(cap - used, size)
                if score > chosen_score:
                    chosen_score = score
                    chosen = i
            if chosen < 0:
                break
            sequence.append(chosen)
            remaining[chosen] -= 1
            used += lengths[chosen] if horizontal else widths[chosen]
        return tuple(sequence)

    def strip_knapsack(fit, counts, cap, band_limit, horizontal):
        """Small Pareto beam for a high-value bounded one-dimensional strip."""
        items = []
        best_ratio = 0.0
        for i in fit:
            size = lengths[i] if horizontal else widths[i]
            cross = widths[i] if horizontal else lengths[i]
            if cross > band_limit + eps or size > cap + eps:
                continue
            usable = min(counts[i], int((cap + eps) / size))
            if usable > 0:
                items.extend([i] * usable)
                best_ratio = max(best_ratio, values[i] / size)
            if len(items) >= 100:
                break
        if len(items) < 2:
            return ()

        frontier = [(0.0, 0.0, ())]
        for i in items:
            if time.time() >= deadline:
                break
            size = lengths[i] if horizontal else widths[i]
            added = []
            for used, value, seq in frontier:
                if used + size <= cap + eps:
                    added.append((used + size, value + values[i], seq + (i,)))
            merged = frontier + added
            merged.sort(key=lambda z: (z[0], -z[1]))
            pareto = []
            high_value = -1e300
            for state in merged:
                if state[1] > high_value + eps:
                    pareto.append(state)
                    high_value = state[1]
            if len(pareto) > 140:
                # Keep both strong completed strips and light states with
                # enough residual capacity to overtake them later.
                ranked = sorted(
                    pareto,
                    key=lambda z: z[1] + 0.55 * best_ratio * (cap - z[0]),
                    reverse=True,
                )[:100]
                ranked += sorted(pareto, key=lambda z: z[1], reverse=True)[:40]
                unique = {}
                for state in ranked:
                    key = (round(state[0], 8), round(state[1], 8))
                    unique[key] = state
                frontier = list(unique.values())
            else:
                frontier = pareto
        if not frontier:
            return ()
        return max(frontier, key=lambda z: z[1])[2]

    def make_actions(rect, counts, fit, rng, thorough):
        sid, x0, y0, rw, rh = rect
        actions = []

        # Preserve candidates from four different economic views.  This is
        # much less myopic than the density-only RCL of the parent BSP method.
        if len(fit) <= 14 or thorough:
            corner_types = list(fit)
        else:
            corner_types = []
            orders = [
                sorted(fit, key=lambda i: densities[i], reverse=True),
                sorted(fit, key=lambda i: values[i], reverse=True),
                sorted(fit, key=lambda i: values[i] / lengths[i], reverse=True),
                sorted(fit, key=lambda i: values[i] / widths[i], reverse=True),
                sorted(
                    fit,
                    key=lambda i: densities[i] * (0.75 + 0.5 * rng.random()),
                    reverse=True,
                ),
            ]
            seen = set()
            for order in orders:
                for i in order[:4]:
                    if i not in seen:
                        seen.add(i)
                        corner_types.append(i)

        for i in corner_types:
            # cv: vertical outer cut, then horizontal around the piece.
            # ch: horizontal outer cut, then vertical around the piece.
            one = (i,)
            actions.append(("cv", sid, x0, y0, rw, rh, one, lengths[i], widths[i]))
            actions.append(("ch", sid, x0, y0, rw, rh, one, widths[i], lengths[i]))

        def band_limits(horizontal):
            keyed = []
            for i in fit:
                cross = widths[i] if horizontal else lengths[i]
                primary = lengths[i] if horizontal else widths[i]
                keyed.append((cross, values[i], densities[i], values[i] / primary))
            if thorough:
                limits = [z[0] for z in keyed]
            else:
                by_value = sorted(keyed, key=lambda z: z[1], reverse=True)
                by_density = sorted(keyed, key=lambda z: z[2], reverse=True)
                by_primary = sorted(keyed, key=lambda z: z[3], reverse=True)
                limits = [min(z[0] for z in keyed), max(z[0] for z in keyed)]
                limits += [z[0] for z in by_value[:2]]
                limits += [z[0] for z in by_density[:2]]
                limits += [z[0] for z in by_primary[:2]]
                limits.append(keyed[int(rng.random() * len(keyed))][0])
            return sorted(set(limits))

        # Shelf/column macro-actions are embedded at every BSP region.  Their
        # two residual rectangles return to the same search frontier.
        strip_seen = set()
        for horizontal in (True, False):
            cap = rw if horizontal else rh
            for limit in band_limits(horizontal):
                for mode in (0, 1, 2):
                    seq = strip_sequence(fit, counts, cap, limit, horizontal, mode, rng)
                    if len(seq) < 2:
                        continue
                    used = sum(lengths[i] if horizontal else widths[i] for i in seq)
                    band = max(widths[i] if horizontal else lengths[i] for i in seq)
                    signature = (horizontal, tuple(sorted(seq)), round(used, 8), round(band, 8))
                    if signature in strip_seen:
                        continue
                    strip_seen.add(signature)
                    kind = "row" if horizontal else "col"
                    actions.append((kind, sid, x0, y0, rw, rh, seq, used, band))
                if thorough and time.time() < deadline:
                    seq = strip_knapsack(fit, counts, cap, limit, horizontal)
                    if len(seq) >= 2:
                        used = sum(lengths[i] if horizontal else widths[i] for i in seq)
                        band = max(widths[i] if horizontal else lengths[i] for i in seq)
                        signature = (
                            horizontal,
                            tuple(sorted(seq)),
                            round(used, 8),
                            round(band, 8),
                        )
                        if signature not in strip_seen:
                            strip_seen.add(signature)
                            kind = "row" if horizontal else "col"
                            actions.append((kind, sid, x0, y0, rw, rh, seq, used, band))
        return actions

    def apply_action(state, rect_pos, action):
        counts, free_rects, actions, value = state
        kind, sid, x0, y0, rw, rh, seq, used, band = action
        new_counts = list(counts)
        gain = 0.0
        for i in seq:
            if new_counts[i] <= 0:
                return None
            new_counts[i] -= 1
            gain += values[i]

        new_free = list(free_rects)
        new_free.pop(rect_pos)
        child_a = sid * 2 + 1
        child_b = sid * 2 + 2
        if kind in ("cv", "row"):
            if rh - band > eps and used > eps:
                new_free.append((child_a, x0, y0 + band, used, rh - band))
            if rw - used > eps:
                new_free.append((child_b, x0 + used, y0, rw - used, rh))
        else:
            if rw - band > eps and used > eps:
                new_free.append((child_a, x0 + band, y0, rw - band, used))
            if rh - used > eps:
                new_free.append((child_b, x0, y0 + used, rw, rh - used))
        return (tuple(new_counts), tuple(new_free), actions + (action,), value + gain)

    def action_merit(action, variant, rng):
        kind, _, _, _, _, _, seq, used, band = action
        gain = sum(values[i] for i in seq)
        box_area = max(used * band, eps)
        if variant == 0:
            merit = gain / box_area
        elif variant == 1:
            merit = gain
        elif variant == 2:
            merit = gain / (box_area ** 0.72)
        else:
            # Encourage genuine strip actions on some restarts, without
            # forcing them when their shelf waste is poor.
            merit = gain / (box_area ** 0.84)
            if kind in ("row", "col"):
                merit *= 1.08 + 0.015 * min(len(seq), 8)
        return merit * (0.92 + 0.16 * rng.random())

    def greedy_construct(seed, variant, thorough):
        state = initial_state
        rng = random.Random(seed)
        while time.time() < deadline:
            counts, free_rects, _, _ = state
            rect_pos, fit = choose_rectangle(free_rects, counts, variant >= 2, rng)
            if rect_pos < 0:
                break
            actions = make_actions(
                free_rects[rect_pos], counts, fit, rng, thorough and not state[2]
            )
            if not actions:
                # This rectangle cannot help; discard it and continue.
                reduced = list(free_rects)
                reduced.pop(rect_pos)
                state = (counts, tuple(reduced), state[2], state[3])
                continue
            action = max(actions, key=lambda a: action_merit(a, variant, rng))
            next_state = apply_action(state, rect_pos, action)
            if next_state is None:
                break
            state = next_state
        return state

    def state_rank(state, rng, temperature, shape_aware):
        counts, free_rects, _, value = state
        free_area = sum(r[3] * r[4] for r in free_rects)
        remaining_value = 0.0
        max_density = 0.0
        fractional_items = []
        for i in range(n):
            if counts[i] > 0 and values[i] > 0.0:
                if shape_aware and not any(
                    lengths[i] <= r[3] + eps and widths[i] <= r[4] + eps
                    for r in free_rects
                ):
                    continue
                remaining_value += counts[i] * values[i]
                if densities[i] > max_density:
                    max_density = densities[i]
                if shape_aware:
                    fractional_items.append((densities[i], areas[i], values[i], counts[i]))
        if shape_aware:
            capacity = free_area
            fractional_value = 0.0
            for _, area, item_value, count in sorted(fractional_items, reverse=True):
                if capacity <= eps:
                    break
                take = min(float(count), capacity / area)
                fractional_value += take * item_value
                capacity -= take * area
            optimistic = min(remaining_value, fractional_value)
        else:
            optimistic = min(remaining_value, free_area * max_density)
        noise = temperature * rng.random() * max(1.0, value + optimistic) * 0.025
        return value + 0.43 * optimistic + noise

    def state_signature(state):
        counts, free_rects, _, _ = state
        shapes = tuple(sorted((round(r[3], 5), round(r[4], 5)) for r in free_rects))
        return counts, shapes

    def beam_run(seed, beam_width, temperature, shape_aware):
        nonlocal best_state, best_value
        rng = random.Random(seed)
        beam = [initial_state]
        while beam and time.time() < deadline:
            children = []
            for state in beam:
                if time.time() >= deadline:
                    break
                counts, free_rects, _, _ = state
                rect_pos, fit = choose_rectangle(free_rects, counts, temperature > 0.0, rng)
                if rect_pos < 0:
                    continue
                actions = make_actions(free_rects[rect_pos], counts, fit, rng, False)
                for action in actions:
                    if time.time() >= deadline:
                        break
                    child = apply_action(state, rect_pos, action)
                    if child is None:
                        continue
                    children.append(child)
                    if child[3] > best_value + eps:
                        best_value = child[3]
                        best_state = child
            if not children or time.time() >= deadline:
                break
            ranked = []
            for child in children:
                if time.time() >= deadline:
                    break
                ranked.append((state_rank(child, rng, temperature, shape_aware), child))
            ranked.sort(key=lambda z: z[0], reverse=True)
            seen = set()
            next_beam = []
            for _, child in ranked:
                signature = state_signature(child)
                if signature in seen:
                    continue
                seen.add(signature)
                next_beam.append(child)
                if len(next_beam) >= beam_width:
                    break
            beam = next_beam

    def materialize(state):
        _, _, actions, _ = state
        selection = [[0 for _ in range(demands[i])] for i in range(n)]
        placements = [
            [[-1.0, -1.0] for _ in range(demands[i])] for i in range(n)
        ]
        next_instance = [0 for _ in range(n)]
        leaf_ids = {}

        for action in actions:
            kind, sid, x0, y0, _, _, seq, _, _ = action
            cursor = x0 if kind in ("cv", "row") else y0
            for k, i in enumerate(seq):
                p = next_instance[i]
                if p >= demands[i]:
                    continue
                next_instance[i] += 1
                selection[i][p] = 1
                if kind in ("cv", "row"):
                    placements[i][p] = [float(cursor), float(y0)]
                    cursor += lengths[i]
                else:
                    placements[i][p] = [float(x0), float(cursor)]
                    cursor += widths[i]
                leaf_ids[(sid, k)] = [i, p]

        action_by_slot = {action[1]: action for action in actions}

        def leaf(slot, k):
            piece_index = leaf_ids.get((slot, k))
            if piece_index is None:
                return {"type": "empty"}
            return {"type": "leaf", "piece_index": piece_index}

        def row_tree(slot, seq, x0, k):
            node_leaf = leaf(slot, k)
            if k + 1 >= len(seq):
                return node_leaf
            cut = x0 + lengths[seq[k]]
            return {
                "type": "node",
                "cut_direction": "vertical",
                "cut_coordinate": float(cut),
                "left_child": node_leaf,
                "right_child": row_tree(slot, seq, cut, k + 1),
            }

        def col_tree(slot, seq, y0, k):
            node_leaf = leaf(slot, k)
            if k + 1 >= len(seq):
                return node_leaf
            cut = y0 + widths[seq[k]]
            return {
                "type": "node",
                "cut_direction": "horizontal",
                "cut_coordinate": float(cut),
                "bottom": node_leaf,
                "top": col_tree(slot, seq, cut, k + 1),
            }

        def build(slot):
            action = action_by_slot.get(slot)
            if action is None:
                return {"type": "empty"}
            kind, _, x0, y0, rw, rh, seq, used, band = action
            child_a = slot * 2 + 1
            child_b = slot * 2 + 2
            if kind in ("cv", "row"):
                packed = row_tree(slot, seq, x0, 0)
                if band < rh - eps:
                    left = {
                        "type": "node",
                        "cut_direction": "horizontal",
                        "cut_coordinate": float(y0 + band),
                        "bottom": packed,
                        "top": build(child_a),
                    }
                else:
                    left = packed
                if used < rw - eps:
                    return {
                        "type": "node",
                        "cut_direction": "vertical",
                        "cut_coordinate": float(x0 + used),
                        "left_child": left,
                        "right_child": build(child_b),
                    }
                return left

            packed = col_tree(slot, seq, y0, 0)
            if band < rw - eps:
                bottom = {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": float(x0 + band),
                    "left_child": packed,
                    "right_child": build(child_a),
                }
            else:
                bottom = packed
            if used < rh - eps:
                return {
                    "type": "node",
                    "cut_direction": "horizontal",
                    "cut_coordinate": float(y0 + used),
                    "bottom": bottom,
                    "top": build(child_b),
                }
            return bottom

        tree = build(0)
        if tree.get("type") == "node":
            tree["type"] = "root"
        return {
            "selection_status": selection,
            "placements": placements,
            "guillotine_cut_tree": tree,
        }

    try:
        # First obtain a complete constructive incumbent.  The subsequent
        # search may stop anywhere without endangering the return value.
        seed_base = int(
            sheet_l * 17.0
            + sheet_w * 31.0
            + sum((i + 3) * (lengths[i] + widths[i]) for i in range(n))
        ) & 0x7FFFFFFF
        initial = greedy_construct(seed_base, 0, True)
        if initial[3] > best_value:
            best_state = initial
            best_value = initial[3]

        variant = 1
        while time.time() < deadline:
            greedy = greedy_construct(seed_base + variant * 1009, variant % 4, False)
            if greedy[3] > best_value + eps:
                best_state = greedy
                best_value = greedy[3]
            if time.time() >= deadline:
                break
            width = 44 + (variant % 4) * 12
            if variant == 1:
                width = 48
            temperature = 0.0 if variant % 5 == 0 else 0.35 + 0.12 * (variant % 4)
            beam_run(
                seed_base + variant * 7919,
                width,
                temperature,
                variant == 1,
            )
            variant += 1
    except Exception:
        # Keep the last fully constructed incumbent on any unexpected input.
        pass

    try:
        return materialize(best_state)
    except Exception:
        return empty_solution
