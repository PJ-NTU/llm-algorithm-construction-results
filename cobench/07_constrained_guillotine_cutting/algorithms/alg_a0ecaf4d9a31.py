# Library slot 8/10  id=alg_a0ecaf4d9a31
# score=-11942.104  feasible_rate=1.0
# score_vec=[-2881.06, -2434.98, -6743.379999999999, -6473.94, -2426.6899999999996, -6664.9, -6878.5999999999985, -16144.33, -36028.060000000005, -32745.1]

import time
import random


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime beam/GRASP solver for bounded two-dimensional slicing packings."""
    t0 = time.time()
    deadline = t0 + max(0.0, float(time_limit_s)) * 0.85

    try:
        L = float(instance["sheet_length"])
        W = float(instance["sheet_width"])
        n = int(instance["num_piece_types"])
        lengths = [float(x) for x in instance["piece_lengths"]]
        widths = [float(x) for x in instance["piece_widths"]]
        demands = [max(0, int(x)) for x in instance["max_quantities"]]
        values = [float(x) for x in instance["piece_values"]]
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

    fallback = empty_solution()
    if n <= 0 or L <= 0.0 or W <= 0.0 or time.time() >= deadline:
        return fallback

    eps = 1e-9
    valid = []
    areas = [0.0] * n
    densities = [0.0] * n
    max_value = 1.0
    max_density = 1.0
    for i in range(n):
        if i >= len(lengths) or i >= len(widths) or i >= len(values):
            continue
        area = lengths[i] * widths[i]
        areas[i] = area
        if (
            demands[i] > 0
            and values[i] > 0.0
            and lengths[i] > 0.0
            and widths[i] > 0.0
            and lengths[i] <= L
            and widths[i] <= W
        ):
            valid.append(i)
            densities[i] = values[i] / area
            max_value = max(max_value, values[i])
            max_density = max(max_density, densities[i])

    if not valid:
        return fallback

    density_order = sorted(valid, key=lambda i: densities[i], reverse=True)
    value_order = sorted(valid, key=lambda i: values[i], reverse=True)
    total_copies = sum(demands[i] for i in valid)
    sheet_area = L * W

    # State: value, packed area, used-count tuple, free rectangles, linked trail.
    # A free rectangle is x, y, length, width, and its path in the slicing tree.
    initial = (0.0, 0.0, (0,) * n, ((0.0, 0.0, L, W, ()),), None)
    best_box = [initial]

    seed = n * 1000003 + total_copies * 9176
    seed += int(L * 10.0 + W * 17.0)
    for i in valid:
        seed += int((lengths[i] * 13.0 + widths[i] * 19.0 + values[i] * 23.0) * (i + 1))
    rng = random.Random(seed)

    def residuals(rect, i, orient):
        x, y, rw, rh, path = rect
        pw = lengths[i]
        ph = widths[i]
        if orient == 0:  # vertical outer cut; top of left strip, then right
            return (
                (x, y + ph, pw, rh - ph, path + (0,)),
                (x + pw, y, rw - pw, rh, path + (1,)),
            )
        return (
            (x + pw, y, rw - pw, ph, path + (0,)),
            (x, y + ph, rw, rh - ph, path + (1,)),
        )

    def apply_action(state, action):
        value, packed_area, counts, free, trail = state
        rect_index, i, orient = action
        rect = free[rect_index]
        new_counts = list(counts)
        instance_index = new_counts[i]
        new_counts[i] += 1
        children = residuals(rect, i, orient)
        new_free = list(free[:rect_index] + free[rect_index + 1 :])
        for child in children:
            if child[2] > eps and child[3] > eps:
                new_free.append(child)
        # Large regions first makes greedy construction less myopic.  Geometry,
        # rather than path, is used for ordering so equivalent states coalesce.
        new_free.sort(key=lambda r: r[2] * r[3], reverse=True)
        record = (
            trail,
            rect[4],
            i,
            instance_index,
            rect[0],
            rect[1],
            rect[2],
            rect[3],
            orient,
        )
        child_state = (
            value + values[i],
            packed_area + areas[i],
            tuple(new_counts),
            tuple(new_free),
            record,
        )
        if child_state[0] > best_box[0][0] + eps:
            best_box[0] = child_state
        return child_state

    def action_features(state, rect_index, i, orient):
        rect = state[3][rect_index]
        rw, rh = rect[2], rect[3]
        kids = residuals(rect, i, orient)
        dead_area = 0.0
        live_children = 0
        next_counts = state[2]
        for child in kids:
            cw, ch = child[2], child[3]
            if cw <= eps or ch <= eps:
                continue
            fillable = False
            for scan_index, j in enumerate(valid):
                if scan_index % 32 == 0 and time.time() >= deadline:
                    break
                remaining = demands[j] - next_counts[j] - (1 if j == i else 0)
                if remaining > 0 and lengths[j] <= cw and widths[j] <= ch:
                    fillable = True
                    break
            if fillable:
                live_children += 1
            else:
                dead_area += cw * ch

        density_score = densities[i] / max_density
        value_score = values[i] / max_value
        fill_score = areas[i] / max(eps, rw * rh)
        exact_score = 0.0
        if abs(lengths[i] - rw) <= 1e-7 * max(1.0, rw):
            exact_score += 0.5
        if abs(widths[i] - rh) <= 1e-7 * max(1.0, rh):
            exact_score += 0.5
        dead_score = dead_area / max(eps, rw * rh)
        return density_score, value_score, fill_score, exact_score, dead_score, live_children

    profiles = (
        (0.50, 0.20, 0.16, 0.12, 0.34),
        (0.28, 0.36, 0.20, 0.12, 0.30),
        (0.36, 0.18, 0.32, 0.13, 0.42),
        (0.62, 0.08, 0.10, 0.08, 0.24),
        (0.20, 0.48, 0.15, 0.10, 0.28),
    )

    def enumerate_actions(state, profile_index, noisy, action_width):
        counts = state[2]
        free = state[3]
        wd, wv, wf, we, ww = profiles[profile_index % len(profiles)]
        ranked = []
        best_density_action = None
        best_value_action = None
        best_fit_action = None
        for rect_index, rect in enumerate(free):
            if time.time() >= deadline:
                break
            rw, rh = rect[2], rect[3]
            for type_index, i in enumerate(valid):
                if type_index % 16 == 0 and time.time() >= deadline:
                    break
                if counts[i] >= demands[i]:
                    continue
                if lengths[i] > rw or widths[i] > rh:
                    continue
                for orient in (0, 1):
                    f = action_features(state, rect_index, i, orient)
                    score = wd * f[0] + wv * f[1] + wf * f[2] + we * f[3] - ww * f[4]
                    score += 0.012 * f[5]
                    if noisy:
                        score += rng.uniform(-0.075, 0.075)
                    entry = (score, rect_index, i, orient, f)
                    ranked.append(entry)
                    if best_density_action is None or f[0] > best_density_action[0]:
                        best_density_action = (f[0], entry)
                    if best_value_action is None or f[1] > best_value_action[0]:
                        best_value_action = (f[1], entry)
                    fit_key = f[2] + f[3] - f[4]
                    if best_fit_action is None or fit_key > best_fit_action[0]:
                        best_fit_action = (fit_key, entry)

        if not ranked:
            return []
        ranked.sort(key=lambda e: e[0], reverse=True)
        chosen = ranked[:action_width]
        # Preserve three qualitatively different branches even when their scalar
        # scores fall just below the beam's per-state action cut.
        if best_density_action is not None:
            chosen.append(best_density_action[1])
        if best_value_action is not None:
            chosen.append(best_value_action[1])
        if best_fit_action is not None:
            chosen.append(best_fit_action[1])
        result = []
        seen = set()
        for entry in chosen:
            action = (entry[1], entry[2], entry[3])
            if action not in seen:
                seen.add(action)
                result.append(action)
        return result

    def upper_bound(state):
        capacity = sum(r[2] * r[3] for r in state[3])
        if capacity <= eps:
            return state[0]
        potential = 0.0
        counts = state[2]
        for order_index, i in enumerate(density_order):
            if order_index % 16 == 0 and time.time() >= deadline:
                break
            copies = demands[i] - counts[i]
            if copies <= 0:
                continue
            fits = False
            for r in state[3]:
                if lengths[i] <= r[2] and widths[i] <= r[3]:
                    fits = True
                    break
            if not fits:
                continue
            possible_area = copies * areas[i]
            if possible_area <= capacity + eps:
                potential += copies * values[i]
                capacity -= possible_area
            else:
                potential += values[i] * capacity / areas[i]
                break
        return state[0] + potential

    def state_rank(state, jitter):
        ub = upper_bound(state)
        # A modest incumbent term prevents an optimistic but badly fragmented
        # state from occupying the whole beam.
        score = 0.84 * ub + 0.16 * state[0]
        if jitter > 0.0:
            score *= 1.0 + rng.uniform(-jitter, jitter)
        return score

    def greedy_construct(profile_index, random_top):
        state = initial
        while time.time() < deadline:
            actions = enumerate_actions(state, profile_index, random_top > 1, 7)
            if not actions:
                break
            if random_top > 1:
                top = min(random_top, len(actions))
                # Bias toward the best action while retaining genuine restarts.
                pick = int((rng.random() ** 2.2) * top)
                action = actions[pick]
            else:
                action = actions[0]
            state = apply_action(state, action)
        return state

    try:
        # Immediate, maximal feasible incumbent before the more expensive search.
        greedy_construct(0, 1)
        if time.time() >= deadline:
            raise TimeoutError

        # The first 62% of the caller's budget is spent on a bounded-demand beam.
        # Remaining time is converted into independent perturbed constructions.
        beam_deadline = min(deadline, t0 + max(0.0, time_limit_s) * 0.62)
        if total_copies <= 24:
            beam_width = 420
            action_width = 12
        elif total_copies <= 70:
            beam_width = 190
            action_width = 8
        else:
            beam_width = 80
            action_width = 6

        pass_index = 0
        while time.time() < beam_deadline:
            frontier = [initial]
            level = 0
            while frontier and time.time() < beam_deadline:
                children = []
                for state in frontier:
                    if time.time() >= beam_deadline:
                        break
                    actions = enumerate_actions(
                        state,
                        pass_index + level,
                        pass_index > 0,
                        action_width,
                    )
                    for action in actions:
                        if time.time() >= beam_deadline:
                            break
                        children.append(apply_action(state, action))
                if not children:
                    break

                ranked = []
                for child in children:
                    if time.time() >= beam_deadline:
                        break
                    ranked.append((state_rank(child, 0.008 * pass_index), child))
                ranked.sort(key=lambda pair: pair[0], reverse=True)

                frontier = []
                signatures = set()
                for _, child in ranked:
                    dims = tuple(sorted((round(r[2], 5), round(r[3], 5)) for r in child[3]))
                    signature = (child[2], dims)
                    if signature in signatures:
                        continue
                    signatures.add(signature)
                    frontier.append(child)
                    if len(frontier) >= beam_width:
                        break
                level += 1
            pass_index += 1

        restart = 0
        while time.time() < deadline:
            greedy_construct(restart + 1, 2 + (restart % 4))
            restart += 1
    except Exception:
        # The current incumbent is always a complete slicing packing, including
        # when a rare numeric or time-related issue interrupts improvement.
        pass

    best = best_box[0]
    try:
        selection = [[0 for _ in range(demands[i])] for i in range(n)]
        placements = [
            [[-1.0, -1.0] for _ in range(demands[i])] for i in range(n)
        ]
        decisions = {}
        trail = best[4]
        while trail is not None:
            previous, path, i, p, x, y, rw, rh, orient = trail
            decisions[path] = (i, p, x, y, rw, rh, orient)
            selection[i][p] = 1
            placements[i][p] = [float(x), float(y)]
            trail = previous

        def build_tree(path):
            decision = decisions.get(path)
            if decision is None:
                return {"type": "empty"}
            i, p, x, y, rw, rh, orient = decision
            leaf = {"type": "leaf", "piece_index": [i, p]}
            has_right_remainder = rw - lengths[i] > eps
            has_top_remainder = rh - widths[i] > eps

            if orient == 0:
                left = leaf
                if has_top_remainder:
                    left = {
                        "type": "node",
                        "cut_direction": "horizontal",
                        "cut_coordinate": float(y + widths[i]),
                        "bottom": leaf,
                        "top": build_tree(path + (0,)),
                    }
                if not has_right_remainder:
                    return left
                return {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": float(x + lengths[i]),
                    "left_child": left,
                    "right_child": build_tree(path + (1,)),
                }

            bottom = leaf
            if has_right_remainder:
                bottom = {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": float(x + lengths[i]),
                    "left_child": leaf,
                    "right_child": build_tree(path + (0,)),
                }
            if not has_top_remainder:
                return bottom
            return {
                "type": "node",
                "cut_direction": "horizontal",
                "cut_coordinate": float(y + widths[i]),
                "bottom": bottom,
                "top": build_tree(path + (1,)),
            }

        tree = build_tree(()) if decisions else {"type": "empty"}
        if tree.get("type") == "node":
            tree["type"] = "root"
        return {
            "selection_status": selection,
            "placements": placements,
            "guillotine_cut_tree": tree,
        }
    except Exception:
        return fallback
