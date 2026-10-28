# Library slot 3/10  id=alg_b83e72953e7d
# score=-12065.377  feasible_rate=1.0
# score_vec=[-2869.19, -2434.98, -6775.889999999999, -6473.94, -2426.6899999999996, -6743.29, -6878.5999999999985, -16144.33, -37025.83000000001, -32881.03]

import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime beam/GRASP solver for bounded 2-D guillotine packing."""
    t0 = time.time()

    # Leave ample time for Python/JSON overhead outside this function.  The empty
    # solution below is also the exception-safe incumbent for malformed inputs.
    try:
        n = int(instance["num_piece_types"])
        demands = [max(0, int(q)) for q in instance["max_quantities"][:n]]
    except Exception:
        n = 0
        demands = []

    empty_solution = {
        "selection_status": [[0 for _ in range(q)] for q in demands],
        "placements": [[[-1.0, -1.0] for _ in range(q)] for q in demands],
        "guillotine_cut_tree": {"type": "empty"},
    }
    best_solution = empty_solution

    try:
        budget = max(0.0, float(time_limit_s))
        deadline = t0 + budget * 0.85
        if n <= 0 or len(demands) != n or time.time() >= deadline:
            return best_solution

        sheet_l = float(instance["sheet_length"])
        sheet_w = float(instance["sheet_width"])
        lengths = [float(x) for x in instance["piece_lengths"][:n]]
        widths = [float(x) for x in instance["piece_widths"][:n]]
        values = [float(x) for x in instance["piece_values"][:n]]
        if (len(lengths) != n or len(widths) != n or len(values) != n or
                sheet_l <= 0.0 or sheet_w <= 0.0):
            return best_solution

        eps = 1e-8
        areas = [lengths[i] * widths[i] for i in range(n)]

        def fits(i, rw, rh):
            # Selection is a feasibility decision, so do not admit a piece that
            # is even slightly larger than its actual residual rectangle.
            return lengths[i] <= rw and widths[i] <= rh

        useful = [
            demands[i] > 0 and values[i] > 0.0 and lengths[i] > 0.0 and
            widths[i] > 0.0 and fits(i, sheet_l, sheet_w)
            for i in range(n)
        ]
        density_order = sorted(
            (i for i in range(n) if useful[i]),
            key=lambda i: values[i] / max(areas[i], eps),
            reverse=True,
        )
        total_items = sum(demands[i] for i in density_order)
        if total_items == 0:
            return best_solution

        # A free rectangle is (tree_node_id, x, y, length, width).  Every action
        # replaces one such leaf by two guillotine residual rectangles.  Keeping
        # this construction invariant makes every incumbent feasible by design.
        initial_rects = ((0, 0.0, 0.0, sheet_l, sheet_w),)

        def rect_bound(rw, rh, counts):
            """Fractional bounded-knapsack estimate inside one rectangle."""
            capacity = rw * rh
            estimate = 0.0
            for i in density_order:
                left = demands[i] - counts[i]
                if left <= 0 or not fits(i, rw, rh):
                    continue
                take = min(float(left), capacity / areas[i])
                if take <= 0.0:
                    break
                estimate += take * values[i]
                capacity -= take * areas[i]
                if capacity <= eps:
                    break
            return estimate

        def residuals(rect, i, orientation, child_a, child_b):
            _, x, y, rw, rh = rect
            pw = lengths[i]
            ph = widths[i]
            if orientation == 0:  # vertical first: top of piece, then right
                return (
                    (child_a, x, y + ph, pw, max(0.0, rh - ph)),
                    (child_b, x + pw, y, max(0.0, rw - pw), rh),
                )
            return (
                (child_a, x + pw, y, max(0.0, rw - pw), ph),
                (child_b, x, y + ph, rw, max(0.0, rh - ph)),
            )

        def can_use(rect):
            rw, rh = rect[3], rect[4]
            if rw <= eps or rh <= eps:
                return False
            for i in density_order:
                if fits(i, rw, rh):
                    return True
            return False

        def apply_action(counts, rects, actions, rect_index, i, orientation):
            rect = rects[rect_index]
            node_id, x, y, _, _ = rect
            p = counts[i]
            child_a = 1 + 2 * len(actions)
            child_b = child_a + 1
            ra, rb = residuals(rect, i, orientation, child_a, child_b)

            new_counts = list(counts)
            new_counts[i] += 1
            new_rects = list(rects)
            new_rects.pop(rect_index)
            if can_use(ra):
                new_rects.append(ra)
            if can_use(rb):
                new_rects.append(rb)
            action = (node_id, i, p, x, y, orientation, child_a, child_b)
            return tuple(new_counts), tuple(new_rects), actions + (action,)

        def action_future(rect, i, orientation, counts):
            """Value retained in the two residual shapes after placing i."""
            next_counts = list(counts)
            next_counts[i] += 1
            ra, rb = residuals(rect, i, orientation, -1, -2)
            future = 0.0
            if ra[3] > eps and ra[4] > eps:
                future += rect_bound(ra[3], ra[4], next_counts)
            if rb[3] > eps and rb[4] > eps:
                future += rect_bound(rb[3], rb[4], next_counts)
            old = rect_bound(rect[3], rect[4], next_counts)
            edge_fit = lengths[i] / rect[3] + widths[i] / rect[4]
            return future - old + 0.08 * values[i] * edge_fit

        def build_solution(actions):
            action_by_node = {a[0]: a for a in actions}

            def make_node(node_id, region, is_root=False):
                action = action_by_node.get(node_id)
                if action is None:
                    return {"type": "empty"}
                _, i, p, x, y, orientation, child_a, child_b = action
                x0, y0, x1, y1 = region
                cut_x = float(x + lengths[i])
                cut_y = float(y + widths[i])
                leaf = {"type": "leaf", "piece_index": [i, p]}
                if orientation == 0:
                    if y0 < cut_y < y1:
                        left = {
                            "type": "node",
                            "cut_direction": "horizontal",
                            "cut_coordinate": cut_y,
                            "bottom": leaf,
                            "top": make_node(
                                child_a, (x0, cut_y, cut_x, y1)
                            ),
                        }
                    elif cut_y == y1 and child_a not in action_by_node:
                        # The piece reaches the region boundary, so there is no
                        # top rectangle and therefore no horizontal cut to make.
                        left = leaf
                    else:
                        raise ValueError("invalid horizontal residual")

                    if x0 < cut_x < x1:
                        node = {
                            "type": "node",
                            "cut_direction": "vertical",
                            "cut_coordinate": cut_x,
                            "left_child": left,
                            "right_child": make_node(
                                child_b, (cut_x, y0, x1, y1)
                            ),
                        }
                    elif cut_x == x1 and child_b not in action_by_node:
                        # The piece-width strip reaches the right boundary.
                        node = left
                    else:
                        raise ValueError("invalid vertical residual")
                else:
                    if x0 < cut_x < x1:
                        bottom = {
                            "type": "node",
                            "cut_direction": "vertical",
                            "cut_coordinate": cut_x,
                            "left_child": leaf,
                            "right_child": make_node(
                                child_a, (cut_x, y0, x1, cut_y)
                            ),
                        }
                    elif cut_x == x1 and child_a not in action_by_node:
                        # The piece reaches the right edge of the bottom strip.
                        bottom = leaf
                    else:
                        raise ValueError("invalid vertical residual")

                    if y0 < cut_y < y1:
                        node = {
                            "type": "node",
                            "cut_direction": "horizontal",
                            "cut_coordinate": cut_y,
                            "bottom": bottom,
                            "top": make_node(
                                child_b, (x0, cut_y, x1, y1)
                            ),
                        }
                    elif cut_y == y1 and child_b not in action_by_node:
                        # The bottom strip reaches the top boundary.
                        node = bottom
                    else:
                        raise ValueError("invalid horizontal residual")
                if is_root and node.get("type") == "node":
                    node["type"] = "root"
                return node

            status = [[0 for _ in range(q)] for q in demands]
            placements = [[[-1.0, -1.0] for _ in range(q)] for q in demands]
            for action in actions:
                _, i, p, x, y, _, _, _ = action
                status[i][p] = 1
                placements[i][p] = [float(x), float(y)]
            tree = (
                make_node(0, (0.0, 0.0, sheet_l, sheet_w), True)
                if actions else {"type": "empty"}
            )
            return {
                "selection_status": status,
                "placements": placements,
                "guillotine_cut_tree": tree,
            }

        best_value = 0.0
        best_actions = ()

        def save_if_better(value, actions):
            nonlocal best_value, best_actions, best_solution
            if value > best_value + 1e-9:
                best_value = value
                best_actions = actions
                best_solution = build_solution(actions)

        def best_placement(rects, counts, i, random_tie=0.0):
            choice = None
            choice_score = -float("inf")
            for r_idx, rect in enumerate(rects):
                if time.time() >= deadline:
                    break
                if not fits(i, rect[3], rect[4]):
                    continue
                for orientation in (0, 1):
                    score = action_future(rect, i, orientation, counts)
                    # Prefer using a tight region, preserving large versatile ones.
                    fill = areas[i] / max(rect[3] * rect[4], eps)
                    score += values[i] * 0.12 * fill
                    if random_tie:
                        score += random_tie * values[i] * random.random()
                    if score > choice_score:
                        choice_score = score
                        choice = (r_idx, orientation)
            return choice

        def construct_from_order(order, random_tie=0.0):
            counts = tuple(0 for _ in range(n))
            rects = initial_rects
            actions = ()
            value = 0.0
            for i in order:
                if time.time() >= deadline:
                    break
                if counts[i] >= demands[i]:
                    continue
                choice = best_placement(rects, counts, i, random_tie)
                if choice is None:
                    continue
                counts, rects, actions = apply_action(
                    counts, rects, actions, choice[0], i, choice[1]
                )
                value += values[i]
                save_if_better(value, actions)
            return value, actions

        # Fast deterministic incumbents.  These include the dominant density
        # heuristic but also value-heavy and dimension-aware orderings.
        alphas = (1.0, 0.75, 0.5, 0.25, 0.0, 1.2)
        seed_orders = []
        for alpha in alphas:
            ranked = sorted(
                density_order,
                key=lambda i: values[i] / max(areas[i] ** alpha, eps),
                reverse=True,
            )
            seed_orders.append(tuple(i for i in ranked for _ in range(demands[i])))
        seed_orders.append(tuple(
            i for i in sorted(density_order, key=lambda j: (widths[j], values[j]), reverse=True)
            for _ in range(demands[i])
        ))
        seed_orders.append(tuple(
            i for i in sorted(density_order, key=lambda j: (lengths[j], values[j]), reverse=True)
            for _ in range(demands[i])
        ))
        for order in seed_orders:
            if time.time() >= deadline:
                return best_solution
            construct_from_order(order)

        # A bounded beam retains geometrically different partial layouts.  The
        # state rank is an optimistic value bound; equivalent residual shapes and
        # demand vectors are merged, regardless of their absolute coordinates.
        beam_stop = min(deadline, t0 + budget * 0.30)

        def state_upper(value, counts, rects):
            total_area = sum(r[3] * r[4] for r in rects)
            if total_area <= eps:
                return value
            global_extra = 0.0
            capacity = total_area
            for i in density_order:
                left = demands[i] - counts[i]
                if left <= 0:
                    continue
                if not any(fits(i, r[3], r[4]) for r in rects):
                    continue
                take = min(float(left), capacity / areas[i])
                global_extra += take * values[i]
                capacity -= take * areas[i]
                if capacity <= eps:
                    break
            shape_extra = sum(rect_bound(r[3], r[4], counts) for r in rects)
            return value + min(global_extra, shape_extra)

        if total_items <= 12:
            beam_width = 1000
            branch_width = 1000
        elif total_items <= 28:
            beam_width = 220
            branch_width = 14
        else:
            beam_width = 90
            branch_width = 8

        beam = [(0.0, tuple(0 for _ in range(n)), initial_rects, ())]
        while beam and time.time() < beam_stop:
            children = {}
            for value, counts, rects, actions in beam:
                if time.time() >= beam_stop:
                    break
                options = []
                for r_idx, rect in enumerate(rects):
                    for i in density_order:
                        if counts[i] >= demands[i]:
                            continue
                        if not fits(i, rect[3], rect[4]):
                            continue
                        base = math.log(max(values[i], eps)) - 0.65 * math.log(max(areas[i], eps))
                        for orientation in (0, 1):
                            look = action_future(rect, i, orientation, counts)
                            rank = base + 0.22 * look / max(values[i], eps)
                            rank += 0.08 * areas[i] / max(rect[3] * rect[4], eps)
                            options.append((rank, r_idx, i, orientation))
                if not options:
                    continue
                options.sort(reverse=True)
                chosen_options = options[:branch_width]
                if branch_width < len(options):
                    # Density-only branching recreates the incumbent family too
                    # closely.  Reserve a few branches for high absolute value,
                    # tight fits, and large pieces.
                    diverse = (
                        max(options, key=lambda o: values[o[2]]),
                        max(options, key=lambda o: values[o[2]] / max(math.sqrt(areas[o[2]]), eps)),
                        max(options, key=lambda o: areas[o[2]] / max(rects[o[1]][3] * rects[o[1]][4], eps)),
                    )
                    present = {(o[1], o[2], o[3]) for o in chosen_options}
                    for option in diverse:
                        marker = (option[1], option[2], option[3])
                        if marker not in present:
                            present.add(marker)
                            chosen_options.append(option)
                for _, r_idx, i, orientation in chosen_options:
                    if time.time() >= beam_stop:
                        break
                    nc, nr, na = apply_action(counts, rects, actions, r_idx, i, orientation)
                    nv = value + values[i]
                    save_if_better(nv, na)
                    shape_key = tuple(sorted((round(r[3], 5), round(r[4], 5)) for r in nr))
                    key = (nc, shape_key)
                    rank = state_upper(nv, nc, nr)
                    old = children.get(key)
                    if old is None or rank > old[0]:
                        children[key] = (rank, nv, nc, nr, na)
            if not children:
                break
            ranked_children = sorted(children.values(), key=lambda s: (s[0], s[1]), reverse=True)
            beam = [(s[1], s[2], s[3], s[4]) for s in ranked_children[:beam_width]]

        # Online GRASP chooses the type, residual region, and cut direction
        # together.  Parameter randomization supplies the O5 diversity; elite
        # order perturbations intensify around the best slicing patterns found.
        seed_value = 1469598103934665603
        for i in range(n):
            seed_value ^= int(abs(lengths[i]) * 1000.0) + 31 * int(abs(widths[i]) * 1000.0)
            seed_value = (seed_value * 1099511628211) & ((1 << 64) - 1)
        rng = random.Random(seed_value)

        def grasp_construct(alpha, noise, rcl_size, future_weight):
            counts = tuple(0 for _ in range(n))
            rects = initial_rects
            actions = ()
            value = 0.0
            while time.time() < deadline:
                candidates = []
                for i in density_order:
                    if counts[i] >= demands[i]:
                        continue
                    best = None
                    best_score = -float("inf")
                    for r_idx, rect in enumerate(rects):
                        if not fits(i, rect[3], rect[4]):
                            continue
                        for orientation in (0, 1):
                            look = action_future(rect, i, orientation, counts)
                            fill = areas[i] / max(rect[3] * rect[4], eps)
                            score = (
                                math.log(max(values[i], eps))
                                - alpha * math.log(max(areas[i], eps))
                                + future_weight * look / max(values[i], eps)
                                + 0.15 * fill
                            )
                            if score > best_score:
                                best_score = score
                                best = (r_idx, orientation)
                    if best is not None:
                        jitter = noise * (rng.random() - 0.5)
                        candidates.append((best_score + jitter, i, best[0], best[1]))
                    if time.time() >= deadline:
                        break
                if not candidates or time.time() >= deadline:
                    break
                candidates.sort(reverse=True)
                k = min(rcl_size, len(candidates))
                # Bias the RCL draw toward its best member without eliminating
                # the deliberately different lower-ranked choices.
                pick = int((rng.random() ** 2.0) * k)
                _, i, r_idx, orientation = candidates[pick]
                counts, rects, actions = apply_action(
                    counts, rects, actions, r_idx, i, orientation
                )
                value += values[i]
                save_if_better(value, actions)
            return value, actions

        def fast_recursive_construct(priority, top_k, split_mode, source_rng=None):
            """Very low-overhead corner recursion used for massive restarts."""
            if source_rng is None:
                source_rng = rng
            counts = [0 for _ in range(n)]
            actions = []
            next_node = [1]
            value_box = [0.0]

            def fill(node_id, x, y, rw, rh, depth):
                if time.time() >= deadline:
                    return
                fitting = []
                for i in priority:
                    if counts[i] < demands[i] and fits(i, rw, rh):
                        fitting.append(i)
                        if len(fitting) >= top_k:
                            break
                if not fitting:
                    return
                i = fitting[source_rng.randrange(len(fitting))]
                pw, ph = lengths[i], widths[i]

                if split_mode == 0 or split_mode == 5:
                    orientation = 0 if source_rng.random() < 0.5 else 1
                elif split_mode == 6:
                    orientation = 1  # horizontal shelves
                elif split_mode == 7:
                    orientation = 0  # vertical strips
                elif split_mode == 1:
                    orientation = 0 if rw - pw >= rh - ph else 1
                    if source_rng.random() < 0.12:
                        orientation = 1 - orientation
                elif split_mode == 2:
                    # Keep the two residual rectangles as compact as possible.
                    def aspect(a, b):
                        if a <= eps or b <= eps:
                            return 0.0
                        return max(a / b, b / a)
                    vertical_bad = aspect(pw, rh - ph) + aspect(rw - pw, rh)
                    horizontal_bad = aspect(rw - pw, ph) + aspect(rw, rh - ph)
                    orientation = 0 if vertical_bad <= horizontal_bad else 1
                elif split_mode == 3:
                    # Match a piece edge to the tighter side of the region.
                    orientation = 0 if pw / rw >= ph / rh else 1
                else:
                    # Standard guillotine short-leftover rule.
                    orientation = 0 if rw - pw <= rh - ph else 1

                p = counts[i]
                counts[i] += 1
                value_box[0] += values[i]
                child_a = next_node[0]
                child_b = child_a + 1
                next_node[0] += 2
                actions.append((node_id, i, p, x, y, orientation, child_a, child_b))

                if orientation == 0:
                    first = (child_a, x, y + ph, pw, rh - ph)
                    second = (child_b, x + pw, y, rw - pw, rh)
                else:
                    first = (child_a, x + pw, y, rw - pw, ph)
                    second = (child_b, x, y + ph, rw, rh - ph)

                # Demand is shared across both children.  Alternating their visit
                # order avoids systematically starving either residual shape.
                if split_mode not in (5, 6, 7) and (depth + split_mode) & 1:
                    first, second = second, first
                if first[3] > eps and first[4] > eps:
                    fill(*first, depth + 1)
                if second[3] > eps and second[4] > eps:
                    fill(*second, depth + 1)

            fill(0, 0.0, 0.0, sheet_l, sheet_w, 0)
            frozen_actions = tuple(actions)
            save_if_better(value_box[0], frozen_actions)
            return value_box[0], frozen_actions

        def baseline_score_construct(priority, top_k):
            """Score a baseline restart cheaply; rebuild only improvements."""
            counts = [0 for _ in range(n)]
            state_before = baseline_rng.getstate()

            def score_region(rw, rh):
                if time.time() >= deadline:
                    return 0.0
                fitting = []
                for i in priority:
                    if counts[i] < demands[i] and fits(i, rw, rh):
                        fitting.append(i)
                        if len(fitting) >= top_k:
                            break
                if not fitting:
                    return 0.0
                i = fitting[baseline_rng.randrange(len(fitting))]
                counts[i] += 1
                if baseline_rng.random() < 0.5:
                    return (
                        values[i]
                        + score_region(lengths[i], rh - widths[i])
                        + score_region(rw - lengths[i], rh)
                    )
                return (
                    values[i]
                    + score_region(rw - lengths[i], widths[i])
                    + score_region(rw, rh - widths[i])
                )

            value = score_region(sheet_l, sheet_w)
            state_after = baseline_rng.getstate()
            if value > best_value + 1e-9 and time.time() < deadline:
                baseline_rng.setstate(state_before)
                fast_recursive_construct(priority, top_k, 5, baseline_rng)
                baseline_rng.setstate(state_after)
            return value

        elite_order = tuple(a[1] for a in best_actions)
        all_items = tuple(i for i in density_order for _ in range(demands[i]))
        current_order = elite_order + all_items
        current_value, current_actions = construct_from_order(current_order)
        baseline_rng = random.Random(4)
        baseline_iteration = 0
        baseline_seen = 0.0

        # Give the incumbent family a guaranteed continuous time slice.  A
        # probabilistic mix is misleading here because one expensive rollout can
        # otherwise consume the time intended for tens of thousands of restarts.
        baseline_stop = min(deadline, t0 + budget * 0.68)
        while time.time() < baseline_stop:
            priority = list(density_order)
            if baseline_iteration > 0:
                if baseline_rng.random() < 0.30:
                    baseline_rng.shuffle(priority)
                else:
                    for _ in range(int(n * 0.1) + 1):
                        if len(priority) >= 2:
                            a = baseline_rng.randrange(len(priority))
                            b = baseline_rng.randrange(len(priority))
                            priority[a], priority[b] = priority[b], priority[a]
            top_k = min(len(priority), 1 + baseline_iteration % 5)
            baseline_seen = max(baseline_seen, baseline_score_construct(priority, top_k))
            baseline_iteration += 1

        while time.time() < deadline:
            phase = rng.random()
            if phase < 0.34:
                style = rng.random()
                if style < 0.48:
                    priority = list(density_order)
                    swaps = 1 + int(rng.random() * max(1, n // 8))
                    for _ in range(swaps):
                        if len(priority) >= 2:
                            a = rng.randrange(len(priority))
                            b = rng.randrange(len(priority))
                            priority[a], priority[b] = priority[b], priority[a]
                elif style < 0.83:
                    alpha = rng.uniform(0.0, 1.25)
                    temperature = rng.uniform(0.0, 0.45)
                    priority = sorted(
                        density_order,
                        key=lambda i: (
                            math.log(max(values[i], eps))
                            - alpha * math.log(max(areas[i], eps))
                            + temperature * (rng.random() - 0.5)
                        ),
                        reverse=True,
                    )
                elif style < 0.93:
                    priority = list(density_order)
                    rng.shuffle(priority)
                else:
                    # Intensify around the current incumbent's type sequence.
                    seen = set()
                    priority = []
                    for action in best_actions:
                        i = action[1]
                        if i not in seen:
                            seen.add(i)
                            priority.append(i)
                    priority.extend(i for i in density_order if i not in seen)
                    if len(priority) >= 2:
                        a = rng.randrange(len(priority))
                        b = rng.randrange(len(priority))
                        priority[a], priority[b] = priority[b], priority[a]
                top_k = 1 + rng.randrange(min(5, max(1, len(priority))))
                fast_recursive_construct(priority, top_k, rng.randrange(5))
            elif phase < 0.61:
                # Explicit shelf/strip patterns are a materially different
                # slicing family and are especially useful for long thin items.
                horizontal = rng.random() < 0.5
                alpha = rng.uniform(0.45, 1.05)
                if horizontal:
                    priority = sorted(
                        density_order,
                        key=lambda i: (
                            widths[i],
                            values[i] / max(areas[i] ** alpha, eps),
                        ),
                        reverse=True,
                    )
                    split_mode = 6
                else:
                    priority = sorted(
                        density_order,
                        key=lambda i: (
                            lengths[i],
                            values[i] / max(areas[i] ** alpha, eps),
                        ),
                        reverse=True,
                    )
                    split_mode = 7
                # Light noise changes shelf composition while preserving the
                # monotone dimension bias needed for clean strips.
                for _ in range(1 + int(rng.random() * max(1, n // 10))):
                    if len(priority) >= 2:
                        a = rng.randrange(len(priority))
                        b = rng.randrange(len(priority))
                        priority[a], priority[b] = priority[b], priority[a]
                fast_recursive_construct(priority, 1 + rng.randrange(min(3, len(priority))), split_mode)
            elif phase < 0.78:
                alpha = rng.uniform(0.05, 1.25)
                noise = rng.uniform(0.0, 1.4)
                rcl = 1 + int(rng.random() * min(6, max(1, len(density_order))))
                future_weight = rng.uniform(0.05, 0.5)
                value, actions = grasp_construct(alpha, noise, rcl, future_weight)
                if value >= current_value or rng.random() < 0.08:
                    current_value, current_actions = value, actions
                    selected = tuple(a[1] for a in actions)
                    current_order = selected + all_items
            else:
                # Variable-neighbourhood moves on an elite-derived item order.
                candidate_order = list(current_order)
                m = len(candidate_order)
                if m >= 2:
                    move = rng.random()
                    a = rng.randrange(m)
                    b = rng.randrange(m)
                    if move < 0.45:
                        candidate_order[a], candidate_order[b] = candidate_order[b], candidate_order[a]
                    elif move < 0.8:
                        item = candidate_order.pop(a)
                        candidate_order.insert(b, item)
                    else:
                        lo, hi = sorted((a, b))
                        candidate_order[lo:hi + 1] = reversed(candidate_order[lo:hi + 1])
                value, actions = construct_from_order(candidate_order, random_tie=0.025)
                if value >= current_value or rng.random() < 0.025:
                    current_value = value
                    current_actions = actions
                    current_order = tuple(candidate_order)

        print("DEBUG_BASELINE", baseline_iteration, baseline_seen, best_value)
        return best_solution
    except Exception:
        # The incumbent is rebuilt on every strict improvement, so even an
        # unexpected numerical corner case returns a complete feasible object.
        return best_solution
