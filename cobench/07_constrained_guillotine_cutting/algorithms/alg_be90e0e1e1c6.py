# Library slot 4/10  id=alg_be90e0e1e1c6
# score=-12056.810999999998  feasible_rate=1.0
# score_vec=[-2867.67, -2434.98, -6743.379999999999, -6473.94, -2426.6899999999996, -6743.29, -6878.5999999999985, -16144.33, -36718.310000000005, -33136.91999999999]

"""Anytime heuristic for demand-constrained 2-D guillotine cutting stock.

The search mixes two different slicing-layout neighborhoods:

* demand-aware best-first expansion of all current free leaves;
* horizontal/vertical strip patterns whose one-dimensional contents are found
  with a bounded knapsack dynamic program.

Every incumbent is represented by its slicing construction, so producing the
required cut tree is exact rather than a post-processing guess.
"""

import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    deadline = t0 + max(0.0, float(time_limit_s)) * 0.85

    # Parse only the declared schema.  Keeping an empty incumbent from the
    # outset guarantees a structurally complete return even under a tiny limit.
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

    n = min(n, len(lengths), len(widths), len(demands), len(values))

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

    if n <= 0 or L <= 0.0 or W <= 0.0:
        return best

    eps = 1e-9 * max(1.0, L, W)
    valid_types = [
        i
        for i in range(n)
        if demands[i] > 0
        and values[i] > 0.0
        and lengths[i] > 0.0
        and widths[i] > 0.0
        and lengths[i] <= L
        and widths[i] <= W
    ]
    if not valid_types:
        return best

    stock_area = L * W
    densities = [0.0 for _ in range(n)]
    for i in valid_types:
        densities[i] = values[i] / (lengths[i] * widths[i])
    max_density = max(densities[i] for i in valid_types)

    # Stable instance-derived seed: repeatable, but different exams explore
    # different randomized neighborhoods.
    seed = n * 1000003 + int(L * 10.0) * 9176 + int(W * 10.0) * 131
    for i in valid_types:
        seed = (
            seed * 1000003
            + int(lengths[i] * 10.0) * 37
            + int(widths[i] * 10.0) * 101
            + int(values[i] * 100.0)
        ) & 0xFFFFFFFFFFFFFFFF
    rng = random.Random(seed)

    def make_solution(placed, tree):
        selection = [[0 for _ in range(demands[i])] for i in range(n)]
        coordinates = [
            [[-1.0, -1.0] for _ in range(demands[i])] for i in range(n)
        ]
        for i, p, x, y in placed:
            if 0 <= i < n and 0 <= p < demands[i]:
                selection[i][p] = 1
                coordinates[i][p] = [float(x), float(y)]
        if tree and tree.get("type") == "node":
            tree["type"] = "root"
        return {
            "selection_status": selection,
            "placements": coordinates,
            "guillotine_cut_tree": tree if tree else {"type": "empty"},
        }

    def consider(value, placed, tree):
        nonlocal best, best_value
        if value > best_value + 1e-8:
            best = make_solution(placed, tree)
            best_value = float(value)

    def future_potential(a, b, remaining):
        """Cheap optimistic value of one free leaf, used only for ranking."""
        if a <= eps or b <= eps:
            return 0.0
        area = a * b
        dens = 0.0
        available_value = 0.0
        largest = 0.0
        for j in valid_types:
            if remaining[j] <= 0:
                continue
            if lengths[j] <= a and widths[j] <= b:
                if densities[j] > dens:
                    dens = densities[j]
                amount = values[j] * remaining[j]
                available_value += amount
                if values[j] > largest:
                    largest = values[j]
        if dens <= 0.0:
            return 0.0
        # The density bound values usable area; the availability bound prevents
        # a rare high-density type from being treated as unlimited.
        return min(area * dens, available_value) + 0.08 * largest

    def build_tree(actions, leaf_id):
        action = actions.get(leaf_id)
        if action is None:
            return {"type": "empty"}
        orientation, i, p, x, y, li, wi, first_id, second_id = action
        piece_leaf = {"type": "leaf", "piece_index": [i, p]}
        if orientation == 0:
            left = piece_leaf
            if first_id is not None:
                left = {
                    "type": "node",
                    "cut_direction": "horizontal",
                    "cut_coordinate": float(y + wi),
                    "bottom": piece_leaf,
                    "top": build_tree(actions, first_id),
                }
            if second_id is None:
                return left
            return {
                "type": "node",
                "cut_direction": "vertical",
                "cut_coordinate": float(x + li),
                "left_child": left,
                "right_child": build_tree(actions, second_id),
            }
        bottom = piece_leaf
        if first_id is not None:
            bottom = {
                "type": "node",
                "cut_direction": "vertical",
                "cut_coordinate": float(x + li),
                "left_child": piece_leaf,
                "right_child": build_tree(actions, first_id),
            }
        if second_id is None:
            return bottom
        return {
            "type": "node",
            "cut_direction": "horizontal",
            "cut_coordinate": float(y + wi),
            "bottom": bottom,
            "top": build_tree(actions, second_id),
        }

    def leaf_constructor(config, stop_at):
        """Globally select a free leaf, piece, and one of two slicing cuts."""
        remaining = list(demands[:n])
        used = [0 for _ in range(n)]
        free_rects = [(0, 0.0, 0.0, L, W)]
        actions = {}
        placed = []
        total = 0.0
        next_id = 1

        density_weight, lookahead, snug_weight, temperature, elite = config
        type_bias = [1.0 for _ in range(n)]
        for i in valid_types:
            jitter = rng.uniform(-temperature, temperature)
            type_bias[i] = max(0.35, 1.0 + jitter)
            if elite is not None and elite[i] > 0:
                type_bias[i] *= 1.0 + 0.10 * rng.random()

        while free_rects and time.time() < stop_at:
            candidates = []
            for rect_pos, rect in enumerate(free_rects):
                if time.time() >= stop_at:
                    break
                fid, x, y, a, b = rect
                parent_area = a * b
                for i in valid_types:
                    if remaining[i] <= 0:
                        continue
                    li = lengths[i]
                    wi = widths[i]
                    if li > a or wi > b:
                        continue

                    remaining[i] -= 1
                    if lookahead > 1e-12:
                        parent_future = future_potential(a, b, remaining)
                    else:
                        parent_future = 0.0
                    rho_part = densities[i] / max_density if max_density > 0 else 0.0
                    snug = max(li / a, wi / b)
                    fill = (li * wi) / parent_area

                    # Vertical-first: a narrow upper leaf plus a full-height
                    # right leaf.  Horizontal-first is the transposed topology.
                    for orientation in (0, 1):
                        if orientation == 0:
                            children = ((li, b - wi), (a - li, b))
                        else:
                            children = ((a - li, wi), (a, b - wi))
                        future = 0.0
                        live_children = 0
                        if lookahead > 1e-12:
                            for ca, cb in children:
                                q = future_potential(ca, cb, remaining)
                                future += q
                                if q > 0.0:
                                    live_children += 1

                        score = values[i] * type_bias[i]
                        score *= 1.0 + density_weight * rho_part
                        score += lookahead * (future - parent_future)
                        score += values[i] * snug_weight * (0.65 * snug + 0.35 * fill)
                        # A tiny preference for retaining two useful branches;
                        # it breaks ties without forcing fragmentation.
                        score += values[i] * 0.012 * live_children
                        if temperature > 0.0:
                            score *= 1.0 + rng.uniform(-0.12, 0.12) * temperature
                        candidates.append((score, rect_pos, i, orientation))
                    remaining[i] += 1

            if not candidates:
                break
            candidates.sort(key=lambda z: z[0], reverse=True)
            rcl = min(len(candidates), 1 + int(temperature * 9.0))
            if rcl <= 1 or rng.random() > temperature * 0.55:
                chosen = candidates[0]
            else:
                # Squared bias keeps diversification close to the greedy front.
                chosen = candidates[int((rng.random() ** 2) * rcl)]

            _, rect_pos, i, orientation = chosen
            fid, x, y, a, b = free_rects.pop(rect_pos)
            li = lengths[i]
            wi = widths[i]
            p = used[i]
            used[i] += 1
            remaining[i] -= 1
            total += values[i]
            placed.append((i, p, x, y))

            first_id = None
            second_id = None
            new_rects = []
            if orientation == 0:
                if b - wi > eps:
                    first_id = next_id
                    next_id += 1
                    new_rects.append((first_id, x, y + wi, li, b - wi))
                if a - li > eps:
                    second_id = next_id
                    next_id += 1
                    new_rects.append((second_id, x + li, y, a - li, b))
            else:
                if a - li > eps:
                    first_id = next_id
                    next_id += 1
                    new_rects.append((first_id, x + li, y, a - li, wi))
                if b - wi > eps:
                    second_id = next_id
                    next_id += 1
                    new_rects.append((second_id, x, y + wi, a, b - wi))
            free_rects.extend(new_rects)
            actions[fid] = (orientation, i, p, x, y, li, wi, first_id, second_id)

        return total, placed, build_tree(actions, 0)

    def beam_constructor(stop_at):
        """Keep several competing partial slicing trees before committing.

        A canonical promising leaf is expanded in each state.  The explicit
        discard branch is important: it lets the beam abandon a bad sliver and
        continue elsewhere without inserting a low-value blocking piece.
        """

        def rank_state(value, free_rects, remaining):
            future = 0.0
            for _, _, _, a, b in free_rects:
                future += future_potential(a, b, remaining)
            remaining_value = 0.0
            for i in valid_types:
                remaining_value += values[i] * remaining[i]
            return value + 0.50 * min(future, remaining_value)

        initial_remaining = tuple(demands[:n])
        # State fields: value, remaining, used, free, actions, placed, next id.
        beam = [
            (
                0.0,
                initial_remaining,
                tuple(0 for _ in range(n)),
                ((0, 0.0, 0.0, L, W),),
                {},
                (),
                1,
            )
        ]
        winning = beam[0]
        width = 34 if n <= 60 else 20

        while beam and time.time() < stop_at:
            generated = []
            any_placement = False
            for state in beam:
                if time.time() >= stop_at:
                    break
                value, rem_tuple, used_tuple, free_tuple, actions, placed, next_id = state
                if value > winning[0] + 1e-8:
                    winning = state
                if not free_tuple:
                    continue
                remaining = list(rem_tuple)

                # Expand the leaf with the most immediate opportunity.  This
                # removes action-order duplicates from independent branches.
                rect_pos = max(
                    range(len(free_tuple)),
                    key=lambda q: future_potential(
                        free_tuple[q][3], free_tuple[q][4], remaining
                    ),
                )
                fid, x, y, a, b = free_tuple[rect_pos]
                other_free = list(free_tuple)
                other_free.pop(rect_pos)

                # The empty decision is a genuine beam branch, not merely a
                # fallback, because reserving demand for another leaf can win.
                if other_free:
                    discard = (
                        value,
                        rem_tuple,
                        used_tuple,
                        tuple(other_free),
                        actions,
                        placed,
                        next_id,
                    )
                    generated.append((rank_state(value, other_free, remaining), discard))

                parent_potential = future_potential(a, b, remaining)
                for i in valid_types:
                    if remaining[i] <= 0:
                        continue
                    li = lengths[i]
                    wi = widths[i]
                    if li > a or wi > b:
                        continue
                    any_placement = True
                    new_remaining = list(remaining)
                    new_remaining[i] -= 1
                    new_used = list(used_tuple)
                    p = new_used[i]
                    new_used[i] += 1
                    for orientation in (0, 1):
                        first_id = None
                        second_id = None
                        child_rects = list(other_free)
                        fresh_id = next_id
                        if orientation == 0:
                            if b - wi > eps:
                                first_id = fresh_id
                                fresh_id += 1
                                child_rects.append((first_id, x, y + wi, li, b - wi))
                            if a - li > eps:
                                second_id = fresh_id
                                fresh_id += 1
                                child_rects.append((second_id, x + li, y, a - li, b))
                        else:
                            if a - li > eps:
                                first_id = fresh_id
                                fresh_id += 1
                                child_rects.append((first_id, x + li, y, a - li, wi))
                            if b - wi > eps:
                                second_id = fresh_id
                                fresh_id += 1
                                child_rects.append((second_id, x, y + wi, a, b - wi))

                        new_actions = actions.copy()
                        new_actions[fid] = (
                            orientation,
                            i,
                            p,
                            x,
                            y,
                            li,
                            wi,
                            first_id,
                            second_id,
                        )
                        new_value = value + values[i]
                        child_future = 0.0
                        for _, _, _, ca, cb in child_rects:
                            child_future += future_potential(ca, cb, new_remaining)
                        # Rank primarily by an incumbent-compatible bound, with
                        # a small marginal-flexibility tie breaker.
                        rank = rank_state(new_value, child_rects, new_remaining)
                        rank += 0.04 * (child_future - parent_potential)
                        new_state = (
                            new_value,
                            tuple(new_remaining),
                            tuple(new_used),
                            tuple(child_rects),
                            new_actions,
                            placed + ((i, p, x, y),),
                            fresh_id,
                        )
                        generated.append((rank, new_state))

            if not generated or not any_placement:
                break
            generated.sort(key=lambda z: z[0], reverse=True)
            beam = [z[1] for z in generated[:width]]

        for state in beam:
            if state[0] > winning[0] + 1e-8:
                winning = state
        value, _, _, _, actions, placed, _ = winning
        return value, list(placed), build_tree(actions, 0)

    def bounded_line_pattern(capacity, line_dims, cross_dims, remaining, height, noise, stop_at):
        """Conservative discretized bounded knapsack for one strip."""
        units = 520
        dp = [-1.0] * (units + 1)
        paths = [None] * (units + 1)
        dp[0] = 0.0
        paths[0] = ()
        for i in valid_types:
            if time.time() >= stop_at:
                return ()
            if remaining[i] <= 0 or cross_dims[i] > height or line_dims[i] > capacity:
                continue
            weight = max(1, int(math.ceil(line_dims[i] * units / capacity - 1e-12)))
            adjusted = values[i] * (1.0 + noise[i])
            for _ in range(remaining[i]):
                for c in range(units, weight - 1, -1):
                    old = dp[c - weight]
                    if old >= 0.0 and old + adjusted > dp[c] + 1e-12:
                        dp[c] = old + adjusted
                        paths[c] = paths[c - weight] + (i,)
        best_c = max(range(units + 1), key=lambda c: dp[c])
        return paths[best_c] or ()

    def strip_tree(strips, horizontal):
        """Build the exact two-stage slicing tree for a strip layout."""
        if not strips:
            return {"type": "empty"}

        def within_strip(items, base):
            if not items:
                return {"type": "empty"}
            i, p, x, y = items[0]
            leaf = {"type": "leaf", "piece_index": [i, p]}
            if len(items) == 1:
                return leaf
            rest = within_strip(items[1:], base)
            if horizontal:
                return {
                    "type": "node",
                    "cut_direction": "vertical",
                    "cut_coordinate": float(x + lengths[i]),
                    "left_child": leaf,
                    "right_child": rest,
                }
            return {
                "type": "node",
                "cut_direction": "horizontal",
                "cut_coordinate": float(y + widths[i]),
                "bottom": leaf,
                "top": rest,
            }

        first = strips[0]
        subtree = within_strip(first[1], first[0])
        if len(strips) == 1:
            return subtree
        rest = strip_tree(strips[1:], horizontal)
        if horizontal:
            return {
                "type": "node",
                "cut_direction": "horizontal",
                "cut_coordinate": float(first[0] + first[2]),
                "bottom": subtree,
                "top": rest,
            }
        return {
            "type": "node",
            "cut_direction": "vertical",
            "cut_coordinate": float(first[0] + first[2]),
            "left_child": subtree,
            "right_child": rest,
        }

    def strip_constructor(horizontal, temperature, stop_at):
        """Pack optimized one-dimensional patterns into successive strips."""
        if horizontal:
            line_capacity = L
            cross_capacity = W
            line_dims = lengths
            cross_dims = widths
        else:
            line_capacity = W
            cross_capacity = L
            line_dims = widths
            cross_dims = lengths

        remaining = list(demands[:n])
        used = [0 for _ in range(n)]
        cross_used = 0.0
        total = 0.0
        placed = []
        strips = []

        while cross_used < cross_capacity - eps and time.time() < stop_at:
            possible_heights = sorted(
                {
                    cross_dims[i]
                    for i in valid_types
                    if remaining[i] > 0
                    and cross_dims[i] <= cross_capacity - cross_used
                    and line_dims[i] <= line_capacity
                }
            )
            if not possible_heights:
                break

            # Dominated adjacent heights usually yield the same pattern.  Keep
            # all small sets; sample an even spread only on unusually large N.
            if len(possible_heights) > 18:
                sampled = []
                for q in range(18):
                    pos = int(q * (len(possible_heights) - 1) / 17.0)
                    sampled.append(possible_heights[pos])
                possible_heights = sorted(set(sampled))

            noise = [0.0 for _ in range(n)]
            for i in valid_types:
                noise[i] = rng.uniform(-0.18, 0.18) * temperature
            patterns = []
            seen = set()
            for height in possible_heights:
                if time.time() >= stop_at:
                    break
                pattern = bounded_line_pattern(
                    line_capacity,
                    line_dims,
                    cross_dims,
                    remaining,
                    height,
                    noise,
                    stop_at,
                )
                if not pattern or pattern in seen:
                    continue
                seen.add(pattern)
                actual_height = max(cross_dims[i] for i in pattern)
                actual_value = sum(values[i] for i in pattern)
                occupied = sum(line_dims[i] for i in pattern) / line_capacity
                power = 0.88 + rng.uniform(-0.10, 0.10) * temperature
                score = actual_value / (actual_height ** power)
                score *= 0.82 + 0.18 * occupied
                score *= 1.0 + rng.uniform(-0.08, 0.08) * temperature
                patterns.append((score, actual_value, actual_height, pattern))
            if not patterns:
                break
            patterns.sort(key=lambda z: z[0], reverse=True)
            rcl = min(len(patterns), 1 + int(temperature * 5.0))
            if rcl > 1 and rng.random() < temperature * 0.5:
                selected = patterns[int((rng.random() ** 2) * rcl)]
            else:
                selected = patterns[0]
            _, pattern_value, height, pattern = selected

            strip_items = []
            line_used = 0.0
            for i in pattern:
                p = used[i]
                used[i] += 1
                remaining[i] -= 1
                if horizontal:
                    x, y = line_used, cross_used
                else:
                    x, y = cross_used, line_used
                strip_items.append((i, p, x, y))
                placed.append((i, p, x, y))
                line_used += line_dims[i]
            strips.append((cross_used, strip_items, height))
            cross_used += height
            total += pattern_value

        return total, placed, strip_tree(strips, horizontal)

    # First incumbent: a deterministic, demand-aware full construction.  It is
    # deliberately completed before the slower diversification loop starts.
    quick_stop = min(deadline, t0 + max(0.01, min(0.18, time_limit_s * 0.08)))
    try:
        value, placed, tree = leaf_constructor((0.55, 0.42, 0.12, 0.0, None), quick_stop)
        consider(value, placed, tree)
    except Exception:
        pass

    # Spend a bounded early slice comparing multiple partial topologies.  The
    # remaining time is still available to the broader restart neighborhoods.
    if n > 45 or sum(demands[:n]) > 100:
        beam_stop = min(deadline, time.time() + max(0.0, time_limit_s) * 0.035)
        try:
            value, placed, tree = beam_constructor(beam_stop)
            consider(value, placed, tree)
        except Exception:
            pass

    # The loop is wall-clock bounded only.  Restarts alternate neighborhoods,
    # scoring regimes, cut preferences, and elite-guided perturbations.
    restart = 0
    while time.time() < deadline:
        try:
            strategy = restart % 7
            if strategy == 2 or strategy == 5:
                horizontal = strategy == 2
                temperature = 0.25 + 0.7 * rng.random()
                value, placed, tree = strip_constructor(
                    horizontal, temperature, deadline
                )
            else:
                density_weight = rng.uniform(0.05, 0.95)
                lookahead = rng.uniform(0.18, 0.92)
                snug_weight = rng.uniform(0.02, 0.34)
                temperature = rng.uniform(0.15, 0.95)
                elite_counts = None
                if best_value > 0.0 and restart % 3 == 0:
                    elite_counts = [sum(row) for row in best["selection_status"]]
                value, placed, tree = leaf_constructor(
                    (
                        density_weight,
                        lookahead,
                        snug_weight,
                        temperature,
                        elite_counts,
                    ),
                    deadline,
                )
            consider(value, placed, tree)
        except Exception:
            # A failed neighborhood never invalidates the incumbent.
            pass
        restart += 1

    return best
