# Library slot 5/10  id=alg_e03fea29a132
# score=-12045.416000000001  feasible_rate=1.0
# score_vec=[-2914.7499999999995, -2434.98, -6775.889999999999, -6473.94, -2426.6899999999996, -6705.530000000001, -6878.5999999999985, -15710.289999999995, -37525.59000000001, -32607.899999999998]

"""Anytime heuristic for bounded two-dimensional guillotine knapsack."""

import math
import random
import time


def _solve(instance, time_limit_s, t0):
    """Implementation separated so the public entry point can guarantee fallback."""

    try:
        limit = max(0.0, float(time_limit_s))
    except Exception:
        limit = 0.0
    deadline = t0 + limit * 0.85

    try:
        n_types = int(instance["num_piece_types"])
        sheet_l = float(instance["sheet_length"])
        sheet_w = float(instance["sheet_width"])
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

    # This is installed before any search: even a tiny budget has a valid answer.
    def empty_solution():
        return {
            "selection_status": [[0] * demands[i] for i in range(n_types)],
            "placements": [
                [[-1.0, -1.0] for _ in range(demands[i])]
                for i in range(n_types)
            ],
            "guillotine_cut_tree": {"type": "empty"},
        }

    fallback = empty_solution()
    if (
        n_types < 0
        or len(lengths) < n_types
        or len(widths) < n_types
        or len(demands) < n_types
        or len(values) < n_types
        or sheet_l <= 0.0
        or sheet_w <= 0.0
        or time.time() >= deadline
    ):
        return fallback

    # item = (type, copy, length, width, value, area)
    items = []
    for i in range(n_types):
        li, wi, vi = lengths[i], widths[i], values[i]
        if (
            li > 0.0
            and wi > 0.0
            and li <= sheet_l + 1e-9
            and wi <= sheet_w + 1e-9
            and vi > 0.0
            and math.isfinite(li)
            and math.isfinite(wi)
            and math.isfinite(vi)
        ):
            for p in range(demands[i]):
                items.append((i, p, li, wi, vi, li * wi))

    if not items or time.time() >= deadline:
        return fallback

    min_l = min(q[2] for q in items)
    min_w = min(q[3] for q in items)
    eps = 1e-9 * max(1.0, sheet_l, sheet_w)

    # Internal tree nodes: [0] empty, [1, item_id] leaf,
    # [2, direction(0=V,1=H), coordinate, low_child, high_child] cut.
    def leaf(item_id):
        return [1, item_id]

    def cut(direction, coordinate, low, high):
        return [2, direction, float(coordinate), low, high]

    def export_tree(root):
        kind = root[0]
        if kind == 0:
            return {"type": "empty"}
        if kind == 1:
            q = items[root[1]]
            return {"type": "leaf", "piece_index": [q[0], q[1]]}
        low = export_tree(root[3])
        high = export_tree(root[4])
        if root[1] == 0:
            return {
                "type": "node",
                "cut_direction": "vertical",
                "cut_coordinate": root[2],
                "left_child": low,
                "right_child": high,
            }
        return {
            "type": "node",
            "cut_direction": "horizontal",
            "cut_coordinate": root[2],
            "bottom": low,
            "top": high,
        }

    def make_solution(result):
        status = [[0] * demands[i] for i in range(n_types)]
        positions = [
            [[-1.0, -1.0] for _ in range(demands[i])]
            for i in range(n_types)
        ]
        for item_id, x, y in result[2]:
            q = items[item_id]
            status[q[0]][q[1]] = 1
            positions[q[0]][q[1]] = [float(x), float(y)]
        return {
            "selection_status": status,
            "placements": positions,
            "guillotine_cut_tree": export_tree(result[3]),
        }

    def residual_stats(rectangles):
        dead = 0.0
        largest = 0.0
        aspect = 0.0
        for a, b in rectangles:
            if a <= eps or b <= eps:
                continue
            area = a * b
            if a + eps < min_l or b + eps < min_w:
                dead += area
            if area > largest:
                largest = area
            aspect += area * (max(a, b) / max(eps, min(a, b)) - 1.0)
        return dead, largest, aspect

    def placement_key(rw, rh, pw, ph, orientation, mode, jitter):
        dw, dh = max(0.0, rw - pw), max(0.0, rh - ph)
        waste = max(0.0, rw * rh - pw * ph)
        if orientation == 0:  # vertical full cut, then horizontal in left strip
            residuals = ((dw, rh), (pw, dh))
        else:
            residuals = ((rw, dh), (dw, ph))
        dead, largest, aspect = residual_stats(residuals)
        short = min(dw, dh)
        long_side = max(dw, dh)
        edge_misses = int(dw > eps) + int(dh > eps)
        if mode == 0:
            key = (waste, dead, short, long_side, aspect)
        elif mode == 1:
            key = (dead, waste, aspect, -largest)
        elif mode == 2:
            key = (short, long_side, waste, dead, aspect)
        elif mode == 3:
            key = (long_side, short, waste, dead, aspect)
        elif mode == 4:
            key = (edge_misses, waste, dead, aspect, -largest)
        elif mode == 5:
            key = (waste + 3.0 * dead + 0.015 * aspect, -largest, short)
        elif mode == 6:
            key = (dead, -largest, waste, aspect)
        else:
            key = (waste * jitter + 2.0 * dead, short * jitter, aspect)
        return key

    def put_in_free(node, x, y, rw, rh, item_id, orientation):
        q = items[item_id]
        pw, ph = q[2], q[3]
        dw, dh = max(0.0, rw - pw), max(0.0, rh - ph)
        new_free = []

        if dw <= eps and dh <= eps:
            node[:] = leaf(item_id)
        elif dw <= eps:
            top = [0]
            node[:] = cut(1, y + ph, leaf(item_id), top)
            new_free.append((top, x, y + ph, rw, dh))
        elif dh <= eps:
            right = [0]
            node[:] = cut(0, x + pw, leaf(item_id), right)
            new_free.append((right, x + pw, y, dw, rh))
        elif orientation == 0:
            top = [0]
            right = [0]
            left = cut(1, y + ph, leaf(item_id), top)
            node[:] = cut(0, x + pw, left, right)
            new_free.append((top, x, y + ph, pw, dh))
            new_free.append((right, x + pw, y, dw, rh))
        else:
            right = [0]
            top = [0]
            bottom = cut(0, x + pw, leaf(item_id), right)
            node[:] = cut(1, y + ph, bottom, top)
            new_free.append((right, x + pw, y, dw, ph))
            new_free.append((top, x, y + ph, rw, dh))
        return new_free

    def decode_bsp(order, mode, rng):
        root = [0]
        free = [(root, 0.0, 0.0, sheet_l, sheet_w)]
        placed = []
        total_value = 0.0
        total_area = 0.0

        for step, item_id in enumerate(order):
            if (step & 15) == 0 and time.time() >= deadline:
                break
            q = items[item_id]
            pw, ph = q[2], q[3]
            best_choice = None
            best_key = None
            timed_out = False
            for f_idx, region in enumerate(free):
                if (f_idx & 63) == 0 and time.time() >= deadline:
                    timed_out = True
                    break
                rw, rh = region[3], region[4]
                if pw <= rw + eps and ph <= rh + eps:
                    orientations = (0, 1) if rw - pw > eps and rh - ph > eps else (0,)
                    for orientation in orientations:
                        jitter = 0.86 + 0.28 * rng.random() if mode == 7 else 1.0
                        key = placement_key(
                            rw, rh, pw, ph, orientation, mode, jitter
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best_choice = (f_idx, orientation)
            if timed_out:
                break
            if best_choice is None:
                continue

            f_idx, orientation = best_choice
            node, x, y, rw, rh = free.pop(f_idx)
            free.extend(put_in_free(node, x, y, rw, rh, item_id, orientation))
            placed.append((item_id, x, y))
            total_value += q[4]
            total_area += q[5]

        return (total_value, total_area, placed, root)

    def chain_slices(parts, direction, outer_end):
        """Build contiguous shelf/strip cuts from high coordinate to low."""
        if not parts:
            return [0]
        used_end = parts[-1][1]
        tail = [0] if used_end < outer_end - eps else None
        for item_id, end_coord in reversed(parts):
            one = leaf(item_id)
            if tail is None:
                tail = one
            else:
                tail = cut(direction, end_coord, one, tail)
        return tail

    def decode_shelves(order, vertical):
        # Horizontal shelves contain left-to-right pieces.  With vertical=True,
        # transpose that construction into bottom-to-top pieces in vertical strips.
        shelves = []
        outer_used = 0.0
        placed = []
        total_value = 0.0
        total_area = 0.0

        for step, item_id in enumerate(order):
            if (step & 15) == 0 and time.time() >= deadline:
                break
            q = items[item_id]
            inner_size = q[3] if vertical else q[2]
            outer_size = q[2] if vertical else q[3]
            inner_limit = sheet_w if vertical else sheet_l
            outer_limit = sheet_l if vertical else sheet_w

            best_idx = -1
            best_left = float("inf")
            for s_idx, shelf in enumerate(shelves):
                if (s_idx & 63) == 0 and time.time() >= deadline:
                    break
                if outer_size <= shelf[1] + eps and shelf[2] + inner_size <= inner_limit + eps:
                    left = inner_limit - shelf[2] - inner_size
                    if left < best_left:
                        best_left = left
                        best_idx = s_idx

            if best_idx < 0:
                if outer_used + outer_size > outer_limit + eps:
                    continue
                shelves.append([outer_used, outer_size, 0.0, []])
                best_idx = len(shelves) - 1
                outer_used += outer_size

            shelf = shelves[best_idx]
            inner_pos = shelf[2]
            if vertical:
                x, y = shelf[0], inner_pos
            else:
                x, y = inner_pos, shelf[0]
            shelf[3].append((item_id, inner_pos + inner_size))
            shelf[2] += inner_size
            placed.append((item_id, x, y))
            total_value += q[4]
            total_area += q[5]

        # First build each shelf's inner chain, then chain the shelves themselves.
        outer_parts = []
        for shelf in shelves:
            inner_end = sheet_w if vertical else sheet_l
            inner_direction = 1 if vertical else 0
            shelf_tree = chain_slices(shelf[3], inner_direction, inner_end)
            outer_parts.append((shelf_tree, shelf[0] + shelf[1]))

        outer_limit = sheet_l if vertical else sheet_w
        outer_direction = 0 if vertical else 1
        if not outer_parts:
            root = [0]
        else:
            tail = [0] if outer_parts[-1][1] < outer_limit - eps else None
            for shelf_tree, end_coord in reversed(outer_parts):
                if tail is None:
                    tail = shelf_tree
                else:
                    tail = cut(outer_direction, end_coord, shelf_tree, tail)
            root = tail
        return (total_value, total_area, placed, root)

    # Deterministic seed: reproducible without process-randomized hash().
    seed = 2166136261
    for q in items:
        seed ^= int(abs(q[2]) * 1000.0) + 31 * int(abs(q[3]) * 1000.0)
        seed = (seed * 16777619) & 0xFFFFFFFF
        seed ^= int(abs(q[4]) * 1000.0)
        seed = (seed * 16777619) & 0xFFFFFFFF
    rng = random.Random(seed)

    item_ids = list(range(len(items)))
    def sorted_order(kind):
        def key(item_id):
            q = items[item_id]
            value, area = q[4], q[5]
            if kind == 0:
                primary = value / area
            elif kind == 1:
                primary = value / math.sqrt(area)
            elif kind == 2:
                primary = value
            elif kind == 3:
                primary = area
            elif kind == 4:
                primary = q[3]
            elif kind == 5:
                primary = q[2]
            elif kind == 6:
                primary = value / (area ** 0.78)
            else:
                fill = min(sheet_l / q[2], sheet_w / q[3])
                primary = value / area + 1e-12 * fill
            return (-primary, -value, -area, q[0], q[1])

        return sorted(item_ids, key=key)

    # Portfolio entry: (value, area, order, mode, decoder_kind, selected, result)
    portfolio = []
    signatures = {}
    best_result = (0.0, 0.0, [], [0])

    def consider(result, order, mode, decoder_kind):
        nonlocal best_result
        selected = frozenset(x[0] for x in result[2])
        counts = [0] * n_types
        for item_id in selected:
            counts[items[item_id][0]] += 1
        signature = tuple(counts)
        old = signatures.get(signature)
        entry = (result[0], result[1], tuple(order), mode, decoder_kind, selected, result)
        if old is None or (result[0], result[1]) > (old[0], old[1]):
            signatures[signature] = entry
        if (result[0], result[1]) > (best_result[0], best_result[1]):
            best_result = result

    # Fast, structurally diverse starts.  Every loop also respects the same deadline.
    for kind in range(8):
        if time.time() >= deadline:
            break
        order = sorted_order(kind)
        for mode in range(8):
            if time.time() >= deadline:
                break
            consider(decode_bsp(order, mode, rng), order, mode, 0)
        if time.time() < deadline:
            consider(decode_shelves(order, False), order, 0, 1)
        if time.time() < deadline:
            consider(decode_shelves(order, True), order, 0, 2)

    def rebuild_portfolio():
        entries = list(signatures.values())
        entries.sort(key=lambda e: (e[0], e[1]), reverse=True)
        del entries[40:]
        # Rebuild the signature table too, preventing it from growing for the full run.
        signatures.clear()
        for entry in entries:
            counts = [0] * n_types
            for item_id in entry[5]:
                counts[items[item_id][0]] += 1
            signatures[tuple(counts)] = entry
        return entries

    portfolio = rebuild_portfolio()
    refresh_at = time.time() + 0.04

    def random_priority_order():
        alpha = rng.uniform(0.0, 1.18)
        noise = rng.uniform(0.12, 1.75)
        width_bias = rng.uniform(-0.32, 0.32)
        height_bias = rng.uniform(-0.32, 0.32)

        def priority(item_id):
            q = items[item_id]
            score = (
                math.log(max(1e-12, q[4]))
                - alpha * math.log(max(1e-12, q[5]))
                + width_bias * math.log(max(1e-12, q[2] / sheet_l))
                + height_bias * math.log(max(1e-12, q[3] / sheet_w))
                + rng.random() * noise
            )
            return -score

        return sorted(item_ids, key=priority)

    # Wall-clock-driven evolutionary variable-neighborhood search.  There is no
    # iteration cap: extra time is always converted into further restarts/mutations.
    while time.time() < deadline:
        if time.time() >= refresh_at or not portfolio:
            portfolio = rebuild_portfolio()
            refresh_at = time.time() + 0.04

        if not portfolio or rng.random() < 0.24:
            child = random_priority_order()
        else:
            rank = int((rng.random() ** 2.2) * len(portfolio))
            parent = portfolio[min(rank, len(portfolio) - 1)]
            child = list(parent[2])
            selected = parent[5]
            unselected = [x for x in child if x not in selected]

            # Eject a selected piece by promoting an excluded one; subsequent
            # best-fit decoding decides whether the exchange is profitable.
            if unselected and selected and rng.random() < 0.58:
                promoted = unselected[int(rng.random() * len(unselected))]
                old_pos = child.index(promoted)
                child.pop(old_pos)
                target = int((rng.random() ** 2.0) * max(1, old_pos))
                child.insert(target, promoted)

            # Variable-size permutation perturbation (the search itself remains
            # governed solely by the wall clock).
            changes = 1
            while rng.random() < 0.64 and changes < max(2, len(child) // 3):
                changes += 1
            for _ in range(changes):
                if time.time() >= deadline:
                    break
                if len(child) < 2:
                    break
                a = rng.randrange(len(child))
                b = rng.randrange(len(child))
                if rng.random() < 0.50:
                    token = child.pop(a)
                    child.insert(b, token)
                else:
                    child[a], child[b] = child[b], child[a]
            if len(child) > 4 and rng.random() < 0.18:
                a = rng.randrange(len(child) - 1)
                b = rng.randrange(a + 1, len(child))
                child[a:b] = reversed(child[a:b])

        decoder_roll = rng.random()
        if decoder_roll < 0.84:
            mode = rng.randrange(8)
            result = decode_bsp(child, mode, rng)
            decoder_kind = 0
        elif decoder_roll < 0.92:
            mode = 0
            result = decode_shelves(child, False)
            decoder_kind = 1
        else:
            mode = 0
            result = decode_shelves(child, True)
            decoder_kind = 2
        consider(result, child, mode, decoder_kind)

    return make_solution(best_result)


def solve(instance: dict, time_limit_s: float) -> dict:
    """Return a feasible slicing layout before the internal wall-clock deadline."""
    t0 = time.time()
    try:
        return _solve(instance, time_limit_s, t0)
    except Exception:
        # Valid challenge instances reach the main fallback above.  This last guard
        # also makes malformed or unexpectedly huge inputs fail closed, never raise.
        try:
            n_types = max(0, int(instance["num_piece_types"]))
            raw_demands = instance["max_quantities"]
            demands = [
                max(0, int(raw_demands[i])) if i < len(raw_demands) else 0
                for i in range(n_types)
            ]
        except Exception:
            n_types, demands = 0, []
        return {
            "selection_status": [[0] * demands[i] for i in range(n_types)],
            "placements": [
                [[-1.0, -1.0] for _ in range(demands[i])]
                for i in range(n_types)
            ],
            "guillotine_cut_tree": {"type": "empty"},
        }
