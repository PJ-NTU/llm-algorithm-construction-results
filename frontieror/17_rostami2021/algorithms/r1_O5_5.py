"""Stochastic hub search with min-cut expansion and whole-cluster migration
Uses vectorised incremental descent, submodular assignment min-cuts, whole-cluster swaps and small-state-space enumeration, keeping the best feasible solution within the wall-clock limit.
"""

import math
import random
import time

try:
    import numpy as np
except Exception:
    np = None
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow, breadth_first_order
except Exception:
    maximum_flow = None


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
    except Exception:
        budget = 0.0
    deadline = started + 0.96 * budget
    try:
        n = max(0, int(instance['n']))
        ns = max(0, int(instance['num_scenarios']))
    except Exception:
        return {'hub_selection': [1], 'scenario_allocations': {'0': [0]}}
    if n == 0:
        return {'hub_selection': [], 'scenario_allocations': {str(s): [] for s in range(ns)}}
    try:
        first = min(range(n), key=lambda h: float(instance['fixed_costs'][h]))
    except Exception:
        first = 0
    fallback = {
        'hub_selection': [int(i == first) for i in range(n)],
        'scenario_allocations': {str(s): [first] * n for s in range(ns)},
    }
    if np is None or n == 1 or time.perf_counter() >= deadline:
        return fallback
    best = None

    try:
        d = np.asarray(instance['distances'], dtype=float)
        f = np.asarray(instance['fixed_costs'], dtype=float)
        prob = np.asarray(instance['scenarios_probability'], dtype=float)
        w = np.asarray(instance['scenarios_demands'], dtype=float)
        alpha = float(instance['cost_parameters_alpha'])
        chi = float(instance['cost_parameters_chi'])
        delta = float(instance['cost_parameters_delta'])
        if d.shape != (n, n) or f.shape != (n,) or prob.shape != (ns,):
            return fallback
        if ns == 0:
            chosen = [i for i in range(n) if f[i] < 0]
            if not chosen:
                chosen = [int(np.argmin(f))]
            return {'hub_selection': [int(i in chosen) for i in range(n)], 'scenario_allocations': {}}
        if w.shape != (ns, n, n):
            return fallback
        nodes = np.arange(n)
        scenes = np.arange(ns)[:, None]
        rows = nodes[None, :]
        diagonal = np.diag(d)
        # Probabilities are folded into scenario coefficients; storage and comparison always use the full unrounded objective.
        q = alpha * prob[:, None, None] * w
        u = prob[:, None, None] * (
            chi * w.sum(axis=2)[:, :, None] * d[None, :, :]
            + delta * w.sum(axis=1)[:, :, None] * d.T[None, :, :]
        )
        u[:, nodes, nodes] = 0.0
        u += q[:, nodes, nodes][:, :, None] * diagonal[None, None, :]
        q[:, nodes, nodes] = 0.0
        if not (np.isfinite(u).all() and np.isfinite(q).all() and np.isfinite(f).all()):
            return fallback
        nonnegative_pairs = bool((q >= 0).all() and (d >= 0).all())
        if time.perf_counter() >= deadline:
            return fallback
        strength = 1.0 + math.log1p(budget)
        width = max(2, int(math.ceil(strength * math.sqrt(n))))
        elite_limit = min(2 * n, width)
        radius_limit = min(n, max(1, int(math.ceil(strength * n ** 0.25))))
        slice_length = budget / (strength * math.sqrt(n * max(1, ns)))
        seed = (n * 1000003 + ns * 9176 + int(abs(float(f.sum()))) * 131) & 0xffffffff
        rng = random.Random(seed)
        cache = {}
        elite = []
        tried = {}
        last_improvement = started

        def hubs_of(mask):
            return np.array([i for i in range(n) if mask & (1 << i)], dtype=int)

        def scene_values(a):
            pair_d = d[a[:, :, None], a[:, None, :]]
            return u[scenes, rows, a].sum(axis=1) + np.einsum('sij,sij->s', q, pair_d)

        def scene_value(s, a):
            return float(u[s, nodes, a].sum() + np.sum(q[s] * d[a[:, None], a[None, :]]))

        def remember(mask, a, values):
            nonlocal best, last_improvement
            old = cache.get(mask)
            if old is not None:
                improve = values < old[3]
                if not improve.any():
                    return old
                a = np.where(improve[:, None], a, old[2])
                values = np.minimum(values, old[3])
            value = float(f[hubs_of(mask)].sum() + values.sum())
            state = (value, mask, a.copy(), values.copy())
            cache[mask] = state
            if best is None or value < best[0]:
                best = state
                last_improvement = time.perf_counter()
            if mask not in elite:
                elite.append(mask)
            elite.sort(key=lambda m: cache[m][0])
            del elite[elite_limit:]
            return state

        def normalize(hubs, supplied=None):
            a = hubs[np.argmin(u[:, :, hubs], axis=2)]
            if supplied is not None:
                allowed = np.zeros(n, dtype=bool)
                allowed[hubs] = True
                a = np.where(allowed[supplied], supplied, a)
            a[:, hubs] = hubs
            return a

        def local_table(a, hubs, targets, cutoff):
            p = len(hubs)
            position = np.full(n, -1, dtype=int)
            position[hubs] = np.arange(p)
            out = np.empty((ns, n, p))
            inc = np.empty_like(out)
            offset = nodes[:, None] * p
            for s in range(ns):
                if time.perf_counter() >= cutoff:
                    return None
                index = (offset + position[a[s]][None, :]).ravel()
                out[s] = np.bincount(index, weights=q[s].ravel(), minlength=n * p).reshape(n, p)
                inc[s] = np.bincount(index, weights=q[s].T.ravel(), minlength=n * p).reshape(n, p)
            local = u[:, :, targets].copy()
            local += np.einsum('sil,kl->sik', out, d[np.ix_(targets, hubs)])
            local += np.einsum('sil,lk->sik', inc, d[np.ix_(hubs, targets)])
            return local, out, inc, position

        def descend_assignments(a, hubs, cutoff):
            a = a.copy()
            if len(hubs) <= 1 or len(hubs) == n or time.perf_counter() >= cutoff:
                return a
            built = local_table(a, hubs, hubs, cutoff)
            if built is None:
                return a
            local, _, _, position = built
            old_pos = position[a]
            while time.perf_counter() < cutoff:
                choices = np.argmin(local, axis=2)
                old_energy = local[scenes, rows, old_pos]
                changes = local[scenes, rows, choices] - old_energy
                changes[:, hubs] = np.inf
                selected = np.argmin(changes, axis=1)
                active = np.flatnonzero(
                    changes[np.arange(ns), selected]
                    < -1e-11 * (1.0 + np.abs(old_energy[np.arange(ns), selected]))
                )
                if not len(active):
                    break
                for s in active:
                    i = selected[s]
                    old = a[s, i]
                    pos = choices[s, i]
                    new = hubs[pos]
                    local[s] += (
                        q[s, :, i, None] * (d[hubs, new] - d[hubs, old])[None, :]
                        + q[s, i, :, None] * (d[new, hubs] - d[old, hubs])[None, :]
                    )
                    a[s, i] = new
                    old_pos[s, i] = pos
            return a

        def cut_scene(s, initial, hubs, target, cutoff, binary=False):
            if maximum_flow is None or time.perf_counter() >= cutoff:
                return initial
            a = initial.copy()
            free = np.ones(n, dtype=bool)
            free[hubs] = False
            if binary:
                a[free] = hubs[0]
            free &= a != target
            v = np.flatnonzero(free)
            m = len(v)
            if not m or m * m > max(0.0, cutoff - time.perf_counter()) * 1000000.0:
                return initial
            labels = a[v]
            pair = q[s][np.ix_(v, v)]
            directed = pair * (
                d[labels, target][:, None] + d[target, labels][None, :]
                - d[np.ix_(labels, labels)] - d[target, target]
            )
            capacity = 0.5 * (directed + directed.T)
            magnitude = max(1.0, float(np.max(np.abs(capacity))))
            if float(capacity.min()) < -1e-10 * magnitude:
                return initial
            capacity = np.maximum(capacity, 0.0)
            np.fill_diagonal(capacity, 0.0)
            old_pair = q[s] * d[a[:, None], a[None, :]]
            changes = u[s, :, target] - u[s, nodes, a]
            changes += (q[s] * d[target, a][None, :]).sum(axis=1)
            changes += (q[s] * d[a, target][:, None]).sum(axis=0)
            changes -= old_pair.sum(axis=1) + old_pair.sum(axis=0)
            terminal = changes[v] - capacity.sum(axis=1)
            graph = np.zeros((m + 2, m + 2))
            graph[:m, :m] = capacity
            graph[m, :m] = np.maximum(terminal, 0.0)
            graph[:m, m + 1] = np.maximum(-terminal, 0.0)
            largest = max(float(graph.sum(axis=0).max()), float(graph.sum(axis=1).max()))
            if largest == 0.0:
                return a
            # Control the total integer capacity; rounding affects only proposals, and the full floating-point objective decides acceptance.
            integer_graph = np.rint(graph * ((2 ** 28 - 1) / largest)).astype(np.int64)
            if time.perf_counter() >= cutoff:
                return initial
            sparse = csr_matrix(integer_graph)
            flow = maximum_flow(sparse, m, m + 1)
            residual = sparse - flow.flow
            residual.eliminate_zeros()
            reachable = breadth_first_order(residual, m, directed=True, return_predecessors=False)
            on_source = np.zeros(m + 2, dtype=bool)
            on_source[reachable] = True
            a[v[~on_source[:m]]] = target
            return a

        def refine(state, cutoff):
            mask = state[1]
            hubs = hubs_of(mask)
            if len(hubs) < 2 or len(hubs) == n:
                return state
            a = state[2].copy()
            values = state[3].copy()
            order = sorted(range(ns), key=lambda s: -abs(values[s]))
            for index, s in enumerate(order):
                now = time.perf_counter()
                if now >= cutoff:
                    break
                scene_end = now + (cutoff - now) / (ns - index)
                if len(hubs) == 2:
                    trial = cut_scene(s, a[s], hubs, hubs[1], scene_end, True)
                    value = scene_value(s, trial)
                    if value < values[s]:
                        a[s], values[s] = trial, value
                    continue
                while time.perf_counter() < scene_end:
                    changed = False
                    target_order = list(hubs)
                    rng.shuffle(target_order)
                    for target in target_order:
                        if time.perf_counter() >= scene_end:
                            break
                        trial = cut_scene(s, a[s], hubs, target, scene_end)
                        value = scene_value(s, trial)
                        if value < values[s] - 1e-11 * (1.0 + abs(values[s])):
                            a[s], values[s] = trial, value
                            changed = True
                    if not changed:
                        break
            a = descend_assignments(a, hubs, cutoff)
            return remember(mask, a, scene_values(a))

        def evaluate(mask, warm=None, cutoff=None):
            if cutoff is None:
                cutoff = deadline
            hubs = hubs_of(mask)
            old = cache.get(mask)
            if time.perf_counter() >= cutoff:
                return old
            start = normalize(hubs, warm if warm is not None else (old[2] if old else None))
            a = descend_assignments(start, hubs, cutoff)
            state = remember(mask, a, scene_values(a))
            nearest = normalize(hubs)
            if time.perf_counter() < cutoff and not np.array_equal(nearest, start):
                nearest = descend_assignments(nearest, hubs, cutoff)
                state = remember(mask, nearest, scene_values(nearest))
            if len(hubs) == 2 and time.perf_counter() < cutoff:
                state = refine(state, cutoff)
            return state

        def proxy(mask):
            hubs = hubs_of(mask)
            costs = u[:, :, hubs].min(axis=2)
            costs[:, hubs] = u[:, hubs, hubs]
            return float(f[hubs].sum() + costs.sum())

        # Compute all single-hub costs at once; store a complete solution first, then start the expensive search.
        singleton = f + u.sum(axis=(0, 1)) + (q.sum(axis=(1, 2))[:, None] * diagonal).sum(axis=0)
        ranking = np.argsort(singleton).tolist()
        for h in ranking[:elite_limit]:
            if time.perf_counter() >= deadline:
                break
            a = np.full((ns, n), h, dtype=int)
            values = u[:, :, h].sum(axis=1) + q.sum(axis=(1, 2)) * d[h, h]
            remember(1 << h, a, values)
        full_mask = (1 << n) - 1
        if time.perf_counter() < deadline:
            identity = np.broadcast_to(nodes, (ns, n)).copy()
            remember(full_mask, identity, scene_values(identity))

        # A cheap facility-location surrogate only produces seeds; all retained solutions are scored by the full objective.
        current_mask = 1 << ranking[0]
        construction_end = min(deadline, started + 0.14 * budget)
        while time.perf_counter() < construction_end:
            hubs = hubs_of(current_mask)
            if len(hubs) == n:
                break
            current = u[:, :, hubs].min(axis=2)
            current[:, hubs] = u[:, hubs, hubs]
            differences = u - current[:, :, None]
            gains = np.minimum(differences, 0.0)
            gains[:, hubs, :] = 0.0
            scores = f + gains.sum(axis=(0, 1))
            forced = differences[:, nodes, nodes]
            scores += (forced - np.minimum(forced, 0.0)).sum(axis=0)
            scores[hubs] = np.inf
            add = int(np.argmin(scores))
            if scores[add] >= -1e-10:
                break
            current_mask |= 1 << add
            evaluate(current_mask, cutoff=construction_end)

        def neighbors(state, cutoff):
            mask, a = state[1], state[2]
            hubs = hubs_of(mask)
            closed = np.array([i for i in range(n) if not mask & (1 << i)], dtype=int)
            built = local_table(a, hubs, nodes, cutoff)
            if built is None:
                return []
            local, out, inc, _ = built
            current = local[scenes, rows, a]
            difference = local - current[:, :, None]
            gains = np.minimum(difference, 0.0)
            gains[:, hubs, :] = 0.0
            add_scores = f + gains.sum(axis=(0, 1))
            forced = difference[:, nodes, nodes]
            add_scores += (forced - np.minimum(forced, 0.0)).sum(axis=0)
            moves = [(float(add_scores[k]), mask | (1 << int(k)), -1, int(k)) for k in closed]
            for hp, old in enumerate(hubs):
                if time.perf_counter() >= cutoff:
                    break
                old = int(old)
                belonging = a == old
                if len(hubs) > 1:
                    others = hubs[hubs != old]
                    alternatives = local[:, :, others].min(axis=2)
                    closing = float(((alternatives - current) * belonging).sum() - f[old])
                    moves.append((closing, mask ^ (1 << old), old, -1))
                if not len(closed):
                    continue
                # Move the whole cluster of the old hub to the candidate node, while forcing the new hub to point to itself.
                # Single-node deltas plus second-order corrections for all affected node pairs give the exact change.
                swap_scores = f - f[old]
                for s in range(ns):
                    members = belonging[s]
                    change = difference[s, members, :].sum(axis=0)
                    internal = float(out[s, members, hp].sum())
                    change += internal * (diagonal - d[:, old] - d[old, :] + d[old, old])
                    extra = ~members
                    original = a[s]
                    correction = local[s, nodes, nodes] - current[s]
                    correction += out[s, :, hp] * (
                        diagonal - d[:, old] - d[original, nodes] + d[original, old]
                    )
                    correction += inc[s, :, hp] * (
                        diagonal - d[nodes, original] - d[old, :] + d[old, original]
                    )
                    change += correction * extra
                    swap_scores = swap_scores + change
                moves.extend(
                    (float(swap_scores[k]), (mask ^ (1 << old)) | (1 << int(k)), old, int(k))
                    for k in closed
                )
            moves.sort(key=lambda item: item[0])
            untried = [item for item in moves if tried.get((mask, item[1]), float('inf')) > state[0] + 1e-9]
            if len(untried) <= width:
                return untried
            tail_count = min(len(untried) - width, max(1, int(math.ceil(math.sqrt(width)))))
            return untried[:width] + rng.sample(untried[width:], tail_count)

        def basin(state, cutoff):
            while time.perf_counter() < cutoff:
                origin = state
                moved = False
                for _, mask, old, new in neighbors(state, cutoff):
                    if time.perf_counter() >= cutoff:
                        break
                    warm = state[2]
                    if old >= 0 and new >= 0:
                        warm = warm.copy()
                        warm[warm == old] = new
                        warm[:, new] = new
                    tried[(state[1], mask)] = state[0]
                    trial_end = min(cutoff, time.perf_counter() + slice_length)
                    candidate = evaluate(mask, warm, trial_end)
                    if candidate is not None and candidate[0] < state[0] - 1e-9:
                        state = candidate
                        moved = True
                        break
                if moved:
                    continue
                refine_end = min(cutoff, time.perf_counter() + slice_length * strength)
                state = refine(state, refine_end)
                if state[0] >= origin[0] - 1e-9:
                    return state
            return state

        def exact_scenario(s, hubs, incumbent, cutoff):
            free = np.array([i for i in range(n) if i not in set(hubs.tolist())], dtype=int)
            if len(hubs) == 1 or not len(free):
                return incumbent.copy(), True
            activity = np.abs(q[s]).sum(axis=0) + np.abs(q[s]).sum(axis=1)
            free = free[np.argsort(-activity[free])]
            p = len(hubs)
            terms = u[s][np.ix_(free, hubs)].copy()
            terms += np.einsum('ij,kj->ik', q[s][np.ix_(free, hubs)], d[np.ix_(hubs, hubs)])
            terms += np.einsum('ji,jk->ik', q[s][np.ix_(hubs, free)], d[np.ix_(hubs, hubs)])
            constant = float(u[s, hubs, hubs].sum() + np.sum(q[s][np.ix_(hubs, hubs)] * d[np.ix_(hubs, hubs)]))
            suffix = np.zeros(len(free) + 1)
            if not nonnegative_pairs:
                hd = d[np.ix_(hubs, hubs)]
                for j in range(len(free) - 1, -1, -1):
                    suffix[j] = suffix[j + 1]
                    for k in range(j + 1, len(free)):
                        suffix[j] += np.min(q[s, free[j], free[k]] * hd + q[s, free[k], free[j]] * hd.T)
            champion = incumbent.copy()
            champion_value = scene_value(s, champion)
            work = incumbent.copy()
            completed = True

            def visit(position, partial, remaining):
                nonlocal champion, champion_value, completed
                if time.perf_counter() >= cutoff:
                    completed = False
                    return
                if position == len(free):
                    if partial < champion_value:
                        champion_value = partial
                        champion = work.copy()
                    return
                lower = partial + float(remaining.min(axis=1).sum()) + suffix[position]
                if lower >= champion_value - 1e-11 * (1.0 + abs(champion_value)):
                    return
                i = free[position]
                rest = free[position + 1:]
                for hp in np.argsort(remaining[0]):
                    if not completed:
                        return
                    h = hubs[hp]
                    work[i] = h
                    updated = remaining[1:] + (
                        q[s, rest, i, None] * d[hubs, h][None, :]
                        + q[s, i, rest, None] * d[h, hubs][None, :]
                    )
                    visit(position + 1, partial + float(remaining[0, hp]), updated)

            visit(0, constant, terms)
            return champion, completed

        # The state-space threshold depends on the time limit and dimension, not on a fixed enumeration range for some evaluation scale.
        enumeration_size = budget * 1600.0 / max(1, ns)
        small = n < math.log2(max(1.0, enumeration_size))
        if best is not None and time.perf_counter() < deadline:
            first_end = min(deadline, started + budget * (0.28 if small else 0.55))
            basin(best, first_end)
        if small and time.perf_counter() < deadline:
            ordered = []
            for mask in range(1, full_mask + 1):
                if time.perf_counter() >= deadline:
                    break
                ordered.append((proxy(mask), mask))
            ordered.sort()
            for lower, mask in ordered:
                if time.perf_counter() >= deadline:
                    break
                if nonnegative_pairs and best is not None and lower >= best[0] - 1e-9:
                    break
                state = evaluate(mask, best[2] if best else None,
                                 min(deadline, time.perf_counter() + slice_length))
                if state is None:
                    continue
                a = state[2].copy()
                values = state[3].copy()
                hubs = hubs_of(mask)
                for s in range(ns):
                    if time.perf_counter() >= deadline:
                        break
                    a[s], _ = exact_scenario(s, hubs, a[s], deadline)
                    values[s] = scene_value(s, a[s])
                    remember(mask, a, values)

        # No cap on restarts; with more time, neighbourhood width, elite count and perturbation range grow together.
        while time.perf_counter() < deadline and best is not None:
            pool = elite[:max(1, int(math.ceil(math.sqrt(len(elite)))))]
            parent = cache[rng.choice(pool)] if rng.random() < 0.45 else best
            stagnation = (time.perf_counter() - last_improvement) / max(slice_length, budget / (1.0 + strength))
            radius = min(radius_limit, 1 + int(stagnation * strength))
            mask = parent[1]
            if rng.random() < 1.0 / strength:
                count = max(1, min(n, mask.bit_count() + rng.randint(-radius, radius)))
                pool_size = min(n, max(count, int(math.ceil(count * strength))))
                mask = sum(1 << h for h in rng.sample(ranking[:pool_size], count))
            else:
                for _ in range(rng.randint(1, radius)):
                    opened = [i for i in range(n) if mask & (1 << i)]
                    closed = [i for i in range(n) if not mask & (1 << i)]
                    roll = rng.random()
                    if closed and (roll < 0.34 or len(opened) == 1):
                        mask |= 1 << rng.choice(closed)
                    elif len(opened) > 1 and (roll < 0.67 or not closed):
                        mask ^= 1 << rng.choice(opened)
                    elif closed:
                        mask ^= 1 << rng.choice(opened)
                        mask |= 1 << rng.choice(closed)
            warm = parent[2]
            hubs = hubs_of(mask)
            if rng.random() < 1.0 / strength:
                warm = normalize(hubs, warm)
                free = [i for i in range(n) if not mask & (1 << i)]
                if free:
                    target = rng.choice(hubs.tolist())
                    count = min(len(free), max(1, int(math.ceil(math.sqrt(len(free)) * radius / strength))))
                    changed = rng.sample(free, count)
                    warm[:, changed] = target
            phase_end = min(deadline, time.perf_counter() + slice_length * strength)
            state = evaluate(mask, warm, phase_end)
            if state is not None and time.perf_counter() < deadline:
                state = refine(state, min(deadline, time.perf_counter() + slice_length))
                basin(state, min(deadline, time.perf_counter() + (deadline - time.perf_counter()) / strength))
    except Exception:
        # No unfinished candidate may replace the complete feasible solution already saved.
        pass

    if best is not None:
        try:
            mask, allocations = best[1], best[2]
            return {
                'hub_selection': [int(bool(mask & (1 << i))) for i in range(n)],
                'scenario_allocations': {str(s): allocations[s].astype(int).tolist() for s in range(ns)},
            }
        except Exception:
            pass
    return fallback
