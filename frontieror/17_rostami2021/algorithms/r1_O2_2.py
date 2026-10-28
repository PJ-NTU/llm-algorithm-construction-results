"""Cross-scenario joint-cut fusion search
Links whole-group assignment and hub opening/closing through binary min-cuts, improving continually with assignment descent, solution fusion and adaptive perturbation.
"""

import math
import random
import time

try:
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_flow, breadth_first_order
except Exception:
    np = None


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    try:
        budget = float(time_limit_s)
        if not math.isfinite(budget):
            budget = 0.0
        budget = max(0.0, budget)
    except Exception:
        budget = 0.0
    deadline = start + budget * 0.975
    try:
        n = max(0, int(instance['n']))
        ns = max(0, int(instance['num_scenarios']))
    except Exception:
        return {'hub_selection': [1], 'scenario_allocations': {'0': [0]}}
    if n == 0:
        return {'hub_selection': [], 'scenario_allocations': {str(s): [] for s in range(ns)}}
    try:
        hub = min(range(n), key=lambda k: float(instance['fixed_costs'][k]))
    except Exception:
        hub = 0
    flags = [int(i == hub) for i in range(n)]
    answer = {'hub_selection': flags,
              'scenario_allocations': {str(s): [hub] * n for s in range(ns)}}
    if time.perf_counter() >= deadline or np is None:
        return answer

    try:
        d = np.asarray(instance['distances'], dtype=float)
        f = np.asarray(instance['fixed_costs'], dtype=float)
        p = np.asarray(instance['scenarios_probability'], dtype=float)
        alpha = float(instance['cost_parameters_alpha'])
        chi = float(instance['cost_parameters_chi'])
        delta = float(instance['cost_parameters_delta'])
        ids = np.arange(n)
        diagonal = np.diag(d).copy()
        u = np.empty((ns, n, n), dtype=float)
        w = np.empty((ns, n, n), dtype=float)
        # The input gives only 2-D distances; build the scenario collection/distribution cost c following the supplied reference material.
        # A hub's own collection/distribution cost is zero; the transfer cost of self-loop demand is kept.
        for s in range(ns):
            if time.perf_counter() >= deadline:
                return answer
            demand = np.asarray(instance['scenarios_demands'][s], dtype=float)
            u[s] = p[s] * (chi * demand.sum(axis=1)[:, None] * d
                           + delta * demand.sum(axis=0)[:, None] * d.T)
            u[s, ids, ids] = 0.0
            w[s] = (p[s] * alpha) * demand
        if time.perf_counter() >= deadline:
            return answer
        symmetric = bool(np.array_equal(d, d.T))
        pair_flow = w + w.transpose(0, 2, 1) if symmetric else None
        rng = random.Random(104729 * n + 13007 * ns)
        best_value = float('inf')
        best_state = None
        pool = []
        pool_limit = min(n + 1, max(2, int(math.ceil(math.log1p(budget) + 1))))
        last_cut_time = 0.0
        last_score_time = 0.0
        cut_enabled = True

        def value(z, a):
            result = float(f[z].sum())
            for s in range(ns):
                if time.perf_counter() >= deadline:
                    return float('inf')
                labels = a[s]
                result += float(u[s, ids, labels].sum())
                result += float(np.sum(w[s] * d[labels[:, None], labels[None, :]]))
            return result

        def keep(state):
            nonlocal best_value, best_state, answer
            score, z, a = state
            if not math.isfinite(score):
                return
            if score < best_value:
                best_value = score
                best_state = (score, z.copy(), a.copy())
                answer = {'hub_selection': z.astype(int).tolist(),
                          'scenario_allocations': {str(s): a[s].tolist() for s in range(ns)}}
            key = z.tobytes()
            for j, item in enumerate(pool):
                if item[1].tobytes() == key:
                    if score < item[0]:
                        pool[j] = (score, z.copy(), a.copy())
                    break
            else:
                pool.append((score, z.copy(), a.copy()))
            pool.sort(key=lambda item: item[0])
            del pool[pool_limit:]

        single_values = f + u.sum(axis=(0, 1)) + w.sum(axis=(1, 2)).sum() * diagonal
        hub = int(np.argmin(single_values))
        z = np.zeros(n, dtype=bool)
        z[hub] = True
        a = np.full((ns, n), hub, dtype=np.int64)
        singleton = (float(single_values[hub]), z, a)
        keep(singleton)
        identity = np.tile(ids, (ns, 1))
        all_z = np.ones(n, dtype=bool)
        all_state = (value(all_z, identity), all_z, identity)
        keep(all_state)
        if ns == 0:
            z = f < 0
            if not z.any():
                z[hub] = True
            keep((float(f[z].sum()), z, a))
            return answer

        def polish(state, cutoff):
            """Assignment descent with exact deltas under fixed hubs, with no iteration cap."""
            score, z, old_a = state
            hubs = np.flatnonzero(z)
            if (len(hubs) < 2 or time.perf_counter() >= cutoff
                    or n * n * len(hubs) > 30000000 * max(0.0, cutoff - time.perf_counter())):
                return state
            a = old_a.copy()
            movable = np.flatnonzero(~z).tolist()
            changed_any = False
            for s in rng.sample(range(ns), ns):
                if time.perf_counter() >= cutoff:
                    break
                labels = a[s]
                ws = w[s]
                # Disallow multithreaded matrix multiplication from prolonging a single non-interruptible computation.
                g = u[s][:, hubs].copy()
                g += np.einsum('ij,hj->ih', ws, d[hubs[:, None], labels[None, :]], optimize=False)
                g += np.einsum('ji,jh->ih', ws, d[labels[:, None], hubs[None, :]], optimize=False)
                positions = np.full(n, -1, dtype=np.int64)
                positions[hubs] = np.arange(len(hubs))
                while time.perf_counter() < cutoff:
                    changed = False
                    rng.shuffle(movable)
                    for i in movable:
                        if time.perf_counter() >= cutoff:
                            break
                        old = int(labels[i])
                        costs = g[i] + ws[i, i] * (diagonal[hubs] - d[hubs, old] - d[old, hubs])
                        q = int(np.argmin(costs))
                        gain = float(costs[q] - costs[positions[old]])
                        if gain >= -1e-10 * max(1.0, abs(float(costs[positions[old]]))):
                            continue
                        new = int(hubs[q])
                        g += ws[:, i, None] * (d[hubs, new] - d[hubs, old])[None, :]
                        g += ws[i, :, None] * (d[new, hubs] - d[old, hubs])[None, :]
                        labels[i] = new
                        changed = True
                        changed_any = True
                    if not changed:
                        break
            if changed_any and time.perf_counter() < deadline:
                candidate = (value(z, a), z, a)
                if candidate[0] < score:
                    keep(candidate)
                    return candidate
            return state

        def fuse(left, right):
            """Binary fusion of feasible solutions; non-submodular terms use an upper bound that is tight at both endpoints."""
            nonlocal last_cut_time, cut_enabled
            preferred = left if left[0] <= right[0] else right
            now = time.perf_counter()
            remaining = deadline - now
            # Reserve time for one recomputation of the original objective; when too large, switch to assignment descent and reconstruction.
            if (not cut_enabled or remaining <= 2.5 * last_cut_time
                    or ns * n * n > 3000000 * remaining):
                return preferred
            cut_start = now
            za, aa = left[1], left[2]
            zb, ab = right[1], right[2]
            different = np.flatnonzero(za != zb)
            total = ns * n + len(different)
            source, sink = total, total + 1
            linear = np.zeros(total, dtype=float)
            rows, cols, capacities = [], [], []
            hard_rows, hard_cols = [], []
            for s in range(ns):
                if time.perf_counter() >= deadline - 1.5 * last_cut_time:
                    return preferred
                old, new = aa[s], ab[s]
                if np.array_equal(old, new):
                    continue
                if symmetric:
                    flow = pair_flow[s]
                    e00 = flow * d[old[:, None], old[None, :]]
                    e10 = flow * d[new[:, None], old[None, :]]
                    e11 = flow * d[new[:, None], new[None, :]]
                else:
                    ws = w[s]
                    raw = ws * d[old[:, None], old[None, :]]
                    e00 = raw + raw.T
                    e10 = (ws * d[new[:, None], old[None, :]]
                           + ws.T * d[old[None, :], new[:, None]])
                    raw = ws * d[new[:, None], new[None, :]]
                    e11 = raw + raw.T
                change = e10 - e00
                change[ids, ids] = 0.0
                edges = 0.5 * (e10 + e10.T - e00 - e11)
                edges[ids, ids] = 0.0
                unary = (u[s, ids, new] - u[s, ids, old]
                         + w[s, ids, ids] * (diagonal[new] - diagonal[old])
                         + change.sum(axis=1) - edges.sum(axis=1))
                linear[s * n:(s + 1) * n] = unary
                np.maximum(edges, 0.0, out=edges)
                rr, cc = np.nonzero(edges)
                rows.append(rr + s * n)
                cols.append(cc + s * n)
                capacities.append(edges[rr, cc])
            for j, h in enumerate(different):
                aux = ns * n + j
                linear[aux] = f[h] if zb[h] else -f[h]
                assignment = ab if zb[h] else aa
                ss, ii = np.nonzero(assignment == h)
                nodes = ss * n + ii
                own = np.arange(ns) * n + h
                if zb[h]:
                    # Using a new hub means opening it; its own assignment is equivalent to its open/close switch.
                    hard_rows.extend([np.full(len(nodes), aux), own])
                    hard_cols.extend([nodes, np.full(ns, aux)])
                else:
                    # Closing an old hub means all its customers must migrate.
                    hard_rows.extend([nodes, np.full(ns, aux)])
                    hard_cols.extend([np.full(len(nodes), aux), own])
            positive = np.flatnonzero(linear > 0)
            negative = np.flatnonzero(linear < 0)
            rows.extend([np.full(len(positive), source), negative])
            cols.extend([positive, np.full(len(negative), sink)])
            capacities.extend([linear[positive], -linear[negative]])
            rr = np.concatenate(rows)
            cc = np.concatenate(cols)
            cap = np.concatenate(capacities)
            finite_sum = float(cap.sum())
            if not math.isfinite(finite_sum) or finite_sum <= 0:
                return preferred
            # Total finite edge capacity is at most 2^28 and hard edges are 2^30, avoiding integer flow overflow.
            scale = (1 << 28) / finite_sum
            weights = np.floor(cap * scale).astype(np.int64)
            nonzero = weights > 0
            rr, cc, weights = rr[nonzero], cc[nonzero], weights[nonzero]
            if hard_rows:
                hr = np.concatenate(hard_rows)
                hc = np.concatenate(hard_cols)
                rr = np.concatenate([rr, hr])
                cc = np.concatenate([cc, hc])
                weights = np.concatenate([weights, np.full(len(hr), 1 << 30, dtype=np.int64)])
            if time.perf_counter() >= deadline - max(last_cut_time, budget * 0.003):
                return preferred
            try:
                graph = coo_matrix((weights, (rr, cc)), shape=(total + 2, total + 2)).tocsr()
                flow_result = maximum_flow(graph, source, sink, method='dinic')
                residual = graph - flow_result.flow
                residual.eliminate_zeros()
                reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
            except Exception:
                cut_enabled = False
                return preferred
            take = np.ones(total + 2, dtype=bool)
            take[reachable] = False
            new_z = za.copy()
            new_z[different] = np.where(take[ns * n:total], zb[different], za[different])
            new_a = np.where(take[:ns * n].reshape(ns, n), ab, aa)
            # Every acceptance re-checks feasibility and the full unquantised original objective.
            valid = (new_z.any() and np.all(new_z[new_a])
                     and np.all(new_a[:, new_z] == ids[new_z][None, :]))
            if valid and not (np.array_equal(new_z, preferred[1]) and np.array_equal(new_a, preferred[2])):
                candidate = (value(new_z, new_a), new_z, new_a)
                if candidate[0] < preferred[0]:
                    preferred = candidate
                    keep(candidate)
            last_cut_time = max(time.perf_counter() - cut_start, last_cut_time * 0.75)
            return preferred

        def target_order(state):
            nonlocal last_score_time
            tick = time.perf_counter()
            if (deadline - tick < 2 * last_score_time
                    or ns * n * n * n > 30000000 * max(0.0, deadline - tick)):
                result = list(range(n))
                rng.shuffle(result)
                return result
            score, z, a = state
            estimate = f * (~z)
            free = np.flatnonzero(~z)
            for s in range(ns):
                if time.perf_counter() >= deadline:
                    break
                labels = a[s]
                g = u[s].copy()
                g += np.einsum('ij,hj->ih', w[s], d[:, labels], optimize=False)
                g += np.einsum('ji,jh->ih', w[s], d[labels, :], optimize=False)
                g += w[s, ids, ids, None] * (diagonal[None, :] - d[:, labels].T - d[labels, :])
                changes = g - g[ids, labels][:, None]
                estimate += np.minimum(changes[free], 0).sum(axis=0)
                # When a node is opened, it must point to itself in every scenario.
                estimate[free] += np.maximum(changes[free, free], 0)
            last_score_time = time.perf_counter() - tick
            # The estimate is only for ordering; no candidate hub is excluded.
            return np.argsort(estimate).tolist()

        def reconstruct(base, stagnation):
            z = base[1].copy()
            hubs = np.flatnonzero(z).tolist()
            closed = np.flatnonzero(~z).tolist()
            amplitude = min(n, max(1, int(math.ceil(math.sqrt(math.log1p(budget) * (1 + stagnation))))))
            count = rng.randint(1, amplitude)
            mode = rng.random()
            if mode < 0.7 and len(hubs) > 1:
                for h in rng.sample(hubs, min(count, len(hubs) - 1)):
                    z[h] = False
            if mode > 0.3 and closed:
                for h in rng.sample(closed, min(count, len(closed))):
                    z[h] = True
            if np.array_equal(z, base[1]):
                h = rng.randrange(n)
                if not z[h] or z.sum() > 1:
                    z[h] = not z[h]
            hubs = np.flatnonzero(z)
            a = base[2].copy()
            for s in range(ns):
                broken = ~z[a[s]]
                if rng.random() < 1.0 / (1.0 + math.sqrt(n)):
                    broken = ~z
                changed = np.flatnonzero(broken)
                if len(changed):
                    a[s, changed] = hubs[np.argmin(u[s][np.ix_(changed, hubs)], axis=1)]
                a[s, hubs] = hubs
            candidate = (value(z, a), z, a)
            keep(candidate)
            cutoff = min(deadline, time.perf_counter() + budget / (1 + math.sqrt(n * ns)))
            return polish(candidate, cutoff)

        # Keep both sparse and dense starts; the main loop is terminated only by the wall clock.
        current = best_state
        unused = [all_state, singleton]
        stagnation = 0
        while time.perf_counter() < deadline:
            old_value = current[0]
            for h in target_order(current):
                if time.perf_counter() >= deadline:
                    break
                z = np.zeros(n, dtype=bool)
                z[h] = True
                a = np.full((ns, n), h, dtype=np.int64)
                proposal = (float(single_values[h]), z, a)
                current = fuse(current, proposal)
                keep(current)
            if time.perf_counter() >= deadline:
                break
            cutoff = min(deadline, time.perf_counter() + budget / (1 + n + ns))
            current = polish(current, cutoff)
            if current[0] < old_value - 1e-9 * max(1.0, abs(old_value)):
                stagnation = 0
                continue
            stagnation += 1
            if unused:
                current = unused.pop(0)
            else:
                base = pool[rng.randrange(len(pool))] if rng.random() < 0.5 else best_state
                proposal = reconstruct(base, stagnation)
                merged = fuse(current, proposal)
                # On stagnation, switching search basins is allowed, but a better global record is never overwritten.
                current = merged if merged[0] < current[0] else proposal
                if len(pool) > 1 and time.perf_counter() < deadline:
                    current = fuse(current, pool[rng.randrange(len(pool))])
                keep(current)
        return answer
    except Exception:
        return answer
