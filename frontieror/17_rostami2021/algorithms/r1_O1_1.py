"""Cross-scenario joint graph cut and variable-neighbourhood hub search
Jointly optimises node reassignment and hub opening/closing, combining incremental coordinate descent, whole-cluster migration, perturbation restarts and small-scale branch and bound.
"""

import math
import random
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    try:
        budget = float(time_limit_s)
        if not math.isfinite(budget):
            budget = 0.0
        budget = max(0.0, budget)
    except Exception:
        budget = 0.0
    deadline = started + 0.965 * budget
    try:
        n = max(0, int(instance['n']))
        ns = max(0, int(instance['num_scenarios']))
    except Exception:
        return {'hub_selection': [1], 'scenario_allocations': {'0': [0]}}
    hub = 0
    try:
        hub = min(range(n), key=lambda h: float(instance['fixed_costs'][h]))
    except Exception:
        pass
    best_solution = {
        'hub_selection': [int(i == hub) for i in range(n)],
        'scenario_allocations': {str(s): [hub] * n for s in range(ns)},
    }
    if not n or time.perf_counter() >= deadline:
        return best_solution

    try:
        fixed = np.asarray(instance['fixed_costs'], dtype=float)
        dist = np.asarray(instance['distances'], dtype=float)
        probability = np.asarray(instance['scenarios_probability'], dtype=float)
        flow = np.asarray(instance['scenarios_demands'], dtype=float).reshape(ns, n, n)
        alpha = float(instance['cost_parameters_alpha'])
        chi = float(instance['cost_parameters_chi'])
        delta = float(instance['cost_parameters_delta'])
        idx = np.arange(n)
        # The input gives 2-D distances; collection and distribution coefficients use the same conversion as the reference implementation.
        unary = (chi * flow.sum(axis=2)[:, :, None] * dist[None, :, :]
                 + delta * flow.sum(axis=1)[:, :, None] * dist.T[None, :, :])
        unary *= probability[:, None, None]
        unary[:, idx, idx] = 0.0
        weights = flow * (alpha * probability[:, None, None])
        diagonal = np.diag(dist)
        wdiag = weights[:, idx, idx]
        expected = unary.sum(axis=0)
        rng = random.Random(104729 * n + 8191 * ns)
        strength = math.log1p(budget)
        elite_size = max(2, int(math.ceil(math.sqrt(n) * math.sqrt(1.0 + strength))))
        elite = {}
        best_value = float('inf')
        best_state = None
        graph_time = 0.0
        quantum = budget / (2.0 + math.sqrt(n) + strength)
        active_cap = min(n, max(2, int(math.sqrt(max(1.0, 400000.0 * quantum / max(1, ns))))))
        tolerance = 1e-9

        def cost(z, allocations):
            value = float(fixed[z].sum())
            for s in range(ns):
                a = allocations[s]
                value += float(unary[s, idx, a].sum())
                value += float((weights[s] * dist[a[:, None], a[None, :]]).sum())
            return value

        def record(z, a, value=None):
            nonlocal best_solution, best_value, best_state
            if value is None:
                value = cost(z, a)
            key = tuple(np.flatnonzero(z).tolist())
            old = elite.get(key)
            state = (z.copy(), a.copy(), value)
            if old is None or value < old[2]:
                elite[key] = state
                if len(elite) > elite_size:
                    del elite[max(elite, key=lambda k: elite[k][2])]
            if value < best_value:
                best_value = value
                best_state = state
                best_solution = {
                    'hub_selection': z.astype(int).tolist(),
                    'scenario_allocations': {str(s): a[s].tolist() for s in range(ns)},
                }
            return state

        if ns == 0:
            z = fixed < 0
            if not np.any(z):
                z[hub] = True
            record(z, np.empty((0, n), dtype=int))
            return best_solution
        if time.perf_counter() >= deadline:
            return best_solution

        single_values = fixed + expected.sum(axis=0) + weights.sum(axis=(1, 2)).dot(
            np.broadcast_to(diagonal, (ns, n)))
        single = int(np.argmin(single_values))
        z = np.zeros(n, dtype=bool)
        z[single] = True
        single_state = record(z, np.full((ns, n), single, dtype=int), float(single_values[single]))
        all_state = record(np.ones(n, dtype=bool), np.tile(idx, (ns, 1)))

        def repair(z, a):
            hubs = np.flatnonzero(z)
            if not len(hubs):
                z[single] = True
                hubs = np.array([single])
            a = a.copy()
            for s in range(ns):
                invalid = ~z[a[s]]
                if np.any(invalid):
                    nodes = np.flatnonzero(invalid)
                    a[s, nodes] = hubs[np.argmin(unary[s][np.ix_(nodes, hubs)], axis=1)]
                a[s, hubs] = hubs
            return a

        def improve(z, allocations, cutoff, shuffled=False):
            a = repair(z, allocations)
            hubs = np.flatnonzero(z)
            movable = np.flatnonzero(~z).tolist()
            if len(hubs) < 2 or not movable:
                return a
            dh = dist[np.ix_(hubs, hubs)]
            hubpos = np.full(n, -1, dtype=int)
            hubpos[hubs] = np.arange(len(hubs))
            for s in range(ns):
                now = time.perf_counter()
                if now >= cutoff:
                    break
                scene_end = now + (cutoff - now) / (ns - s)
                ws = weights[s]
                allocation = a[s]
                # Node energies for all candidate hubs; each reassignment incrementally updates only two sets of interactions.
                energy = unary[s][:, hubs].copy()
                energy += np.einsum('ij,kj->ik', ws, dist[hubs[:, None], allocation[None, :]])
                energy += np.einsum('ji,jk->ik', ws, dist[allocation[:, None], hubs[None, :]])
                order = movable[:]
                if not shuffled:
                    order.sort(key=lambda i: -float(np.abs(ws[i]).sum() + np.abs(ws[:, i]).sum()))
                while time.perf_counter() < scene_end:
                    if shuffled:
                        rng.shuffle(order)
                    changed = False
                    for i in order:
                        if time.perf_counter() >= scene_end:
                            break
                        old = int(allocation[i])
                        op = int(hubpos[old])
                        scores = energy[i] + ws[i, i] * (diagonal[hubs] - dist[hubs, old] - dist[old, hubs])
                        hp = int(np.argmin(scores))
                        if scores[hp] >= scores[op] - tolerance * (1.0 + abs(float(scores[op]))):
                            continue
                        new = int(hubs[hp])
                        energy += ws[:, i, None] * (dh[:, hp] - dh[:, op])[None, :]
                        energy += ws[i, :, None] * (dh[hp, :] - dh[op, :])[None, :]
                        allocation[i] = new
                        changed = True
                    if not changed:
                        break
                    order.reverse()
            return a

        def prepare(a):
            base = np.empty((ns, n))
            for s in range(ns):
                aa = a[s]
                interaction = weights[s] * dist[aa[:, None], aa[None, :]]
                base[s] = unary[s, idx, aa] + interaction.sum(axis=0) + interaction.sum(axis=1)
            return base

        def expansion(state, target, base, cutoff):
            nonlocal graph_time, active_cap
            begin = time.perf_counter()
            if begin + 1.6 * graph_time >= cutoff:
                return None
            z, a, value = state
            hubs = np.flatnonzero(z)
            hp = np.full(n, -1, dtype=int)
            hp[hubs] = np.arange(len(hubs))
            mapping = []
            selected = []
            linear_parts = []
            next_id = len(hubs)
            forced_zero = set()
            forced_one = set()
            constraint_from = []
            constraint_to = []
            for s in range(ns):
                if time.perf_counter() >= cutoff:
                    return None
                aa = a[s]
                ws = weights[s]
                gain = unary[s, :, target].copy() - base[s]
                gain += np.einsum('ij,j->i', ws, dist[target, aa])
                gain += np.einsum('ji,j->i', ws, dist[aa, target])
                gain += wdiag[s] * (diagonal[target] - dist[target, aa]
                                    - dist[aa, target] + diagonal[aa])
                if active_cap >= n:
                    nodes = idx
                else:
                    mandatory = hubs if len(hubs) < active_cap else np.empty(0, dtype=int)
                    available = np.ones(n, dtype=bool)
                    available[mandatory] = False
                    available[target] = False
                    eligible = np.flatnonzero(available)
                    count = max(0, active_cap - len(mandatory) - int(target not in mandatory))
                    rank_gain = gain.copy()
                    rank_gain[hubs] -= fixed[hubs] / max(1, ns)
                    if count < len(eligible):
                        eligible = eligible[np.argpartition(rank_gain[eligible], count)[:count]] if count else eligible[:0]
                    nodes = np.unique(np.concatenate((mandatory, eligible, np.array([target]))))
                ids = np.full(n, -1, dtype=int)
                active_hubs = nodes[z[nodes]]
                ids[active_hubs] = hp[active_hubs]
                free = nodes[~z[nodes]]
                ids[free] = np.arange(next_id, next_id + len(free))
                next_id += len(free)
                selected.append(nodes)
                mapping.append(ids)
                linear_parts.append(gain[nodes])
                forced_one.add(int(ids[target]))
                inactive = ids < 0
                for owner in np.unique(aa[inactive]):
                    if owner != target:
                        forced_zero.add(int(hp[owner]))
                clients = nodes[(~z[nodes]) & (aa[nodes] != target)]
                constraint_from.extend(ids[clients].tolist())
                constraint_to.extend(hp[aa[clients]].tolist())
            linear = np.zeros(next_id)
            rows = []
            columns = []
            capacities = []
            for s in range(ns):
                nodes = selected[s]
                ids = mapping[s][nodes]
                np.add.at(linear, ids, linear_parts[s])
                aa = a[s, nodes]
                left, right = np.triu_indices(len(nodes), 1)
                ai, aj = aa[left], aa[right]
                ni, nj = nodes[left], nodes[right]
                forward = weights[s, ni, nj]
                backward = weights[s, nj, ni]
                capacity = 0.5 * (
                    forward * (dist[ai, target] + dist[target, aj] - dist[ai, aj] - diagonal[target])
                    + backward * (dist[target, ai] + dist[aj, target] - dist[aj, ai] - diagonal[target]))
                # Even for non-metric data, improvements are accepted only by the original objective; graph-cut approximations are never used as objective values.
                capacity = np.maximum(capacity, 0.0)
                vi, vj = ids[left], ids[right]
                np.add.at(linear, vi, -capacity)
                np.add.at(linear, vj, -capacity)
                keep = capacity > 0.0
                rows.extend((vi[keep], vj[keep]))
                columns.extend((vj[keep], vi[keep]))
                capacities.extend((capacity[keep], capacity[keep]))
            for h in hubs:
                if h != target:
                    linear[hp[h]] -= fixed[h]
            source, sink = next_id, next_id + 1
            positive = np.flatnonzero(linear > 0.0)
            negative = np.flatnonzero(linear < 0.0)
            rows.extend((np.full(len(positive), source), negative))
            columns.extend((positive, np.full(len(negative), sink)))
            capacities.extend((linear[positive], -linear[negative]))
            rr = np.concatenate(rows).astype(int)
            cc = np.concatenate(columns).astype(int)
            cap = np.concatenate(capacities)
            scale = (2.0 ** 26) / max(1e-100, float(cap.sum()))
            cap = np.rint(cap * scale).astype(np.int64)
            force0 = np.asarray(sorted(forced_zero), dtype=int)
            force1 = np.asarray(sorted(forced_one), dtype=int)
            rr = np.concatenate((rr, np.asarray(constraint_from, dtype=int),
                                 np.full(len(force0), source), force1))
            cc = np.concatenate((cc, np.asarray(constraint_to, dtype=int),
                                 force0, np.full(len(force1), sink)))
            cap = np.concatenate((cap, np.full(len(constraint_from) + len(force0) + len(force1),
                                              2 ** 29, dtype=np.int64)))
            if time.perf_counter() + graph_time >= cutoff:
                return None
            graph = coo_matrix((cap, (rr, cc)), shape=(next_id + 2, next_id + 2)).tocsr()
            graph.eliminate_zeros()
            result = maximum_flow(graph, source, sink)
            residual = graph - result.flow
            residual.data = (residual.data > 0).astype(np.int64)
            residual.eliminate_zeros()
            reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
            switch = np.ones(next_id + 2, dtype=bool)
            switch[reachable] = False
            nz = z.copy()
            for h in hubs:
                if h != target and switch[hp[h]]:
                    nz[h] = False
            nz[target] = True
            na = a.copy()
            for s in range(ns):
                nodes = selected[s]
                moved = nodes[switch[mapping[s][nodes]]]
                na[s, moved] = target
                na[s, target] = target
            elapsed = time.perf_counter() - begin
            graph_time = max(elapsed, 0.8 * graph_time)
            if elapsed > quantum and active_cap > len(hubs) + 1:
                active_cap = max(len(hubs) + 1, int(active_cap * math.sqrt(max(0.1, quantum / elapsed))))
            elif elapsed < quantum / 3.0 and active_cap < n:
                active_cap = min(n, max(active_cap + 1, int(active_cap * 1.2)))
            if not np.all(nz[na]) or not np.all(na[:, nz] == idx[nz]):
                return None
            return nz, na, cost(nz, na)

        def relocation_scores(state, cutoff):
            z, a, value = state
            hubs = np.flatnonzero(z)
            closed = np.flatnonzero(~z)
            if not len(closed):
                return []
            scores = np.broadcast_to(fixed[closed], (len(hubs), len(closed))).copy()
            scores -= fixed[hubs, None]
            for s in range(ns):
                if time.perf_counter() >= cutoff:
                    return []
                aa = a[s]
                ws = weights[s]
                dd = dist[aa[:, None], aa[None, :]]
                node_gain = -unary[s, idx, aa]
                node_gain += (ws * (dist[:, aa] - dd)).sum(axis=1)
                node_gain += (ws * (dist[aa, :] - dd)).sum(axis=0)
                node_gain += wdiag[s] * (diagonal - dist[idx, aa] - dist[aa, idx] + diagonal[aa])
                for p, h in enumerate(hubs):
                    cluster = aa == h
                    into = ws[:, cluster].sum(axis=1)
                    out = ws[cluster, :].sum(axis=0)
                    agg_out = np.bincount(aa, weights=out, minlength=n)
                    agg_in = np.bincount(aa, weights=into, minlength=n)
                    internal = float(out[cluster].sum())
                    change = unary[s, cluster, :].sum(axis=0) - float(unary[s, cluster, h].sum())
                    change += np.einsum('ij,j->i', dist, agg_out) - float(dist[h].dot(agg_out))
                    change += np.einsum('ji,j->i', dist, agg_in) - float(dist[:, h].dot(agg_in))
                    change += internal * (diagonal - dist[:, h] - dist[h, :] + dist[h, h])
                    extra = node_gain.copy()
                    extra += into * (diagonal - dist[:, h] - dist[aa, idx] + dist[aa, h])
                    extra += out * (diagonal - dist[h, :] - dist[idx, aa] + dist[h, aa])
                    change += np.where(cluster, 0.0, extra)
                    scores[p] += change[closed]
            order = np.argsort(scores, axis=None)
            return [(float(scores.flat[q]), int(hubs[q // len(closed)]), int(closed[q % len(closed)]))
                    for q in order]

        def descend(state, cutoff):
            while time.perf_counter() < cutoff:
                z, a, value = state
                base = prepare(a)
                proxy = fixed.copy()
                old_unary = expected[idx, a[0]]
                proxy += np.minimum(expected - old_unary[:, None], 0.0).sum(axis=0)
                order = np.argsort(proxy).tolist()
                improved = False
                for target in order:
                    if time.perf_counter() >= cutoff:
                        return state
                    proposal = expansion(state, target, base, cutoff)
                    if proposal is None:
                        continue
                    nz, na, nv = proposal
                    if nv < state[2] - tolerance * (1.0 + abs(state[2])):
                        state = record(nz, na, nv)
                        end = min(cutoff, time.perf_counter() + quantum)
                        na = improve(nz, na, end)
                        state = record(nz, na)
                        base = prepare(na)
                        improved = True
                if improved:
                    continue
                # Explicit node removal lets the original cluster be split among several surviving hubs, rather than merged as a whole.
                z, a, value = state
                hubs = np.flatnonzero(z).tolist()
                rng.shuffle(hubs)
                if len(hubs) > 1:
                    for h in hubs:
                        if time.perf_counter() >= cutoff:
                            return state
                        nz = z.copy()
                        nz[h] = False
                        na = improve(nz, a, min(cutoff, time.perf_counter() + quantum))
                        nv = cost(nz, na)
                        if nv < value - tolerance * (1.0 + abs(value)):
                            state = record(nz, na, nv)
                            improved = True
                            break
                if improved:
                    continue
                swaps = relocation_scores(state, cutoff)
                width = max(1, int(math.ceil(math.sqrt(n) * (1.0 + strength))))
                for _, old, new in swaps[:width]:
                    if time.perf_counter() >= cutoff:
                        return state
                    nz = z.copy()
                    nz[old], nz[new] = False, True
                    na = a.copy()
                    na[na == old] = new
                    na[:, new] = new
                    na = improve(nz, na, min(cutoff, time.perf_counter() + quantum))
                    nv = cost(nz, na)
                    if nv < value - tolerance * (1.0 + abs(value)):
                        state = record(nz, na, nv)
                        improved = True
                        break
                if not improved:
                    return state
            return state

        # For small problems whose hub subsets can be enumerated within budget, keep an exact branch-and-bound phase.
        small = n < 24 and (2.0 ** n) <= max(1.0, budget * 256.0)
        heuristic_end = min(deadline, started + budget * 0.48) if small else deadline
        starts = [best_state, single_state, all_state]
        # A greedy facility solution computing only single-node collection/distribution costs, used as a complementary start.
        zz = np.zeros(n, dtype=bool)
        zz[single] = True
        near = expected[:, single].copy()
        while time.perf_counter() < heuristic_end:
            gains = fixed + np.minimum(expected - near[:, None], 0.0).sum(axis=0)
            gains[zz] = np.inf
            h = int(np.argmin(gains))
            if gains[h] >= 0.0:
                break
            zz[h] = True
            near = np.minimum(near, expected[:, h])
        if time.perf_counter() < heuristic_end:
            aa = repair(zz, single_state[1])
            aa = improve(zz, aa, min(heuristic_end, time.perf_counter() + quantum))
            starts.append(record(zz, aa))
        stagnation = 0
        while time.perf_counter() < heuristic_end:
            prior = best_value
            if starts:
                state = starts.pop(0)
            else:
                candidates = sorted(elite.values(), key=lambda state: state[2])
                rank = int((rng.random() ** 2) * len(candidates))
                state = candidates[rank]
                nz, aa = state[0].copy(), state[1].copy()
                amplitude = min(n, max(1, int(1 + math.sqrt(1.0 + strength) * math.log1p(stagnation))))
                count = rng.randint(1, amplitude)
                for h in rng.sample(range(n), count):
                    nz[h] = not nz[h]
                if not np.any(nz):
                    nz[rng.randrange(n)] = True
                aa = repair(nz, aa)
                if rng.random() < 1.0 / (1.0 + math.sqrt(n)):
                    hubs = np.flatnonzero(nz)
                    for s in range(ns):
                        aa[s] = np.array(rng.choices(hubs.tolist(), k=n))
                        aa[s, hubs] = hubs
                aa = improve(nz, aa, min(heuristic_end, time.perf_counter() + quantum), True)
                state = record(nz, aa)
            epoch_end = min(heuristic_end, time.perf_counter() + budget / (2.0 + math.sqrt(1.0 + strength)))
            descend(state, epoch_end)
            stagnation = 0 if best_value < prior - tolerance else stagnation + 1

        if small and time.perf_counter() < deadline:
            # Weighted coefficients already include scenario probabilities, so the bound and the original objective use exactly the same values.
            nonnegative = bool(np.all(weights >= 0) and np.all(dist >= 0))
            transfer_lower = 0.0 if nonnegative else float(
                np.maximum(weights, 0).sum() * dist.min()
                + np.minimum(weights, 0).sum() * dist.max())
            masks = []
            for mask in range(1, 1 << n):
                if time.perf_counter() >= deadline:
                    break
                hubs = np.array([i for i in range(n) if mask & (1 << i)], dtype=int)
                zz = np.zeros(n, dtype=bool)
                zz[hubs] = True
                lower = float(fixed[hubs].sum()) + transfer_lower
                for s in range(ns):
                    lower += float(unary[s][np.ix_(np.flatnonzero(~zz), hubs)].min(axis=1).sum())
                if lower < best_value:
                    masks.append((lower, mask))
            masks.sort()
            for lower, mask in masks:
                if time.perf_counter() >= deadline or lower >= best_value:
                    break
                hubs = np.array([i for i in range(n) if mask & (1 << i)], dtype=int)
                zz = np.zeros(n, dtype=bool)
                zz[hubs] = True
                aa = repair(zz, best_state[1])
                aa = improve(zz, aa, min(deadline, time.perf_counter() + quantum))
                record(zz, aa)
                movable = np.flatnonzero(~zz)
                for s in range(ns):
                    if time.perf_counter() >= deadline:
                        break
                    ws = weights[s]
                    order = sorted(movable.tolist(), key=lambda i: -float(np.abs(ws[i]).sum() + np.abs(ws[:, i]).sum()))
                    r = len(order)
                    if not r:
                        continue
                    eff = unary[s][np.ix_(order, hubs)].copy()
                    eff += wdiag[s, order, None] * diagonal[hubs]
                    for p, i in enumerate(order):
                        eff[p] += ws[i, hubs].dot(dist[np.ix_(hubs, hubs)].T)
                        eff[p] += ws[hubs, i].dot(dist[np.ix_(hubs, hubs)])
                    constant = float((ws[np.ix_(hubs, hubs)] * dist[np.ix_(hubs, hubs)]).sum())
                    pair_bound = np.zeros(r + 1)
                    for p in range(r - 1, -1, -1):
                        pair_bound[p] = pair_bound[p + 1]
                        for q in range(p + 1, r):
                            pair_bound[p] += float((ws[order[p], order[q]] * dist[np.ix_(hubs, hubs)]
                                                   + ws[order[q], order[p]] * dist[np.ix_(hubs, hubs)].T).min())
                    scene_best = float(unary[s, idx, aa[s]].sum()
                                       + (ws * dist[aa[s, :, None], aa[s, None, :]]).sum())
                    scene_allocation = aa[s].copy()
                    chosen = np.empty(r, dtype=int)

                    def branch(pos, accumulated, effective):
                        nonlocal scene_best, scene_allocation
                        if time.perf_counter() >= deadline:
                            return
                        bound = constant + accumulated + pair_bound[pos]
                        if pos < r:
                            bound += float(effective[pos:].min(axis=1).sum())
                        if bound >= scene_best - tolerance:
                            return
                        if pos == r:
                            scene_best = constant + accumulated
                            scene_allocation = aa[s].copy()
                            scene_allocation[order] = chosen
                            return
                        i = order[pos]
                        for hp0 in np.argsort(effective[pos]):
                            if time.perf_counter() >= deadline:
                                return
                            h = int(hubs[hp0])
                            chosen[pos] = h
                            nxt = effective.copy()
                            for q in range(pos + 1, r):
                                j = order[q]
                                nxt[q] += ws[j, i] * dist[hubs, h] + ws[i, j] * dist[h, hubs]
                            branch(pos + 1, accumulated + float(effective[pos, hp0]), nxt)
                    branch(0, 0.0, eff)
                    aa[s] = scene_allocation
                    record(zz, aa)
        return best_solution
    except Exception:
        return best_solution
