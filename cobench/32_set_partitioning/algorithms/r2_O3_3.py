"""Dual-guided ejection chains with adaptive-neighbourhood set partitioning
Reduces the problem by forced propagation and linear relaxation, shares reduced costs and the best complete partition, and interleaves ejection chains, conflict-region repair and global local-branching integer programming.
"""
import math
import numbers
import random
import time

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import csc_matrix, vstack
    _SCIPY = True
except Exception:
    _SCIPY = False


def _read(instance):
    if not isinstance(instance, dict):
        raise ValueError("input must be an object")
    n, m = instance.get("n"), instance.get("m")
    if any(isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0
           for v in (n, m)):
        raise ValueError("n and m must be non-negative integers")
    n, m = int(n), int(m)
    costs, covers = instance.get("costs"), instance.get("covers")
    if not isinstance(costs, (list, tuple)) or not isinstance(covers, (list, tuple)):
        raise ValueError("costs and covers must be arrays")
    if len(costs) != n or len(covers) != n:
        raise ValueError("array length does not match n")
    cc, rr = [], []
    for c, rows in zip(costs, covers):
        if isinstance(c, bool) or not isinstance(c, numbers.Real):
            raise ValueError("costs must be finite non-negative reals")
        c = float(c)
        if not math.isfinite(c) or c < 0:
            raise ValueError("costs must be finite non-negative reals")
        if not isinstance(rows, (list, tuple)):
            raise ValueError("cover sets must be arrays")
        seen = set()
        for r in rows:
            if isinstance(r, bool) or not isinstance(r, numbers.Integral):
                raise ValueError("task indices must be integers")
            r = int(r)
            if not 0 <= r < m or r in seen:
                raise ValueError("task index out of range or duplicated within a cover set")
            seen.add(r)
        cc.append(c)
        rr.append(tuple(sorted(seen)))
    return n, m, cc, rr


def _reduce(m, cols0, cost0, deadline):
    K = len(cols0)
    colrows = [set(c) for c in cols0]
    alive = [True] * K
    row_cols = [set() for _ in range(m)]
    for j, c in enumerate(cols0):
        for r in c:
            row_cols[r].add(j)
    active = [True] * m
    fixed = []
    stack = []

    def kill_col(j):
        if not alive[j]:
            return
        alive[j] = False
        for r in colrows[j]:
            s = row_cols[r]
            s.discard(j)
            if len(s) <= 1:
                stack.append(r)

    def process_stack():
        while stack:
            r = stack.pop()
            if not active[r]:
                continue
            s = row_cols[r]
            if len(s) == 0:
                return False
            if len(s) == 1:
                j = next(iter(s))
                fixed.append(j)
                rows = list(colrows[j])
                for rr in rows:
                    for jj in list(row_cols[rr]):
                        if jj != j:
                            kill_col(jj)
                alive[j] = False
                for rr in rows:
                    row_cols[rr].discard(j)
                    active[rr] = False
        return True

    for r in range(m):
        if len(row_cols[r]) <= 1:
            stack.append(r)
    if not process_stack():
        return None

    changed = True
    while changed and time.perf_counter() < deadline:
        changed = False
        # Containment relations among the columns incident to each task yield forced exclusions and redundant constraints.
        order = sorted((r for r in range(m) if active[r]), key=lambda r: len(row_cols[r]))
        for r1 in order:
            if time.perf_counter() >= deadline:
                break
            if not active[r1] or not row_cols[r1]:
                continue
            it = iter(row_cols[r1])
            j0 = next(it)
            cand = set(colrows[j0])
            cand.discard(r1)
            if not cand:
                continue
            for j in it:
                if time.perf_counter() >= deadline:
                    cand.clear()
                    break
                cand &= colrows[j]
                if not cand:
                    break
            if not cand:
                continue
            s1 = row_cols[r1]
            for r2 in cand:
                if time.perf_counter() >= deadline:
                    break
                if not active[r2] or r2 == r1:
                    continue
                extra = row_cols[r2] - s1
                for jj in extra:
                    kill_col(jj)
                for jj in row_cols[r2]:
                    colrows[jj].discard(r2)
                row_cols[r2] = set()
                active[r2] = False
                changed = True
            if not process_stack():
                return None
        if not process_stack():
            return None
        if time.perf_counter() >= deadline:
            break
        # Exclude a column if inserting it would block some other task.
        stampc = [-1] * K
        stampr = [-1] * m
        inj = [-1] * m
        rc = [0] * m
        colorder = [j for j in range(K) if alive[j]]
        colorder.sort(key=lambda j: -len(colrows[j]))
        for j in colorder:
            if time.perf_counter() >= deadline:
                break
            if not alive[j]:
                continue
            rows_j = colrows[j]
            for r in rows_j:
                inj[r] = j
            stampc[j] = j
            bad = False
            expired = False
            for r in rows_j:
                if time.perf_counter() >= deadline:
                    expired = True
                    break
                for jj in row_cols[r]:
                    if stampc[jj] != j:
                        if time.perf_counter() >= deadline:
                            expired = True
                            break
                        stampc[jj] = j
                        for r2 in colrows[jj]:
                            if inj[r2] != j:
                                if stampr[r2] != j:
                                    stampr[r2] = j
                                    rc[r2] = 0
                                rc[r2] += 1
                                if rc[r2] >= len(row_cols[r2]):
                                    bad = True
                                    break
                    if bad:
                        break
                if bad or expired:
                    break
            if expired:
                break
            if bad:
                kill_col(j)
                changed = True
                if not process_stack():
                    return None
        if not process_stack():
            return None

    act_rows = [r for r in range(m) if active[r]]
    for r in act_rows:
        if not row_cols[r]:
            return None
    rmap = {r: i for i, r in enumerate(act_rows)}
    best_of = {}
    for j in range(K):
        if not alive[j]:
            continue
        key = tuple(sorted(rmap[r] for r in colrows[j]))
        if not key:
            continue
        if key not in best_of or cost0[j] < cost0[best_of[key]]:
            best_of[key] = j
    cols = []
    ccost = []
    corig = []
    for key, j in best_of.items():
        cols.append(key)
        ccost.append(cost0[j])
        corig.append(j)
    return fixed, len(act_rows), cols, ccost, corig


class _Packing:
    """Ejects all conflicting columns on insertion and accumulates penalties on uncovered tasks."""
    def __init__(self, rows, costs, incident, key, reduced, budget, rng):
        self.rows, self.costs, self.incident = rows, costs, incident
        self.m, self.n, self.rng = len(incident), len(rows), rng
        self.unit = max(1e-12, math.fsum(costs) / max(1, sum(map(len, rows))))
        # Between complete partitions the dual potentials cancel out; in under-covered states they provide direction.
        self.search_cost = ([0.35 * c + 0.65 * d for c, d in zip(costs, reduced)]
                            if reduced is not None else costs)
        self.weight = [self.unit] * self.m
        self.prize = [self.unit * len(rr) for rr in rows]
        self.order = sorted(range(self.n), key=key.__getitem__)
        self.allowed = [True] * self.n
        self.width = max(2, min(self.n, int(math.sqrt(self.n * max(budget, 0.0))) + 1))
        self.elite = max(1, int(math.sqrt(self.width)))
        self.tenure = max(1, int(math.sqrt(self.m) * math.log1p(budget)
                                 / max(1.0, sum(map(len, rows)) / self.n)))
        self.pressure_time = max(1e-9, budget / (1.0 + math.sqrt(self.m)))
        self.step, self.penalty = 0, 1.0
        self.reset([])

    def reset(self, columns):
        self.owner = [-1] * self.m
        self.open = list(range(self.m))
        self.pos = list(range(self.m))
        self.chosen = set()
        self.value = 0.0
        self.tabu = [0] * self.n
        self.last_complete = time.perf_counter()
        for j in columns:
            self.insert(j, ())

    def insert(self, j, conflicts):
        rows, owner, pos, opened = self.rows, self.owner, self.pos, self.open
        for k in conflicts:
            self.chosen.remove(k)
            self.value -= self.costs[k]
            self.tabu[k] = self.step + self.tenure + self.rng.randrange(self.tenure + 1)
            for r in rows[k]:
                owner[r] = -1
                pos[r] = len(opened)
                opened.append(r)
        self.chosen.add(j)
        self.value += self.costs[j]
        for r in rows[j]:
            p = pos[r]
            if p >= 0:
                last = opened.pop()
                if p < len(opened):
                    opened[p] = last
                    pos[last] = p
                pos[r] = -1
            owner[r] = j

    def merges(self, until):
        for j in self.order:
            if time.perf_counter() >= until:
                break
            if j in self.chosen or not self.allowed[j]:
                continue
            conflicts = {self.owner[r] for r in self.rows[j]}
            if -1 in conflicts:
                continue
            if sum(len(self.rows[k]) for k in conflicts) == len(self.rows[j]) and \
                    self.costs[j] < math.fsum(self.costs[k] for k in conflicts):
                self.insert(j, conflicts)

    def run(self, until, offer, best_value):
        rng, rows, costs = self.rng, self.rows, self.costs
        while time.perf_counter() < until:
            self.step += 1
            if not self.open:
                offer(self.chosen)
                self.last_complete = time.perf_counter()
                self.penalty = max(0.25, self.penalty * 0.9)
                source = self.order
            else:
                r, s = rng.choice(self.open), rng.choice(self.open)
                if self.weight[s] / math.sqrt(len(self.incident[s])) > \
                        self.weight[r] / math.sqrt(len(self.incident[r])):
                    r = s
                increment = self.unit / math.sqrt(len(self.open))
                self.weight[r] += increment
                for j in self.incident[r]:
                    self.prize[j] += increment
                source = self.incident[r]
            if len(source) <= self.width:
                candidates = source
            else:
                candidates = source[:self.elite] + rng.sample(source, self.width - self.elite)
            best, fallback = None, None
            for j in candidates:
                if time.perf_counter() >= until:
                    break
                if j in self.chosen or not self.allowed[j]:
                    continue
                conflicts = {self.owner[r] for r in rows[j] if self.owner[r] >= 0}
                new_open = len(self.open) + sum(len(rows[k]) for k in conflicts) - len(rows[j])
                delta = costs[j] - math.fsum(costs[k] for k in conflicts)
                dual_delta = self.search_cost[j] - math.fsum(self.search_cost[k] for k in conflicts)
                change = math.fsum(self.prize[k] for k in conflicts) - self.prize[j]
                score = dual_delta + self.penalty * change
                score += rng.random() * self.unit * math.sqrt(len(rows[j])) * 0.35
                move = (score, j, conflicts)
                if fallback is None or score < fallback[0]:
                    fallback = move
                aspiration = new_open == 0 and self.value + delta < best_value()
                if self.tabu[j] > self.step and not aspiration:
                    continue
                if best is None or score < best[0]:
                    best = move
            if best is None:
                best = fallback
            if best is None or time.perf_counter() >= until:
                return
            self.insert(best[1], best[2])
            if not self.open:
                offer(self.chosen)
            else:
                elapsed = time.perf_counter() - self.last_complete
                self.penalty = min(1.0 + math.sqrt(self.m),
                                   max(self.penalty, 1.0 + math.sqrt(elapsed / self.pressure_time)))


def _dfs(rows, costs, incident, key, reduced, lower, until, offer, upper, first=False):
    """Iterative backtracking that branches on the task with the fewest candidates first, using both original costs and dual lower bounds."""
    m, nc = len(incident), len(rows)
    available = [True] * nc
    count = [len(v) for v in incident]
    covered = [False] * m
    path, frames = [], []
    value, dual_sum, left = 0.0, 0.0, m

    def push():
        r = min((r for r in range(m) if not covered[r]), key=count.__getitem__)
        if count[r] == 0:
            return False
        candidates = [j for j in incident[r] if available[j]]
        candidates.sort(key=key.__getitem__)
        frames.append([candidates, 0, None])
        return True

    if not push():
        return True
    while frames and time.perf_counter() < until:
        frame = frames[-1]
        if frame[2] is not None:
            j = path.pop()
            value -= costs[j]
            if reduced is not None:
                dual_sum -= max(0.0, reduced[j])
            for k in frame[2]:
                available[k] = True
                for r in rows[k]:
                    count[r] += 1
            for r in rows[j]:
                covered[r] = False
            left += len(rows[j])
            frame[2] = None
        if frame[1] == len(frame[0]):
            frames.pop()
            continue
        j = frame[0][frame[1]]
        frame[1] += 1
        ub = upper()
        if value + costs[j] > ub + 1e-9:
            continue
        if reduced is not None and lower + dual_sum + max(0.0, reduced[j]) > ub + 1e-9:
            continue
        removed = []
        for r in rows[j]:
            covered[r] = True
            for k in incident[r]:
                if available[k]:
                    available[k] = False
                    removed.append(k)
                    for s in rows[k]:
                        count[s] -= 1
        frame[2] = removed
        path.append(j)
        value += costs[j]
        if reduced is not None:
            dual_sum += max(0.0, reduced[j])
        left -= len(rows[j])
        if left == 0:
            accepted = offer(path)
            if first and accepted:
                return False
        else:
            push()
    return not frames


def _matrix(m, rows):
    ptr, rr = [0], []
    for col in rows:
        rr.extend(col)
        ptr.append(len(rr))
    return csc_matrix((np.ones(len(rr)), np.asarray(rr, dtype=np.int32),
                       np.asarray(ptr, dtype=np.int32)), shape=(m, len(rows)))


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    result = {"feasible": False, "x": None, "total_cost": None,
              "error": "No complete feasible solution found within the time limit yet; this does not prove infeasibility"}
    try:
        try:
            budget = float(time_limit_s)
            if not math.isfinite(budget) or budget < 0:
                budget = 0.0
        except Exception:
            budget = 0.0
        # Leave a wall-clock margin for both the native solver and final solution verification.
        deadline = start + 0.94 * budget
        try:
            n, m, raw, original = _read(instance)
        except Exception as exc:
            result["error"] = "Input error: " + str(exc)
            return result
        if m == 0:
            return {"feasible": True, "x": [0] * n, "total_cost": 0.0}

        def offer_original(selected):
            nonlocal result
            counts, x = bytearray(m), [0] * n
            for j in selected:
                if x[j]:
                    return False
                x[j] = 1
                for r in original[j]:
                    if counts[r]:
                        return False
                    counts[r] = 1
            if not all(counts):
                return False
            total = math.fsum(raw[j] for j in selected)
            if result["feasible"] and total >= result["total_cost"]:
                return True
            result = {"feasible": True, "x": x, "total_cost": total}
            return True

        representatives, single = {}, [-1] * m
        full = None
        for j, rr in enumerate(original):
            if not rr:
                continue
            if rr not in representatives or raw[j] < raw[representatives[rr]]:
                representatives[rr] = j
            if len(rr) == 1 and (single[rr[0]] < 0 or raw[j] < raw[single[rr[0]]]):
                single[rr[0]] = j
            if len(rr) == m and (full is None or raw[j] < raw[full]):
                full = j
        if all(j >= 0 for j in single):
            offer_original(single)
        if full is not None:
            offer_original([full])
        if result["feasible"] and result["total_cost"] == 0.0:
            return result
        indices0 = list(representatives.values())
        rows0 = [original[j] for j in indices0]
        scale = max(1.0, max(raw, default=0.0) / 1e6)
        costs0 = [raw[j] / scale for j in indices0]
        seen = set()
        for rr in rows0:
            seen.update(rr)
        if len(seen) != m:
            return {"feasible": False, "x": None, "total_cost": None}
        if time.perf_counter() >= deadline:
            return result
        reduction = _reduce(m, rows0, costs0, min(deadline, start + 0.045 * budget))
        if reduction is None:
            return {"feasible": False, "x": None, "total_cost": None}
        fixed, mr, rows, costs, back = reduction
        fixed_original = [indices0[j] for j in fixed]
        indices = [indices0[j] for j in back]
        nc = len(rows)
        if mr == 0:
            offer_original(fixed_original)
            return result
        incident = [[] for _ in range(mr)]
        single = [-1] * mr
        for j, rr in enumerate(rows):
            for r in rr:
                incident[r].append(j)
            if len(rr) == 1:
                single[rr[0]] = j
        incumbent, best = None, float("inf")
        lp_lower, lower = -float("inf"), -float("inf")
        reduced, fractional = None, None
        tolerance = 1e-8 / scale

        def offer(columns):
            nonlocal incumbent, best
            columns = list(columns)
            counts = bytearray(mr)
            for j in columns:
                for r in rows[j]:
                    if counts[r]:
                        return False
                    counts[r] = 1
            if not all(counts):
                return False
            value = math.fsum(costs[j] for j in columns)
            if value >= best:
                return False
            chosen_original = fixed_original + [indices[j] for j in columns]
            if not offer_original(chosen_original):
                return False
            incumbent, best = columns, value
            return True

        if result["feasible"]:
            offer([j for j, original_j in enumerate(indices) if result["x"][original_j]])
        if all(j >= 0 for j in single):
            offer(single)
        for j, rr in enumerate(rows):
            if len(rr) == mr:
                offer([j])
        if best == 0.0 or time.perf_counter() >= deadline:
            return result

        base = None
        if _SCIPY:
            try:
                base = _matrix(mr, rows)
                seconds = min(0.09 * budget, deadline - time.perf_counter())
                if seconds > 0:
                    lp = linprog(np.asarray(costs), A_eq=base, b_eq=np.ones(mr),
                                 bounds=(0, None), method="highs",
                                 options={"time_limit": seconds})
                    if lp.status == 2 and incumbent is None:
                        return {"feasible": False, "x": None, "total_cost": None}
                    if lp.status == 0 and lp.x is not None:
                        fractional = lp.x
                        dual = np.asarray(lp.eqlin.marginals, dtype=float)
                        reduced = (np.asarray(costs) - base.T.dot(dual)).tolist()
                        correction = math.fsum(min(v, 0.0) for v in reduced)
                        guard = 1e-9 * (1.0 + math.fsum(abs(float(v)) for v in dual))
                        lp_lower = math.fsum(float(v) for v in dual) + correction - guard
                        lower = lp_lower
                        offer(np.flatnonzero(fractional > 0.5).tolist())
            except Exception:
                pass
        unit = max(1e-12, math.fsum(costs) / max(1, sum(map(len, rows))))
        if reduced is None:
            key = [costs[j] / len(rows[j]) for j in range(nc)]
        else:
            key = [max(0.0, reduced[j]) / math.sqrt(len(rows[j]))
                   - 0.08 * unit * float(fractional[j])
                   + 0.001 * costs[j] / len(rows[j]) for j in range(nc)]
        for cols in incident:
            cols.sort(key=key.__getitem__)
        if incumbent is not None and best <= lower + tolerance:
            return result
        if time.perf_counter() < deadline:
            until = min(deadline, time.perf_counter() + budget * (0.04 if incumbent is None else 0.018))
            exhausted = _dfs(rows, costs, incident, key, reduced, lp_lower, until,
                             offer, lambda: best, first=incumbent is None)
            if exhausted:
                if incumbent is None:
                    return {"feasible": False, "x": None, "total_cost": None}
                return result
        if time.perf_counter() >= deadline:
            return result
        rng = random.Random(731 + 1009 * mr + nc)
        packing = _Packing(rows, costs, incident, key, reduced, budget, rng)
        packing.reset(incumbent if incumbent is not None else [j for j in single if j >= 0])
        until = min(deadline, time.perf_counter() + 0.055 * budget)
        packing.merges(until)
        if not packing.open:
            offer(packing.chosen)
        packing.run(until, offer, lambda: best)
        if best == 0.0 or (incumbent is not None and best <= lower + tolerance):
            return result

        def allowed(j):
            return reduced is None or lp_lower + max(0.0, reduced[j]) <= best + tolerance

        def optimize(seconds, radius=None, free=None, anchor=None):
            """All sub-models share the dual column filter; the free-region model keeps the schemes outside the region."""
            nonlocal lower
            if base is None or time.perf_counter() >= deadline:
                return None, False, False
            before = best
            current = incumbent if anchor is None else anchor
            outside = []
            whole = free is None or len(free) == mr
            if whole:
                candidates = [j for j in range(nc) if allowed(j)]
                if current is not None:
                    candidates = sorted(set(candidates).union(current))
                selected = [] if current is None else list(current)
                matrix = base[:, candidates]
                nrows = mr
            else:
                touched = set()
                for r in free:
                    touched.update(incident[r])
                selected = [j for j in current if rows[j][0] in free]
                outside = [j for j in current if rows[j][0] not in free]
                kept = set(selected)
                candidates = [j for j in touched if (j in kept or allowed(j))
                              and all(r in free for r in rows[j])]
                candidates.sort()
                task_ids = sorted(free)
                matrix = base[task_ids, :][:, candidates]
                nrows = len(task_ids)
            if not candidates or time.perf_counter() >= deadline:
                return None, False, False
            local_costs = np.asarray([costs[j] for j in candidates])
            blocks, lo, hi = [matrix], [np.ones(nrows)], [np.ones(nrows)]
            if incumbent is not None:
                cutoff = best - math.fsum(costs[j] for j in outside)
                blocks.append(csc_matrix(local_costs.reshape(1, -1)))
                lo.append(np.array([-np.inf]))
                hi.append(np.array([max(0.0, cutoff) + tolerance]))
            restricted = radius is not None and current is not None and radius < len(selected)
            if restricted:
                mapping = {j: k for k, j in enumerate(candidates)}
                keep = [mapping[j] for j in selected]
                keep_row = csc_matrix((np.ones(len(keep)),
                                       (np.zeros(len(keep), dtype=np.int32),
                                        np.asarray(keep, dtype=np.int32))),
                                      shape=(1, len(candidates)))
                blocks.append(keep_row)
                lo.append(np.array([len(selected) - radius], dtype=float))
                hi.append(np.array([np.inf]))
            constraint = LinearConstraint(vstack(blocks, format="csc"),
                                          np.concatenate(lo), np.concatenate(hi))
            remaining = min(seconds, deadline - time.perf_counter())
            if remaining <= 0:
                return None, False, False
            try:
                res = milp(local_costs, integrality=np.ones(len(candidates), dtype=np.uint8),
                           bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
                           constraints=constraint,
                           options={"time_limit": remaining, "mip_rel_gap": 0.0,
                                    "presolve": True, "disp": False})
            except Exception:
                return None, False, False
            valid = False
            if res.x is not None:
                trial = outside + [candidates[k] for k in np.flatnonzero(res.x > 0.5)]
                old_best = best
                valid = offer(trial)
                if not valid and incumbent is not None:
                    # An unimproved feasible optimum may still serve as a proof, but its coverage must be re-checked independently.
                    counts = [0] * mr
                    for j in trial:
                        for r in rows[j]:
                            counts[r] += 1
                    valid = all(v == 1 for v in counts)
                if valid and best < old_best:
                    packing.allowed = [allowed(j) for j in range(nc)]
            global_model = whole and not restricted
            proven = False
            if global_model:
                if res.status == 0 and valid:
                    proven = True
                elif res.status == 2 and incumbent is None:
                    result.clear()
                    result.update({"feasible": False, "x": None, "total_cost": None})
                    proven = True
                bound = getattr(res, "mip_dual_bound", None)
                if bound is not None and math.isfinite(float(bound)):
                    lower = max(lower, min(before, float(bound)) - tolerance)
            return res.status, best < before, proven

        # The global model uses the upper bound from the ejection chain and the dual column filter, rather than independently restarting another algorithm.
        if base is not None and time.perf_counter() < deadline:
            _, _, proven = optimize(budget * (0.32 if incumbent is not None else 0.42))
            if proven:
                return result
        if incumbent is not None:
            packing.reset(incumbent)
        length = len(incumbent) if incumbent is not None else mr
        radius = max(1, min(length, int(math.log1p(length) * math.sqrt(math.log1p(budget))) + 1))
        region_size = min(mr, max(1, int(mr * min(0.6, math.log1p(budget) / 12.0))))
        quantum = budget / (2.0 + math.log1p(budget) + math.sqrt(math.log1p(nc / mr)))
        use_region = True

        def region(until):
            """Expands along ejection-chain gaps and low reduced-cost columns, and closes the cover sets of the base schemes."""
            anchor = list(incumbent) if incumbent is not None else list(packing.chosen)
            owner = [-1] * mr
            for j in anchor:
                for r in rows[j]:
                    owner[r] = j
            free, frontier = set(), []

            def add(r):
                if r in free:
                    return
                j = owner[r]
                block = (r,) if j < 0 else rows[j]
                for s in block:
                    if s not in free:
                        free.add(s)
                        frontier.append(s)

            if incumbent is None:
                for r in range(mr):
                    if owner[r] < 0:
                        add(r)
            if packing.open:
                seed = max(packing.open, key=lambda r: packing.weight[r])
            elif anchor:
                j = max(rng.choices(anchor, k=max(1, int(math.log1p(len(anchor))))),
                        key=lambda j: key[j])
                seed = rng.choice(rows[j])
            else:
                seed = rng.randrange(mr)
            add(seed)
            # Include in the free region both the current ejection-chain changes and their conflicts with the base solution.
            changed = list(packing.chosen.difference(anchor))
            rng.shuffle(changed)
            for j in changed:
                if len(free) >= region_size or time.perf_counter() >= until:
                    break
                if any(r in free for r in rows[j]):
                    for r in rows[j]:
                        add(r)
            while len(free) < region_size and len(free) < mr and time.perf_counter() < until:
                r = rng.choice(frontier)
                cols = incident[r]
                j = min(rng.choices(cols, k=max(1, int(math.log1p(budget)) + 1)),
                        key=key.__getitem__)
                before_size = len(free)
                if allowed(j):
                    for s in rows[j]:
                        add(s)
                if len(free) == before_size:
                    add(rng.randrange(mr))
            return free, anchor

        while time.perf_counter() < deadline:
            if incumbent is not None and (best == 0.0 or best <= lower + tolerance):
                break
            phase_start = time.perf_counter()
            if base is None:
                packing.run(min(deadline, phase_start + 0.3 * quantum), offer, lambda: best)
                exhausted = _dfs(rows, costs, incident, key, reduced, lp_lower,
                                 min(deadline, time.perf_counter() + 0.7 * quantum),
                                 offer, lambda: best)
                if exhausted:
                    if incumbent is None:
                        result = {"feasible": False, "x": None, "total_cost": None}
                    break
                quantum *= 1.3
                continue
            packing.allowed = [allowed(j) for j in range(nc)]
            old_best = best
            packing.run(min(deadline, phase_start + quantum * 0.17), offer, lambda: best)
            if time.perf_counter() >= deadline:
                break
            if best < old_best:
                radius = max(1, min(radius, len(incumbent)))
            call_start = time.perf_counter()
            if use_region:
                free, anchor = region(min(deadline, call_start + quantum * 0.07))
                status, improved, proven = optimize(quantum * 0.76, free=free, anchor=anchor)
                elapsed = time.perf_counter() - call_start
                if status in (0, 2) and elapsed < quantum * 0.4:
                    region_size = min(mr, max(region_size + 1, int(region_size * 1.3)))
                elif status not in (0, 2):
                    region_size = max(1, int(region_size * 0.85))
            else:
                length = len(incumbent) if incumbent is not None else mr
                use_whole = incumbent is None or radius >= length
                if rng.random() < 1.0 / (1.0 + math.sqrt(max(0.0, budget))):
                    use_whole = True
                status, improved, proven = optimize(quantum * 0.83,
                                                      None if use_whole else radius)
                if status in (0, 2) and not improved:
                    radius = min(length, radius + max(1, radius // 2))
                elif status not in (0, 2):
                    quantum *= 1.2
                    if not use_whole:
                        radius = max(1, int(radius / 1.3))
            use_region = not use_region
            if proven:
                break
            if incumbent is not None and (improved or rng.random() < 0.4):
                packing.reset(incumbent)
        return result
    except Exception:
        # Only a complete partition that passes verification against the original covers may replace result.
        return result
