"""Ejection-chain and global local-branching set partitioning
Searches disjoint scheme sets under a dynamic uncovered-task penalty, uses integer programming to progressively enlarge the number of selected schemes that may be replaced, and always keeps the best complete partition.
"""
import math
import numbers
import random
import time

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
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


class _Packing:
    """Maintains only disjoint schemes; inserting a column ejects every conflicting column."""
    def __init__(self, rows, costs, incident, budget, rng):
        self.rows, self.costs, self.incident = rows, costs, incident
        self.m, self.n = len(incident), len(rows)
        self.rng = rng
        self.unit = math.fsum(costs) / max(1, sum(map(len, rows)))
        if self.unit <= 0:
            self.unit = 1.0
        self.weight = [self.unit] * self.m
        self.prize = [self.unit * len(r) for r in rows]
        self.owner = [-1] * self.m
        self.open = list(range(self.m))
        self.pos = list(range(self.m))
        self.chosen = set()
        self.value = 0.0
        self.tabu = [0] * self.n
        self.step = 0
        self.penalty = 1.0
        mean_size = sum(map(len, rows)) / max(1, self.n)
        self.tenure = max(1, int(math.sqrt(self.m) * math.log1p(budget)
                                 / max(1.0, mean_size)))
        self.width = max(2, min(self.n, int(math.sqrt(self.n * max(budget, 0.01))) + 1))
        self.last_complete = time.perf_counter()
        self.order = sorted(range(self.n), key=lambda j: costs[j] / len(rows[j]))

    def reset(self, columns):
        self.owner = [-1] * self.m
        self.open = list(range(self.m))
        self.pos = list(range(self.m))
        self.chosen = set()
        self.value = 0.0
        for j in columns:
            self.insert(j, set())
        self.tabu = [0] * self.n

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

    def greedy_merges(self, until):
        for j in self.order:
            if time.perf_counter() >= until:
                break
            if j in self.chosen:
                continue
            conflicts = {self.owner[r] for r in self.rows[j]}
            if -1 in conflicts:
                continue
            if sum(len(self.rows[k]) for k in conflicts) != len(self.rows[j]):
                continue
            if self.costs[j] < math.fsum(self.costs[k] for k in conflicts):
                self.insert(j, conflicts)

    def run(self, until, offer, best_value):
        rows, costs, rng = self.rows, self.costs, self.rng
        while time.perf_counter() < until:
            self.step += 1
            complete = not self.open
            if complete:
                offer(self.chosen)
                self.last_complete = time.perf_counter()
                self.penalty = max(0.25, self.penalty * 0.9)
                # Candidates span all tasks; no fixed block of tasks is targeted.
                candidates = (self.order if self.n <= self.width else
                              rng.sample(self.order, self.width))
            else:
                r = self.open[rng.randrange(len(self.open))]
                s = self.open[rng.randrange(len(self.open))]
                if self.weight[s] / math.sqrt(len(self.incident[s])) > \
                        self.weight[r] / math.sqrt(len(self.incident[r])):
                    r = s
                increment = self.unit / math.sqrt(max(1, len(self.open)))
                self.weight[r] += increment
                for j in self.incident[r]:
                    self.prize[j] += increment
                lst = self.incident[r]
                candidates = lst if len(lst) <= self.width else rng.sample(lst, self.width)
            best = None
            fallback = None
            for j in candidates:
                if time.perf_counter() >= until:
                    break
                if j in self.chosen:
                    continue
                conflicts = {self.owner[r] for r in rows[j] if self.owner[r] >= 0}
                removed_size = sum(len(rows[k]) for k in conflicts)
                new_open = len(self.open) + removed_size - len(rows[j])
                delta = costs[j] - math.fsum(costs[k] for k in conflicts)
                change = math.fsum(self.prize[k] for k in conflicts) - self.prize[j]
                score = delta + self.penalty * change
                score += rng.random() * self.unit * math.sqrt(len(rows[j]))
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
                # Completeness pressure grows with wall-clock time spent unrepaired; no cap on search steps.
                elapsed = time.perf_counter() - self.last_complete
                self.penalty = min(1.0 + math.sqrt(self.m),
                                   max(self.penalty, 1.0 + math.sqrt(elapsed)))


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    result = {"feasible": False, "x": None, "total_cost": None,
              "error": "No feasible solution found yet; this does not prove infeasibility"}
    try:
        try:
            budget = float(time_limit_s)
            if not math.isfinite(budget) or budget <= 0:
                budget = 0.0
        except Exception:
            budget = 0.0
        deadline = start + budget * 0.94
        try:
            n, m, original_costs, original_rows = _read(instance)
        except Exception as exc:
            result["error"] = "Input error: " + str(exc)
            return result
        if m == 0:
            return {"feasible": True, "x": [0] * n, "total_cost": 0.0}

        # Keep only the cheapest representative of identical covers; empty columns with non-negative cost are never needed.
        representatives = {}
        for j, rr in enumerate(original_rows):
            if rr and (rr not in representatives or
                       original_costs[j] < original_costs[representatives[rr]]):
                representatives[rr] = j
        indices = list(representatives.values())
        rows = [original_rows[j] for j in indices]
        raw_costs = [original_costs[j] for j in indices]
        nc = len(indices)
        incident = [[] for _ in range(m)]
        singles = [-1] * m
        for j, rr in enumerate(rows):
            for r in rr:
                incident[r].append(j)
            if len(rr) == 1:
                singles[rr[0]] = j
        if any(not cols for cols in incident):
            return {"feasible": False, "x": None, "total_cost": None}
        scale = max(1.0, max(raw_costs, default=0.0) / 1000000.0)
        costs = [c / scale for c in raw_costs]
        incumbent = [None]
        incumbent_value = [float("inf")]

        def offer(columns):
            nonlocal result
            columns = list(columns)
            counts = bytearray(m)
            for j in columns:
                if not 0 <= j < nc:
                    return False
                for r in rows[j]:
                    if counts[r]:
                        return False
                    counts[r] = 1
            if not all(counts):
                return False
            value = math.fsum(costs[j] for j in columns)
            if incumbent[0] is not None and value >= incumbent_value[0]:
                return False
            total = math.fsum(raw_costs[j] for j in columns)
            x = [0] * n
            for j in columns:
                x[indices[j]] = 1
            result = {"feasible": True, "x": x, "total_cost": total}
            incumbent[0] = columns
            incumbent_value[0] = value
            return True

        if all(j >= 0 for j in singles):
            offer(singles)
        for j, rr in enumerate(rows):
            if len(rr) == m:
                offer([j])
        if incumbent_value[0] == 0.0 or time.perf_counter() >= deadline:
            return result

        rng = random.Random(631 + 1009 * m + nc)
        packing = _Packing(rows, costs, incident, budget, rng)
        packing.reset(incumbent[0] if incumbent[0] is not None else
                      [j for j in singles if j >= 0])
        first_until = min(deadline, start + budget * 0.09)
        packing.greedy_merges(first_until)
        if not packing.open:
            offer(packing.chosen)
        packing.run(first_until, offer, lambda: incumbent_value[0])
        if incumbent_value[0] == 0.0:
            return result

        base = None
        if _SCIPY and time.perf_counter() < deadline:
            ptr = [0]
            row_ids = []
            for rr in rows:
                row_ids.extend(rr)
                ptr.append(len(row_ids))
            base = csc_matrix((np.ones(len(row_ids)),
                               np.asarray(row_ids, dtype=np.int32),
                               np.asarray(ptr, dtype=np.int32)), shape=(m, nc))
            objective = np.asarray(costs)
            cost_row = csc_matrix(objective.reshape(1, nc))
            integrality = np.ones(nc, dtype=np.uint8)
            bounds = Bounds(np.zeros(nc), np.ones(nc))

        def optimize(radius, seconds):
            """Radius limits the number of old schemes removed; no specified task is frozen."""
            if base is None or time.perf_counter() >= deadline:
                return None, False, False
            blocks = [base]
            lower = [np.ones(m)]
            upper = [np.ones(m)]
            whole = incumbent[0] is None or radius >= len(incumbent[0])
            has_cutoff = incumbent[0] is not None
            if has_cutoff:
                blocks.append(cost_row)
                lower.append(np.array([-np.inf]))
                upper.append(np.array([max(0.0, incumbent_value[0] - 1e-7 / scale)]))
            if not whole:
                selected = incumbent[0]
                keep_row = csc_matrix((np.ones(len(selected)),
                                       (np.zeros(len(selected), dtype=np.int32),
                                        np.asarray(selected, dtype=np.int32))), shape=(1, nc))
                blocks.append(keep_row)
                lower.append(np.array([len(selected) - radius], dtype=float))
                upper.append(np.array([np.inf]))
            matrix = base if len(blocks) == 1 else vstack(blocks, format="csc")
            constraints = LinearConstraint(matrix, np.concatenate(lower), np.concatenate(upper))
            remaining = min(seconds, deadline - time.perf_counter())
            if remaining <= 0:
                return None, False, False
            try:
                res = milp(objective, integrality=integrality, bounds=bounds,
                           constraints=constraints,
                           options={"time_limit": remaining, "mip_rel_gap": 0.0,
                                    "presolve": True, "disp": False})
            except Exception:
                return None, False, False
            improved = False
            if res.x is not None:
                selected = np.flatnonzero(res.x > 0.5).tolist()
                improved = offer(selected)
            closed = res.status == 2 or (res.status == 0 and res.x is not None)
            if whole and closed:
                if incumbent[0] is not None:
                    return res.status, improved, True
                if res.status == 2 and not has_cutoff:
                    result.clear()
                    result.update({"feasible": False, "x": None, "total_cost": None})
                    return res.status, improved, True
            return res.status, improved, False

        if base is not None:
            initial_share = 0.22 if incumbent[0] is not None else 0.40
            _, _, proven = optimize(nc, min(budget * initial_share,
                                           max(0.0, deadline - time.perf_counter())))
            if proven:
                return result
        if incumbent[0] is not None:
            packing.reset(incumbent[0])

        length = len(incumbent[0]) if incumbent[0] is not None else m
        radius = max(1, min(length, int(math.log1p(length) *
                                        math.sqrt(math.log1p(budget))) + 1))
        # A larger total budget enlarges the per-solve time and initial search radius; the number of rounds is unbounded.
        quantum = max(0.0, math.sqrt(budget) /
                      (1.0 + math.sqrt(math.log1p(nc) / max(1.0, math.log1p(m)))))
        mip_share = 0.8
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if base is None:
                packing.run(deadline, offer, lambda: incumbent_value[0])
                if len(packing.chosen) == nc and not packing.open:
                    break
                continue
            packing.run(min(deadline, now + quantum * (1.0 - mip_share)),
                        offer, lambda: incumbent_value[0])
            if incumbent_value[0] == 0.0:
                break
            if time.perf_counter() >= deadline:
                break
            length = len(incumbent[0]) if incumbent[0] is not None else m
            whole = incumbent[0] is None or radius >= length
            if rng.random() < 1.0 / (1.0 + math.sqrt(budget)):
                whole = True
            search_radius = nc if whole else min(radius, length)
            previous = incumbent_value[0]
            status, improved, proven = optimize(search_radius, quantum * mip_share)
            if proven:
                break
            if improved:
                packing.reset(incumbent[0])
                radius = max(1, min(len(incumbent[0]), radius))
            elif status in (0, 2):
                radius = min(length, radius + max(1, radius // 2))
            else:
                # Give hard neighbourhoods more time while shrinking the replacement count, to avoid getting stuck in the same large model.
                quantum *= 1.15
                if not whole:
                    radius = max(1, int(radius / 1.35))
            if previous == incumbent_value[0] and incumbent[0] is not None:
                if rng.random() < 0.5:
                    packing.reset(incumbent[0])
        return result
    except Exception:
        # Every verified complete solution is stored directly in result, so later failures cannot erase it.
        return result
