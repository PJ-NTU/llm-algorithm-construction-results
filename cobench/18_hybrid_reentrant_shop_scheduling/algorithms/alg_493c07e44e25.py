# Library slot 5/10  id=alg_493c07e44e25
# score=309.6579557769088  feasible_rate=1.0
# score_vec=[334.6, 808.5316687324524, 160.6186412334189, 161.22191595559235, 205.35708088647667, 197.6442509611474, 223.40000000000003, 345.8579999999999, 326.517, 332.831]

"""Anytime heuristic for the hybrid reentrant shop scheduling problem.

A complete list-scheduled solution is built before any optimization, then
simulated annealing explores primary-machine assignments and the remote-server
order until the wall-clock deadline.  Initialization and main processing always
follow natural job order on each primary machine.
"""

import heapq
import math
import random
import time


def _format_solution(assignment, server_order, s1, s2, s3, makespan):
    n = len(assignment)
    return {
        "machine_assignment": {
            "job_%d" % j: int(assignment[j]) for j in range(n)
        },
        "server_sequence": ["job_%d" % j for j in server_order],
        "start_times": {
            "job_%d" % j: [float(s1[j]), float(s2[j]), float(s3[j])]
            for j in range(n)
        },
        "makespan": float(makespan),
    }


def _fallback_solution(n, m, p1, p2, p3):
    """A fast, conservative schedule used before the search starts."""
    if n == 0:
        return _format_solution([], [], [], [], [], 0.0)

    assignment = [j % m for j in range(n)]
    s1 = [0.0] * n
    init_end = [0.0] * m
    for j in range(n):
        k = assignment[j]
        s1[j] = init_end[k]
        init_end[k] += p1[j]

    server_order = list(range(n))
    s2 = [0.0] * n
    server_end = 0.0
    for j in server_order:
        start = server_end
        release = s1[j] + p1[j]
        if release > start:
            start = release
        s2[j] = start
        server_end = start + p2[j]

    s3 = [0.0] * n
    main_end = init_end[:]
    for j in range(n):
        k = assignment[j]
        start = main_end[k]
        release = s2[j] + p2[j]
        if release > start:
            start = release
        s3[j] = start
        main_end[k] = start + p3[j]

    return _format_solution(
        assignment, server_order, s1, s2, s3, max(main_end)
    )


def _jobs_by_machine(assignment, m):
    """Group jobs by primary machine while retaining natural job order."""
    jobs_by_machine = [[] for _ in range(m)]
    for j, k in enumerate(assignment):
        jobs_by_machine[k].append(j)
    return jobs_by_machine


def _initialization_schedule(assignment, _init_rank, p1, m):
    n = len(assignment)
    jobs_by_machine = _jobs_by_machine(assignment, m)
    s1 = [0.0] * n
    init_end = [0.0] * m

    for k in range(m):
        current = 0.0
        for j in jobs_by_machine[k]:
            s1[j] = current
            current += p1[j]
        init_end[k] = current
    return s1, init_end, jobs_by_machine


def _evaluate(assignment, init_rank, server_order, _main_rank, p1, p2, p3, m):
    """Build the earliest feasible schedule represented by a search state."""
    n = len(assignment)
    s1, init_end, jobs_by_machine = _initialization_schedule(
        assignment, init_rank, p1, m
    )

    s2 = [0.0] * n
    setup_end = [0.0] * n
    current = 0.0
    for j in server_order:
        release = s1[j] + p1[j]
        if release > current:
            current = release
        s2[j] = current
        current += p2[j]
        setup_end[j] = current

    s3 = [0.0] * n
    machine_end = init_end[:]

    # Main processing starts only after every initialization on the machine is
    # complete, then follows the same natural job order without overlap.
    for k in range(m):
        current = init_end[k]
        for j in jobs_by_machine[k]:
            release = setup_end[j]
            if release > current:
                current = release
            s3[j] = current
            current += p3[j]
        machine_end[k] = current

    makespan = max(machine_end) if machine_end else 0.0
    return makespan, s1, s2, s3


def _lpt_assignment(weights, m, job_order):
    loads = [0.0] * m
    assignment = [0] * len(weights)
    for j in job_order:
        k = min(range(m), key=lambda machine: (loads[machine], machine))
        assignment[j] = k
        loads[k] += weights[j]
    return assignment


def _improve_load_balance(assignment, weights, m, deadline):
    """Best-improvement exchange search on the dominant load bound."""
    n = len(assignment)
    loads = [0.0] * m
    for j in range(n):
        loads[assignment[j]] += weights[j]

    while time.time() < deadline:
        old_max = max(loads)
        best_max = old_max
        best_change = None

        for j in range(n):
            if (j & 15) == 0 and time.time() >= deadline:
                return assignment
            source = assignment[j]
            for target in range(m):
                if target == source:
                    continue
                changed_source = loads[source] - weights[j]
                changed_target = loads[target] + weights[j]
                candidate_max = 0.0
                for k in range(m):
                    value = loads[k]
                    if k == source:
                        value = changed_source
                    elif k == target:
                        value = changed_target
                    if value > candidate_max:
                        candidate_max = value
                if candidate_max + 1e-10 < best_max:
                    best_max = candidate_max
                    best_change = (0, j, target)

        for a in range(n):
            if (a & 7) == 0 and time.time() >= deadline:
                return assignment
            ka = assignment[a]
            for b in range(a + 1, n):
                kb = assignment[b]
                if ka == kb:
                    continue
                load_a = loads[ka] - weights[a] + weights[b]
                load_b = loads[kb] - weights[b] + weights[a]
                candidate_max = 0.0
                for k in range(m):
                    value = loads[k]
                    if k == ka:
                        value = load_a
                    elif k == kb:
                        value = load_b
                    if value > candidate_max:
                        candidate_max = value
                if candidate_max + 1e-10 < best_max:
                    best_max = candidate_max
                    best_change = (1, a, b)

        # A 2-for-1 or 2-for-2 exchange is the useful next neighborhood after
        # single moves and swaps stall.  Only a currently heaviest machine must
        # be searched: every strict makespan improvement has to lighten one.
        if best_change is None and time.time() < deadline:
            heavy = max(range(m), key=lambda machine: loads[machine])
            heavy_jobs = [j for j in range(n) if assignment[j] == heavy]
            heavy_subsets = []
            for pos, a in enumerate(heavy_jobs):
                heavy_subsets.append((weights[a], (a,)))
                for second_pos, b in enumerate(heavy_jobs[pos + 1:], pos + 1):
                    heavy_subsets.append((weights[a] + weights[b], (a, b)))
                    if len(heavy_jobs) <= 14:
                        for c in heavy_jobs[second_pos + 1:]:
                            heavy_subsets.append(
                                (weights[a] + weights[b] + weights[c], (a, b, c))
                            )

            checked = 0
            for target in range(m):
                if target == heavy:
                    continue
                target_jobs = [j for j in range(n) if assignment[j] == target]
                target_subsets = [(0.0, ())]
                for pos, a in enumerate(target_jobs):
                    target_subsets.append((weights[a], (a,)))
                    for second_pos, b in enumerate(target_jobs[pos + 1:], pos + 1):
                        target_subsets.append((weights[a] + weights[b], (a, b)))
                        if len(target_jobs) <= 14:
                            for c in target_jobs[second_pos + 1:]:
                                target_subsets.append(
                                    (
                                        weights[a] + weights[b] + weights[c],
                                        (a, b, c),
                                    )
                                )

                other_max = 0.0
                for k in range(m):
                    if k != heavy and k != target and loads[k] > other_max:
                        other_max = loads[k]
                for weight_out, jobs_out in heavy_subsets:
                    for weight_in, jobs_in in target_subsets:
                        checked += 1
                        if (checked & 1023) == 0 and time.time() >= deadline:
                            return assignment
                        load_heavy = loads[heavy] - weight_out + weight_in
                        load_target = loads[target] + weight_out - weight_in
                        candidate_max = max(other_max, load_heavy, load_target)
                        if candidate_max + 1e-10 < best_max:
                            best_max = candidate_max
                            best_change = (
                                2, heavy, target, jobs_out, jobs_in,
                                load_heavy, load_target,
                            )

        if best_change is None:
            break
        if best_change[0] == 0:
            j, target = best_change[1], best_change[2]
            source = assignment[j]
            loads[source] -= weights[j]
            loads[target] += weights[j]
            assignment[j] = target
        else:
            if best_change[0] == 1:
                a, b = best_change[1], best_change[2]
                ka, kb = assignment[a], assignment[b]
                loads[ka] += weights[b] - weights[a]
                loads[kb] += weights[a] - weights[b]
                assignment[a], assignment[b] = kb, ka
            else:
                heavy, target = best_change[1], best_change[2]
                jobs_out, jobs_in = best_change[3], best_change[4]
                for j in jobs_out:
                    assignment[j] = target
                for j in jobs_in:
                    assignment[j] = heavy
                loads[heavy] = best_change[5]
                loads[target] = best_change[6]

    return assignment


def _dispatch_server_order(releases, p2, downstream, mode):
    """Create a work-conserving setup sequence with a selectable priority."""
    n = len(releases)
    future = sorted(range(n), key=lambda j: (releases[j], j))
    available = []
    order = []
    pos = 0
    current = 0.0
    while pos < n or available:
        if not available and pos < n and releases[future[pos]] > current:
            current = releases[future[pos]]
        while pos < n and releases[future[pos]] <= current:
            j = future[pos]
            if mode == 0:
                priority = -downstream[j]
            elif mode == 1:
                priority = p2[j]
            elif mode == 2:
                priority = -(downstream[j] / max(p2[j], 1e-12))
            else:
                priority = j
            heapq.heappush(available, (priority, j))
            pos += 1
        priority, j = heapq.heappop(available)
        order.append(j)
        current += p2[j]
    return order


def _seed_states(n, m, p1, p2, p3, rng, deadline):
    weights = [p1[j] + p3[j] for j in range(n)]
    descending = sorted(range(n), key=lambda j: (-weights[j], j))
    orders = [descending, list(range(n))]
    if n > 1:
        orders.append(sorted(range(n), key=lambda j: (-p3[j], -p1[j], j)))

    assignments = []
    for order in orders:
        assignment = _lpt_assignment(weights, m, order)
        if time.time() < deadline:
            assignment = _improve_load_balance(
                assignment, weights, m, deadline
            )
        assignments.append(assignment)

    # Alternating assignment is valuable because it removes many precedence
    # chains; include it even when its load bound is slightly worse.
    assignments.append([j % m for j in range(n)])

    states = []
    for assignment_id, assignment in enumerate(assignments):
        if time.time() >= deadline and states:
            break
        rank_modes = [
            [float(j) for j in range(n)],
            [-p3[j] for j in range(n)],
            [p1[j] for j in range(n)],
        ]
        init_rank = rank_modes[assignment_id % len(rank_modes)][:]
        main_rank = [-p3[j] for j in range(n)]
        s1, init_end, _ = _initialization_schedule(
            assignment, init_rank, p1, m
        )
        releases = [s1[j] + p1[j] for j in range(n)]
        for mode in range(3):
            if time.time() >= deadline and states:
                break
            server_order = _dispatch_server_order(releases, p2, p3, mode)
            states.append(
                (assignment[:], init_rank[:], server_order, main_rank[:])
            )
    return states


def _mutate(state, m, rng, p1, p3, large=False):
    assignment, init_rank, server_order, main_rank = state
    n = len(assignment)
    new_assignment = assignment[:]
    new_init = init_rank[:]
    new_server = server_order[:]
    new_main = main_rank[:]
    changes = 1
    if large:
        changes = max(2, min(12, n // 5 + 1))

    for change_id in range(changes):
        choice = rng.random()
        if m > 1 and choice < 0.34:
            if rng.random() < 0.48:
                j = rng.randrange(n)
                target = rng.randrange(m - 1)
                if target >= new_assignment[j]:
                    target += 1
                new_assignment[j] = target
            else:
                a = rng.randrange(n)
                b = rng.randrange(n)
                attempts = 0
                while new_assignment[a] == new_assignment[b] and attempts < 8:
                    b = rng.randrange(n)
                    attempts += 1
                new_assignment[a], new_assignment[b] = (
                    new_assignment[b], new_assignment[a]
                )
        elif choice < 0.64:
            a = rng.randrange(n)
            b = rng.randrange(n)
            if a != b:
                job = new_server.pop(a)
                new_server.insert(b, job)
        elif choice < 0.78:
            a = rng.randrange(n)
            b = rng.randrange(n)
            new_init[a], new_init[b] = new_init[b], new_init[a]
        elif choice < 0.93:
            a = rng.randrange(n)
            b = rng.randrange(n)
            new_main[a], new_main[b] = new_main[b], new_main[a]
        else:
            # Coordinated urgency mutation: advance a long main job on the
            # server and in both primary-machine priority systems.
            sample_size = min(n, 5)
            sample = [rng.randrange(n) for unused in range(sample_size)]
            j = max(sample, key=lambda job: p3[job])
            new_init[j] = min(new_init) - 1.0
            new_main[j] = min(new_main) - 1.0
            pos = new_server.index(j)
            target = rng.randrange(pos + 1)
            job = new_server.pop(pos)
            new_server.insert(target, job)

    return new_assignment, new_init, new_server, new_main


def solve(instance: dict, time_limit_s: float) -> dict:
    """Return the best complete schedule found within 85% of the wall budget."""
    t0 = time.time()

    # Exam inputs are schema-valid.  These guarded reads keep the function from
    # ever returning None if an unexpected value reaches it.
    try:
        n = int(instance["num_jobs"])
        m = int(instance["num_machines"])
        p1 = [float(value) for value in instance["init_processing_times"]]
        p2 = [float(value) for value in instance["setup_processing_times"]]
        p3 = [float(value) for value in instance["main_processing_times"]]
        if m <= 0 or n < 0 or len(p1) != n or len(p2) != n or len(p3) != n:
            return {
                "machine_assignment": {},
                "server_sequence": [],
                "start_times": {},
                "makespan": 0.0,
            }
    except Exception:
        return {
            "machine_assignment": {},
            "server_sequence": [],
            "start_times": {},
            "makespan": 0.0,
        }

    best_solution = _fallback_solution(n, m, p1, p2, p3)
    if n <= 1:
        return best_solution

    try:
        budget = max(0.0, float(time_limit_s))
    except Exception:
        budget = 0.0
    deadline = t0 + budget * 0.85
    if time.time() >= deadline:
        return best_solution

    # A stable instance-derived seed makes comparisons reproducible while still
    # giving unrelated instances different search trajectories.
    seed = n * 1000003 + m * 9176
    for j in range(n):
        seed = (
            seed * 1000003
            + int((p1[j] * 17.0 + p2[j] * 31.0 + p3[j] * 47.0) * 1000.0)
        ) & 0xFFFFFFFF
    rng = random.Random(seed)

    try:
        # Reserve only a small fraction for deterministic seed construction; the
        # wall-clock loop below owns the rest of the budget.
        seed_deadline = min(deadline, t0 + budget * 0.12)
        states = _seed_states(n, m, p1, p2, p3, rng, seed_deadline)
        if not states:
            return best_solution

        best_state = states[0]
        best_score = float("inf")
        current_state = states[0]
        current_score = float("inf")

        for state in states:
            if time.time() >= deadline and best_score < float("inf"):
                break
            score, s1, s2, s3 = _evaluate(
                state[0], state[1], state[2], state[3], p1, p2, p3, m
            )
            if score < best_score:
                best_score = score
                best_state = (state[0][:], state[1][:], state[2][:], state[3][:])
                best_solution = _format_solution(
                    state[0], state[2], s1, s2, s3, score
                )
            if score < current_score:
                current_score = score
                current_state = state

        stagnation = 0
        restart_after = max(120, n * 8)

        # This is the only optimization termination condition: wall-clock time.
        while time.time() < deadline:
            elapsed_fraction = (time.time() - t0) / max(budget * 0.85, 1e-9)
            if elapsed_fraction > 1.0:
                elapsed_fraction = 1.0

            large = stagnation >= restart_after
            if large:
                current_state = (
                    best_state[0][:],
                    best_state[1][:],
                    best_state[2][:],
                    best_state[3][:],
                )
                current_score = best_score
                stagnation = 0

            candidate = _mutate(current_state, m, rng, p1, p3, large)
            score, s1, s2, s3 = _evaluate(
                candidate[0], candidate[1], candidate[2], candidate[3],
                p1, p2, p3, m
            )

            scale = max(0.02, best_score * 0.012)
            temperature = scale * (1.0 - elapsed_fraction) + 0.002
            delta = score - current_score
            accept = delta <= 0.0
            if not accept and temperature > 0.0:
                accept = rng.random() < math.exp(-delta / temperature)
            if accept:
                current_state = candidate
                current_score = score

            if score + 1e-10 < best_score:
                best_score = score
                best_state = (
                    candidate[0][:], candidate[1][:],
                    candidate[2][:], candidate[3][:]
                )
                best_solution = _format_solution(
                    candidate[0], candidate[2], s1, s2, s3, score
                )
                stagnation = 0
            else:
                stagnation += 1

    except Exception:
        # best_solution was made before entering the optimization and is always
        # structurally complete and feasible for a valid exam instance.
        return best_solution

    return best_solution
