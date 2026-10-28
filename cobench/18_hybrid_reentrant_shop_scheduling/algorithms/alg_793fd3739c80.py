# Library slot 7/10  id=alg_793fd3739c80
# score=311.12912763033506  feasible_rate=1.0
# score_vec=[334.7, 808.5322200120543, 163.6783129017911, 164.59892638151945, 207.62084218919512, 199.69297481879084, 223.40000000000003, 345.858, 327.431, 335.7789999999999]

"""Anytime heuristic for the hybrid reentrant shop scheduling challenge."""

import heapq
import itertools
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    """Return a feasible schedule, improving it until the safety deadline."""
    t0 = time.time()
    budget = max(0.0, float(time_limit_s))
    deadline = t0 + budget * 0.85

    n = int(instance["num_jobs"])
    m = int(instance["num_machines"])
    p1 = [float(x) for x in instance["init_processing_times"]]
    p2 = [float(x) for x in instance["setup_processing_times"]]
    p3 = [float(x) for x in instance["main_processing_times"]]

    if n == 0:
        return {
            "machine_assignment": {},
            "server_sequence": [],
            "start_times": {},
            "makespan": 0.0,
        }

    jobs = list(range(n))

    def repaired_init_orders(assignment, priority):
        """Return each machine's jobs in the required natural order."""
        del priority
        orders = [[] for _ in range(m)]
        for job in jobs:
            orders[assignment[job]].append(job)
        return orders

    def evaluate(assignment, init_priority, server_sequence, materialize=False):
        init_orders = repaired_init_orders(assignment, init_priority)
        s1 = [0.0] * n
        release = [0.0] * n
        machine_init_end = [0.0] * m

        for machine in range(m):
            clock = 0.0
            for job in init_orders[machine]:
                s1[job] = clock
                clock += p1[job]
                release[job] = clock
            machine_init_end[machine] = clock

        s2 = [0.0] * n
        setup_end = [0.0] * n
        server_clock = 0.0
        for job in server_sequence:
            start = server_clock if server_clock >= release[job] else release[job]
            s2[job] = start
            server_clock = start + p2[job]
            setup_end[job] = server_clock

        s3 = [0.0] * n
        machine_main_end = machine_init_end[:]
        score = 0.0
        for job in jobs:
            machine = assignment[job]
            start = setup_end[job]
            if machine_main_end[machine] > start:
                start = machine_main_end[machine]
            s3[job] = start
            completion = start + p3[job]
            machine_main_end[machine] = completion
            if completion > score:
                score = completion

        if not materialize:
            return score

        return {
            "machine_assignment": {
                "job_{}".format(job): int(assignment[job]) for job in jobs
            },
            "server_sequence": ["job_{}".format(job) for job in server_sequence],
            "start_times": {
                "job_{}".format(job): [s1[job], s2[job], s3[job]]
                for job in jobs
            },
            "makespan": float(score),
        }

    def greedy_assignment(mode, offset=0):
        assignment = [0] * n
        loads = [0.0] * m
        largest_tail = [0.0] * m
        for job in jobs:
            choices = list(range(m))
            if m > 1 and job > 0:
                choices.remove(assignment[job - 1])
            if mode == 0:
                machine = min(
                    choices,
                    key=lambda k: (loads[k] + p1[job], loads[k], k),
                )
            elif mode == 1:
                machine = min(
                    choices,
                    key=lambda k: (
                        loads[k] + p1[job] + max(largest_tail[k], p3[job]),
                        loads[k],
                        k,
                    ),
                )
            else:
                machine = (job + offset) % m
                if job > 0 and machine == assignment[job - 1] and m > 1:
                    machine = (machine + 1) % m
            assignment[job] = machine
            loads[machine] += p1[job]
            if p3[job] > largest_tail[machine]:
                largest_tail[machine] = p3[job]
        return assignment

    def lpt_assignment(include_tail):
        """Balance machine loads while properly coloring the job-index path."""
        assignment = [-1] * n
        loads = [0.0] * m
        largest_tail = [0.0] * m
        order = sorted(
            jobs,
            key=lambda j: (-(p1[j] + (0.2 * p3[j] if include_tail else 0.0)), j),
        )
        for job in order:
            forbidden = set()
            if job > 0 and assignment[job - 1] >= 0:
                forbidden.add(assignment[job - 1])
            if job + 1 < n and assignment[job + 1] >= 0:
                forbidden.add(assignment[job + 1])
            choices = [machine for machine in range(m) if machine not in forbidden]
            machine = min(
                choices,
                key=lambda k: (
                    loads[k] + p1[job]
                    + (max(largest_tail[k], p3[job]) if include_tail else 0.0),
                    loads[k],
                    k,
                ),
            )
            assignment[job] = machine
            loads[machine] += p1[job]
            largest_tail[machine] = max(largest_tail[machine], p3[job])
        return assignment

    def dispatch_server(assignment, init_priority, tail_weight=1.0):
        init_orders = repaired_init_orders(assignment, init_priority)
        release = [0.0] * n
        for order in init_orders:
            clock = 0.0
            for job in order:
                clock += p1[job]
                release[job] = clock

        unreleased = sorted(jobs, key=lambda j: (release[j], j))
        available = []
        sequence = []
        clock = 0.0
        pos = 0
        while len(sequence) < n:
            if not available and pos < n and clock < release[unreleased[pos]]:
                clock = release[unreleased[pos]]
            while pos < n and release[unreleased[pos]] <= clock + 1e-12:
                job = unreleased[pos]
                priority = -(tail_weight * p3[job] + p2[job])
                heapq.heappush(available, (priority, release[job], job))
                pos += 1
            if available:
                _, _, job = heapq.heappop(available)
                sequence.append(job)
                clock += p2[job]
        return sequence

    def guided_swap(assignment):
        """Swap unlike-sized jobs to reduce load-plus-tail machine floors."""
        loads = [0.0] * m
        tails = [0.0] * m
        second_tails = [0.0] * m
        tail_counts = [0] * m
        by_machine = [[] for _ in range(m)]
        for job in jobs:
            machine = assignment[job]
            loads[machine] += p1[job]
            if p3[job] > tails[machine]:
                second_tails[machine] = tails[machine]
                tails[machine] = p3[job]
                tail_counts[machine] = 1
            elif p3[job] == tails[machine]:
                tail_counts[machine] += 1
            elif p3[job] > second_tails[machine]:
                second_tails[machine] = p3[job]
            by_machine[machine].append(job)

        source = max(range(m), key=lambda k: loads[k] + tails[k])
        unchanged_floors = [loads[k] + tails[k] for k in range(m)]
        best_floor = max(unchanged_floors)
        best_pair = None

        for first in by_machine[source]:
            if time.time() >= deadline:
                return assignment[:]
            for target in range(m):
                if target == source:
                    continue
                for second in by_machine[target]:
                    def color(index):
                        if index == first:
                            return target
                        if index == second:
                            return source
                        return assignment[index]

                    valid = True
                    affected_edges = (first - 1, first, second - 1, second)
                    for edge in affected_edges:
                        if edge < 0 or edge + 1 >= n:
                            continue
                        if color(edge) == color(edge + 1):
                            valid = False
                            break
                    if not valid:
                        continue

                    source_load = loads[source] - p1[first] + p1[second]
                    target_load = loads[target] - p1[second] + p1[first]
                    old_source_tail = tails[source]
                    if p3[first] == tails[source] and tail_counts[source] == 1:
                        old_source_tail = second_tails[source]
                    old_target_tail = tails[target]
                    if p3[second] == tails[target] and tail_counts[target] == 1:
                        old_target_tail = second_tails[target]
                    source_tail = max(old_source_tail, p3[second])
                    target_tail = max(old_target_tail, p3[first])
                    floor = max(source_load + source_tail, target_load + target_tail)
                    for machine in range(m):
                        if machine not in (source, target):
                            floor = max(floor, unchanged_floors[machine])
                    if floor < best_floor - 1e-12:
                        best_floor = floor
                        best_pair = (first, second)

        result = assignment[:]
        if best_pair is not None:
            first, second = best_pair
            result[first], result[second] = result[second], result[first]
        return result

    def assignment_floor(assignment):
        loads = [0.0] * m
        tails = [0.0] * m
        for job in jobs:
            machine = assignment[job]
            loads[machine] += p1[job]
            tails[machine] = max(tails[machine], p3[job])
        return max(loads[k] + tails[k] for k in range(m))

    init_variants = [
        sorted(jobs, key=lambda j: (-p3[j], p1[j], j)),
        sorted(jobs, key=lambda j: (p1[j], -p3[j], j)),
        sorted(jobs, key=lambda j: (-(p3[j] / max(p1[j], 1e-12)), j)),
        jobs[:],
    ]

    assignments = []
    seen_assignments = set()
    raw_assignments = [greedy_assignment(0), greedy_assignment(1)]
    if m >= 3:
        raw_assignments.extend((lpt_assignment(False), lpt_assignment(True)))
    for offset in range(min(m, 12)):
        raw_assignments.append(greedy_assignment(2, offset))
    for assignment in raw_assignments:
        key = tuple(assignment)
        if key not in seen_assignments:
            seen_assignments.add(key)
            assignments.append(assignment)

    best_score = math.inf
    best_assignment = assignments[0][:]
    best_init = init_variants[0][:]
    best_server = jobs[:]

    for assignment in assignments:
        for init_priority in init_variants:
            for weight in (0.0, 0.5, 1.0, 2.0):
                server_sequence = dispatch_server(
                    assignment, init_priority, weight
                )
                score = evaluate(assignment, init_priority, server_sequence)
                if score < best_score:
                    best_score = score
                    best_assignment = assignment[:]
                    best_init = init_priority[:]
                    best_server = server_sequence[:]

    # Randomized forward/reverse path colorings escape load-balancing minima.
    if m >= 3 and time.time() < deadline:
        construct_seed = n * 1009 + m * 9176
        for value in p1 + p3:
            construct_seed ^= int(abs(value) * 1000003.0) & 0xFFFFFFFFFFFFFFFF
            construct_seed = (construct_seed * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        construct_rng = random.Random(construct_seed ^ 0x9E3779B97F4A7C15)
        coloring_deadline = min(deadline, t0 + budget * 0.08)
        proxy_best = assignment_floor(best_assignment)

        while time.time() < coloring_deadline:
            reverse = construct_rng.random() < 0.5
            order = range(n - 1, -1, -1) if reverse else range(n)
            assignment = [-1] * n
            loads = [0.0] * m
            tails = [0.0] * m
            for job in order:
                neighbor = job + 1 if reverse else job - 1
                forbidden = assignment[neighbor] if 0 <= neighbor < n else -1
                choices = [k for k in range(m) if k != forbidden]
                choices.sort(
                    key=lambda k: (
                        loads[k] + p1[job] + max(tails[k], p3[job]),
                        loads[k],
                        k,
                    )
                )
                draw = construct_rng.random()
                if draw < 0.55:
                    rank = 0
                elif draw < 0.80:
                    rank = min(1, len(choices) - 1)
                else:
                    rank = construct_rng.randrange(min(4, len(choices)))
                machine = choices[rank]
                assignment[job] = machine
                loads[machine] += p1[job]
                tails[machine] = max(tails[machine], p3[job])

            floor = max(loads[k] + tails[k] for k in range(m))
            if floor < proxy_best - 1e-10:
                proxy_best = floor
                for init_priority in init_variants[:3]:
                    for weight in (0.5, 1.0, 2.0):
                        server_sequence = dispatch_server(
                            assignment, init_priority, weight
                        )
                        score = evaluate(
                            assignment, init_priority, server_sequence
                        )
                        if score < best_score:
                            best_score = score
                            best_assignment = assignment[:]
                            best_init = init_priority[:]
                            best_server = server_sequence[:]

    # Tiny instances can be exhaustively polished without risking the incumbent.
    if n <= 5 and m <= 3 and time.time() < deadline:
        permutations = list(itertools.permutations(jobs))
        checked = 0
        stop_exact = False
        for assignment_tuple in itertools.product(range(m), repeat=n):
            assignment = list(assignment_tuple)
            for init_tuple in permutations:
                init_priority = list(init_tuple)
                for server_tuple in permutations:
                    score = evaluate(assignment, init_priority, server_tuple)
                    if score < best_score:
                        best_score = score
                        best_assignment = assignment[:]
                        best_init = init_priority[:]
                        best_server = list(server_tuple)
                    checked += 1
                    if checked % 256 == 0 and time.time() >= deadline:
                        stop_exact = True
                        break
                if stop_exact:
                    break
            if stop_exact:
                break

    seed = 1469598103934665603
    for value in p1 + p2 + p3:
        seed ^= int(abs(value) * 1000003.0) & 0xFFFFFFFFFFFFFFFF
        seed = (seed * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    seed ^= n * 1009 + m * 9176
    rng = random.Random(seed)

    current_assignment = best_assignment[:]
    current_init = best_init[:]
    current_server = best_server[:]
    current_score = best_score
    accepted_since_restart = 0
    restart_at = t0 + max(0.002, budget * 0.04)

    while time.time() < deadline:
        now = time.time()
        if now >= restart_at:
            current_assignment = best_assignment[:]
            current_init = best_init[:]
            current_server = best_server[:]
            current_score = best_score
            strength = 1 + rng.randrange(max(1, min(8, n)))
            for _ in range(strength):
                move_kind = rng.randrange(3)
                if n > 1 and move_kind == 0:
                    i, j = rng.sample(range(n), 2)
                    current_server[i], current_server[j] = (
                        current_server[j], current_server[i]
                    )
                elif n > 1 and move_kind == 1:
                    i, j = rng.sample(range(n), 2)
                    current_init[i], current_init[j] = current_init[j], current_init[i]
                elif m > 1:
                    job = rng.randrange(n)
                    machine = rng.randrange(m - 1)
                    if machine >= current_assignment[job]:
                        machine += 1
                    current_assignment[job] = machine
            current_score = evaluate(
                current_assignment, current_init, current_server
            )
            accepted_since_restart = 0
            restart_at = now + max(0.002, budget * (0.018 + 0.025 * rng.random()))

        candidate_assignment = current_assignment[:]
        candidate_init = current_init[:]
        candidate_server = current_server[:]
        move = rng.random()

        if move < 0.43 and n > 1:
            i, j = rng.sample(range(n), 2)
            if rng.random() < 0.68:
                candidate_server[i], candidate_server[j] = (
                    candidate_server[j], candidate_server[i]
                )
            else:
                job = candidate_server.pop(i)
                candidate_server.insert(j, job)
        elif move < 0.76 and n > 1:
            i, j = rng.sample(range(n), 2)
            if rng.random() < 0.68:
                candidate_init[i], candidate_init[j] = (
                    candidate_init[j], candidate_init[i]
                )
            else:
                job = candidate_init.pop(i)
                candidate_init.insert(j, job)
        elif m > 1:
            assignment_move = rng.random()
            if m >= 3 and assignment_move < 0.55:
                loads = [0.0] * m
                for assigned_job in jobs:
                    loads[candidate_assignment[assigned_job]] += p1[assigned_job]
                source_order = sorted(range(m), key=lambda k: loads[k], reverse=True)
                source = source_order[rng.randrange(min(3, m))]
                source_jobs = [j for j in jobs if candidate_assignment[j] == source]
                rng.shuffle(source_jobs)
                best_move = None
                best_peak = math.inf
                for job in source_jobs:
                    forbidden = set()
                    if job > 0:
                        forbidden.add(candidate_assignment[job - 1])
                    if job + 1 < n:
                        forbidden.add(candidate_assignment[job + 1])
                    for machine in range(m):
                        if machine == source or machine in forbidden:
                            continue
                        new_source = loads[source] - p1[job]
                        new_target = loads[machine] + p1[job]
                        peak = max(
                            new_source,
                            new_target,
                            max(loads[k] for k in range(m) if k not in (source, machine)),
                        )
                        if peak < best_peak:
                            best_peak = peak
                            best_move = (job, machine)
                if best_move is not None:
                    candidate_assignment[best_move[0]] = best_move[1]
            elif assignment_move < 0.88 and n > 1:
                candidate_assignment = guided_swap(candidate_assignment)
            else:
                job = rng.randrange(n)
                machine = rng.randrange(m - 1)
                old_machine = candidate_assignment[job]
                if machine >= old_machine:
                    machine += 1
                candidate_assignment[job] = machine
        elif n > 1:
            i, j = rng.sample(range(n), 2)
            candidate_server[i], candidate_server[j] = (
                candidate_server[j], candidate_server[i]
            )

        candidate_score = evaluate(
            candidate_assignment, candidate_init, candidate_server
        )
        elapsed_fraction = min(1.0, max(0.0, (now - t0) / max(budget * 0.85, 1e-9)))
        cycle = ((now - t0) / max(budget * 0.035, 1e-9)) % 1.0
        temperature = max(1e-8, best_score * (0.00008 + 0.006 * (1.0 - cycle) ** 2) * (1.0 - 0.75 * elapsed_fraction))
        delta = candidate_score - current_score
        accept = delta <= 0.0
        if not accept and delta < 50.0 * temperature:
            accept = rng.random() < math.exp(-delta / temperature)

        if accept:
            current_assignment = candidate_assignment
            current_init = candidate_init
            current_server = candidate_server
            current_score = candidate_score
            accepted_since_restart += 1
            if candidate_score < best_score - 1e-10:
                best_score = candidate_score
                best_assignment = candidate_assignment[:]
                best_init = candidate_init[:]
                best_server = candidate_server[:]

        if accepted_since_restart > max(50, 12 * n):
            restart_at = now

    return evaluate(
        best_assignment, best_init, best_server, materialize=True
    )
