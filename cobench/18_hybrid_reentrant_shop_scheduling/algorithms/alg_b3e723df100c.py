# Library slot 2/10  id=alg_b3e723df100c
# score=309.1140620881771  feasible_rate=1.0
# score_vec=[334.6, 808.531636889778, 160.52865801046764, 161.1782184662499, 205.3613578857649, 197.65574962951038, 223.40000000000003, 343.70899999999995, 326.51399999999995, 329.66200000000003]

import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    deadline = t0 + max(0.0, float(time_limit_s)) * 0.85

    n = int(instance["num_jobs"])
    m_count = int(instance["num_machines"])
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

    # Install a complete feasible incumbent before doing any search.
    best_assignment = [j % m_count for j in range(n)]
    best_sequence = list(range(n))
    best_value = math.inf
    weights = [p1[j] + p3[j] for j in range(n)]
    tolerance = 1e-10

    def assignment_data(assignment):
        init_load = [0.0] * m_count
        main_load = [0.0] * m_count
        release = [0.0] * n
        tail = [0.0] * n

        for j in range(n):
            machine = assignment[j]
            init_load[machine] += p1[j]
            release[j] = init_load[machine]
            main_load[machine] += p3[j]

        remaining = [0.0] * m_count
        j = n - 1
        while j >= 0:
            machine = assignment[j]
            remaining[machine] += p3[j]
            tail[j] = remaining[machine]
            j -= 1

        base = 0.0
        for machine in range(m_count):
            load = init_load[machine] + main_load[machine]
            if load > base:
                base = load
        return release, tail, base

    def sequence_value(sequence, release, tail, base, cutoff=math.inf):
        server_time = 0.0
        value = base
        for job in sequence:
            ready = release[job]
            if server_time < ready:
                server_time = ready
            server_time += p2[job]
            completion = server_time + tail[job]
            if completion > value:
                value = completion
                if value >= cutoff - tolerance:
                    return value
        return value

    def schrage_sequence(release, tail, tie_mode=0):
        order = sorted(range(n), key=lambda job: (release[job], job))
        available = []
        sequence = []
        index = 0
        server_time = 0.0

        while len(sequence) < n:
            if not available and index < n and server_time < release[order[index]]:
                server_time = release[order[index]]
            while index < n and release[order[index]] <= server_time + tolerance:
                job = order[index]
                if tie_mode == 1:
                    tie_key = -p2[job]
                elif tie_mode == 2:
                    tie_key = p2[job]
                else:
                    tie_key = job
                heapq.heappush(available, (-tail[job], tie_key, job))
                index += 1
            if available:
                job = heapq.heappop(available)[2]
                sequence.append(job)
                server_time += p2[job]
        return sequence

    def value_of(assignment, sequence):
        release, tail, base = assignment_data(assignment)
        return sequence_value(sequence, release, tail, base)

    best_value = value_of(best_assignment, best_sequence)

    def consider(assignment, sequence, known_value=None):
        nonlocal best_assignment, best_sequence, best_value
        if known_value is None:
            known_value = value_of(assignment, sequence)
        if known_value < best_value - tolerance:
            best_value = known_value
            best_assignment = list(assignment)
            best_sequence = list(sequence)
        return known_value

    def make_lpt_assignment(mode, randomized=False):
        loads = [0.0] * m_count
        assignment = [0] * n
        jobs = list(range(n))
        if mode == 1:
            key_values = p3
        elif mode == 2:
            key_values = p1
        else:
            key_values = weights
        if randomized:
            jobs.sort(key=lambda job: key_values[job] * (0.88 + 0.24 * random.random()), reverse=True)
        else:
            jobs.sort(key=lambda job: (key_values[job], p3[job], p1[job], -job), reverse=True)
        for job in jobs:
            min_load = min(loads)
            choices = [machine for machine in range(m_count) if loads[machine] <= min_load + tolerance]
            machine = random.choice(choices) if randomized and len(choices) > 1 else choices[0]
            assignment[job] = machine
            loads[machine] += weights[job]
        return assignment

    def balance_assignment(assignment, until):
        """Descent on the unavoidable per-machine p1+p3 load bound."""
        assignment = list(assignment)
        loads = [0.0] * m_count
        for job in range(n):
            loads[assignment[job]] += weights[job]

        while time.time() < until:
            current = max(loads)
            best_load = current
            best_move = None

            for job in range(n):
                if time.time() >= until:
                    return assignment
                old_machine = assignment[job]
                for new_machine in range(m_count):
                    if new_machine == old_machine:
                        continue
                    old_value = loads[old_machine] - weights[job]
                    new_value = loads[new_machine] + weights[job]
                    trial = 0.0
                    for machine in range(m_count):
                        value = loads[machine]
                        if machine == old_machine:
                            value = old_value
                        elif machine == new_machine:
                            value = new_value
                        if value > trial:
                            trial = value
                    if trial < best_load - tolerance:
                        best_load = trial
                        best_move = (job, old_machine, new_machine)

            if best_move is not None:
                job, old_machine, new_machine = best_move
                assignment[job] = new_machine
                loads[old_machine] -= weights[job]
                loads[new_machine] += weights[job]
                continue

            heavy = [machine for machine in range(m_count) if loads[machine] >= current - tolerance]
            best_swap = None
            for first in range(n):
                if time.time() >= until:
                    return assignment
                first_machine = assignment[first]
                if first_machine not in heavy:
                    continue
                for second in range(n):
                    second_machine = assignment[second]
                    if second_machine == first_machine:
                        continue
                    first_new = loads[first_machine] - weights[first] + weights[second]
                    second_new = loads[second_machine] - weights[second] + weights[first]
                    trial = 0.0
                    for machine in range(m_count):
                        value = loads[machine]
                        if machine == first_machine:
                            value = first_new
                        elif machine == second_machine:
                            value = second_new
                        if value > trial:
                            trial = value
                    if trial < best_load - tolerance:
                        best_load = trial
                        best_swap = (first, second, first_machine, second_machine)

            if best_swap is None:
                break
            first, second, first_machine, second_machine = best_swap
            assignment[first], assignment[second] = second_machine, first_machine
            loads[first_machine] += weights[second] - weights[first]
            loads[second_machine] += weights[first] - weights[second]
        return assignment

    def initial_sequences(assignment):
        release, tail, base = assignment_data(assignment)
        candidates = [
            schrage_sequence(release, tail, 0),
            schrage_sequence(release, tail, 1),
            schrage_sequence(release, tail, 2),
            sorted(range(n), key=lambda job: (release[job], -tail[job], p2[job])),
            sorted(range(n), key=lambda job: (-tail[job], release[job], p2[job])),
            sorted(range(n), key=lambda job: (p2[job], release[job], -tail[job])),
        ]
        chosen = candidates[0]
        chosen_value = sequence_value(chosen, release, tail, base)
        for candidate in candidates[1:]:
            value = sequence_value(candidate, release, tail, base, chosen_value)
            if value < chosen_value - tolerance:
                chosen = candidate
                chosen_value = value
        return chosen, chosen_value

    def sequence_descent(assignment, sequence, until):
        release, tail, base = assignment_data(assignment)
        sequence = list(sequence)
        current = sequence_value(sequence, release, tail, base)
        schrage = schrage_sequence(release, tail)
        schrage_value = sequence_value(schrage, release, tail, base, current)
        if schrage_value < current - tolerance:
            sequence, current = schrage, schrage_value

        while time.time() < until and current > base + tolerance:
            improved = False
            best_candidate = None
            best_candidate_value = current

            old_position = 0
            while old_position < n and time.time() < until:
                job = sequence[old_position]
                shortened = sequence[:old_position] + sequence[old_position + 1:]
                new_position = 0
                while new_position < n and time.time() < until:
                    if new_position != old_position:
                        candidate = shortened[:new_position] + [job] + shortened[new_position:]
                        value = sequence_value(candidate, release, tail, base, best_candidate_value)
                        if value < best_candidate_value - tolerance:
                            best_candidate_value = value
                            best_candidate = candidate
                    new_position += 1
                old_position += 1

            if best_candidate is not None:
                sequence, current = best_candidate, best_candidate_value
                improved = True

            if time.time() >= until or current <= base + tolerance:
                break

            if not improved:
                first = 0
                while first < n - 1 and time.time() < until:
                    second = first + 1
                    while second < n and time.time() < until:
                        candidate = list(sequence)
                        candidate[first], candidate[second] = candidate[second], candidate[first]
                        value = sequence_value(candidate, release, tail, base, best_candidate_value)
                        if value < best_candidate_value - tolerance:
                            best_candidate_value = value
                            best_candidate = candidate
                        second += 1
                    first += 1
                if best_candidate is not None:
                    sequence, current = best_candidate, best_candidate_value
                    improved = True

            if not improved:
                break
        return sequence, current

    def assignment_descent(assignment, sequence, until):
        assignment = list(assignment)
        sequence, current = sequence_descent(assignment, sequence, until)
        consider(assignment, sequence, current)

        while time.time() < until:
            move_assignment = None
            move_sequence = None
            move_value = current

            job = 0
            while job < n and time.time() < until:
                old_machine = assignment[job]
                new_machine = 0
                while new_machine < m_count and time.time() < until:
                    if new_machine != old_machine:
                        candidate_assignment = list(assignment)
                        candidate_assignment[job] = new_machine
                        release, tail, base = assignment_data(candidate_assignment)
                        candidate_sequence = schrage_sequence(release, tail)
                        value = sequence_value(candidate_sequence, release, tail, base, move_value)
                        old_sequence_value = sequence_value(sequence, release, tail, base, move_value)
                        if old_sequence_value < value:
                            candidate_sequence = sequence
                            value = old_sequence_value
                        if value < move_value - tolerance:
                            move_assignment = candidate_assignment
                            move_sequence = list(candidate_sequence)
                            move_value = value
                    new_machine += 1
                job += 1

            if move_assignment is not None:
                assignment = move_assignment
                sequence, current = sequence_descent(assignment, move_sequence, until)
                consider(assignment, sequence, current)
                continue

            # A cross-machine exchange escapes the common one-move load-balance trap.
            first = 0
            while first < n - 1 and time.time() < until:
                second = first + 1
                while second < n and time.time() < until:
                    if assignment[first] != assignment[second]:
                        candidate_assignment = list(assignment)
                        candidate_assignment[first], candidate_assignment[second] = (
                            candidate_assignment[second], candidate_assignment[first]
                        )
                        release, tail, base = assignment_data(candidate_assignment)
                        candidate_sequence = schrage_sequence(release, tail)
                        value = sequence_value(candidate_sequence, release, tail, base, move_value)
                        if value < move_value - tolerance:
                            move_assignment = candidate_assignment
                            move_sequence = candidate_sequence
                            move_value = value
                    second += 1
                first += 1

            if move_assignment is None:
                break
            assignment = move_assignment
            sequence, current = sequence_descent(assignment, move_sequence, until)
            consider(assignment, sequence, current)
        return assignment, sequence, current

    # Deterministic constructions give useful diversity at negligible risk.
    constructions = [
        list(best_assignment),
        make_lpt_assignment(0),
        make_lpt_assignment(1),
        make_lpt_assignment(2),
    ]
    sequential_loads = [0.0] * m_count
    sequential_assignment = [0] * n
    for job in range(n):
        machine = min(range(m_count), key=lambda item: (sequential_loads[item], item))
        sequential_assignment[job] = machine
        sequential_loads[machine] += weights[job]
    constructions.append(sequential_assignment)

    for construction in constructions:
        if time.time() >= deadline:
            break
        construction = balance_assignment(construction, deadline)
        sequence, value = initial_sequences(construction)
        consider(construction, sequence, value)

    # Seeded randomness makes repeated grading stable while retaining GRASP diversity.
    seed_value = n * 1000003 + m_count * 9176
    for job in range(n):
        seed_value += int((p1[job] * 17.0 + p2[job] * 31.0 + p3[job] * 43.0) * 1000.0)
    random.seed(seed_value)

    # Intensify the best construction once before entering the restart loop.
    if time.time() < deadline:
        assignment_descent(best_assignment, best_sequence, deadline)

    stagnation = 0
    while time.time() < deadline:
        previous_best = best_value
        if stagnation % 3 == 0:
            assignment = make_lpt_assignment(0, randomized=True)
        else:
            assignment = list(best_assignment)
            changes = 1 + min(max(1, n // 8), 1 + stagnation // 2)
            changed = 0
            while changed < changes and time.time() < deadline:
                job = random.randrange(n)
                assignment[job] = random.randrange(m_count)
                changed += 1

        assignment = balance_assignment(assignment, deadline)
        if time.time() >= deadline:
            break
        sequence, value = initial_sequences(assignment)
        consider(assignment, sequence, value)
        assignment_descent(assignment, sequence, deadline)

        if best_value < previous_best - tolerance:
            stagnation = 0
        else:
            stagnation += 1

    def build_result(assignment, sequence):
        s1 = [0.0] * n
        s2 = [0.0] * n
        s3 = [0.0] * n
        init_free = [0.0] * m_count
        init_done = [0.0] * m_count

        for job in range(n):
            machine = assignment[job]
            s1[job] = init_free[machine]
            init_free[machine] += p1[job]
            init_done[machine] = init_free[machine]

        server_free = 0.0
        for job in sequence:
            server_free = max(server_free, s1[job] + p1[job])
            s2[job] = server_free
            server_free += p2[job]

        main_free = [0.0] * m_count
        makespan = 0.0
        for job in range(n):
            machine = assignment[job]
            s3[job] = max(s2[job] + p2[job], init_done[machine], main_free[machine])
            main_free[machine] = s3[job] + p3[job]
            if main_free[machine] > makespan:
                makespan = main_free[machine]

        return {
            "machine_assignment": {f"job_{job}": int(assignment[job]) for job in range(n)},
            "server_sequence": [f"job_{job}" for job in sequence],
            "start_times": {f"job_{job}": [float(s1[job]), float(s2[job]), float(s3[job])] for job in range(n)},
            "makespan": float(makespan),
        }

    return build_result(best_assignment, best_sequence)
