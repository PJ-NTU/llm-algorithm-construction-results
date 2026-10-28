# Library slot 1/10  id=alg_e4d115590571
# score=309.1124944447624  feasible_rate=1.0
# score_vec=[334.6, 808.5316388797685, 160.51759900955918, 161.1782893507274, 205.35477022762447, 197.65764697994453, 223.40000000000006, 343.70899999999995, 326.5139999999999, 329.66200000000003]

import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()

    n = int(instance["num_jobs"])
    machine_count = int(instance["num_machines"])
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

    # No useful solution uses more than one machine per job.  Restricting the
    # labels also keeps neighborhoods small when an instance has many machines.
    used_machine_count = min(machine_count, n)
    fallback_assignment = [j % used_machine_count for j in range(n)]
    fallback_sequence = list(range(n))

    # For a fixed assignment, let r[j] be the setup release time and q[j] the
    # remaining main-processing tail on j's machine.  The exact makespan is
    # max(max_m(sum(p1+p3)), max_j(C_setup[j] + q[j])).  This turns server
    # sequencing into the classic 1|r_j,q_j|Cmax problem.
    def prepare(assignment):
        init_load = [0.0] * used_machine_count
        main_load = [0.0] * used_machine_count
        release = [0.0] * n
        for j in range(n):
            m = assignment[j]
            init_load[m] += p1[j]
            release[j] = init_load[m]
            main_load[m] += p3[j]

        tail = [0.0] * n
        suffix = [0.0] * used_machine_count
        for j in range(n - 1, -1, -1):
            m = assignment[j]
            suffix[m] += p3[j]
            tail[j] = suffix[m]

        base = 0.0
        for m in range(used_machine_count):
            base = max(base, init_load[m] + main_load[m])
        return release, tail, init_load, base

    def sequence_score(prepared, sequence):
        release, tail, _, base = prepared
        server_time = 0.0
        value = base
        for j in sequence:
            server_time = max(server_time, release[j]) + p2[j]
            value = max(value, server_time + tail[j])
        return value

    def schrage_sequence(prepared):
        release, tail, _, _ = prepared
        by_release = sorted(range(n), key=lambda j: (release[j], j))
        ready = []
        sequence = []
        idx = 0
        server_time = release[by_release[0]]
        while idx < n or ready:
            while idx < n and release[by_release[idx]] <= server_time + 1e-12:
                j = by_release[idx]
                heapq.heappush(ready, (-tail[j], p2[j], j))
                idx += 1
            if not ready:
                server_time = max(server_time, release[by_release[idx]])
                continue
            _, _, j = heapq.heappop(ready)
            sequence.append(j)
            server_time += p2[j]
        return sequence

    def constructed_state(assignment, inherited_sequence=None):
        assignment = list(assignment)
        prepared = prepare(assignment)
        release, tail, _, _ = prepared
        candidates = [
            schrage_sequence(prepared),
            sorted(range(n), key=lambda j: (-tail[j], release[j], p2[j])),
            sorted(range(n), key=lambda j: (release[j], -tail[j], p2[j])),
        ]
        if inherited_sequence is not None:
            candidates.append(list(inherited_sequence))

        best_sequence = candidates[0]
        best_value = sequence_score(prepared, best_sequence)
        for sequence in candidates[1:]:
            value = sequence_score(prepared, sequence)
            if value < best_value - 1e-10:
                best_sequence = sequence
                best_value = value
        return assignment, list(best_sequence), best_value, prepared

    def list_assignment(order, weights):
        heap = [(0.0, m) for m in range(used_machine_count)]
        heapq.heapify(heap)
        assignment = [0] * n
        for j in order:
            load, m = heapq.heappop(heap)
            assignment[j] = m
            heapq.heappush(heap, (load + weights[j], m))
        return assignment

    combined = [p1[j] + p3[j] for j in range(n)]
    seeds = [
        fallback_assignment,
        list_assignment(range(n), combined),
        list_assignment(sorted(range(n), key=lambda j: -combined[j]), combined),
        list_assignment(sorted(range(n), key=lambda j: -p3[j]), p3),
        list_assignment(range(n), p1),
    ]

    # This first state is a complete, immediately returnable feasible fallback.
    best = constructed_state(seeds[0])
    current = best

    budget = max(0.0, float(time_limit_s)) * 0.85
    deadline = t0 + budget
    balance_phase_end = t0 + budget * 0.18
    phase_one_end = t0 + budget * 0.36
    phase_two_end = t0 + budget * 0.50
    seed_value = (
        n * 1000003
        + machine_count * 9176
        + int(sum(p1) * 101.0)
        + int(sum(p2) * 1009.0)
        + int(sum(p3) * 37.0)
    )
    rng = random.Random(seed_value)

    def consider(state):
        nonlocal best
        if state[2] < best[2] - 1e-10:
            best = (list(state[0]), list(state[1]), state[2], state[3])

    def improve_assignment(state, stop_time):
        assignment, sequence, value, prepared = state
        assignment = list(assignment)
        sequence = list(sequence)

        while time.time() < stop_time:
            chosen = None
            chosen_value = value
            jobs = list(range(n))
            rng.shuffle(jobs)

            # Complete one-job relocation neighborhood (best improvement).
            for j in jobs:
                old_m = assignment[j]
                machine_offset = rng.randrange(used_machine_count)
                for step in range(used_machine_count):
                    if time.time() >= stop_time:
                        break
                    new_m = (machine_offset + step) % used_machine_count
                    if new_m == old_m:
                        continue
                    assignment[j] = new_m
                    candidate = constructed_state(assignment, sequence)
                    assignment[j] = old_m
                    if candidate[2] < chosen_value - 1e-10:
                        chosen = candidate
                        chosen_value = candidate[2]
                if time.time() >= stop_time:
                    break

            if chosen is not None:
                assignment, sequence, value, prepared = chosen
                assignment = list(assignment)
                sequence = list(sequence)
                consider(chosen)
                continue

            # Relocation local optima are often only a poor load partition.
            # Cross-machine exchanges preserve job count while moving both p1
            # and p3 load, and are especially valuable on two-machine cases.
            pair_jobs = list(range(n))
            rng.shuffle(pair_jobs)
            for ai in range(n):
                i = pair_jobs[ai]
                for aj in range(ai + 1, n):
                    if time.time() >= stop_time:
                        break
                    j = pair_jobs[aj]
                    if assignment[i] == assignment[j]:
                        continue
                    assignment[i], assignment[j] = assignment[j], assignment[i]
                    candidate = constructed_state(assignment, sequence)
                    assignment[i], assignment[j] = assignment[j], assignment[i]
                    if candidate[2] < value - 1e-10:
                        assignment, sequence, value, prepared = candidate
                        assignment = list(assignment)
                        sequence = list(sequence)
                        consider(candidate)
                        chosen = candidate
                        break
                if chosen is not None or time.time() >= stop_time:
                    break

            if chosen is None:
                break

        return assignment, sequence, value, prepared

    def improve_load_partition(state, stop_time):
        """Fast specialized search for the common primary-machine bottleneck."""
        assignment = list(state[0])
        loads = [0.0] * used_machine_count
        for j in range(n):
            loads[assignment[j]] += combined[j]

        current_peak = max(loads)
        best_peak = current_peak
        best_assignment = list(assignment)
        last_peak_improvement = time.time()

        # First exhaust improving exchanges involving a currently heaviest
        # machine.  Scoring is O(M), instead of rebuilding a full schedule.
        while time.time() < stop_time:
            heavy = max(range(used_machine_count), key=lambda m: loads[m])
            chosen = None
            chosen_peak = current_peak
            for i in range(n):
                if assignment[i] != heavy:
                    continue
                for target in range(used_machine_count):
                    if target == heavy:
                        continue
                    if time.time() >= stop_time:
                        break
                    loads[heavy] -= combined[i]
                    loads[target] += combined[i]
                    candidate_peak = max(loads)
                    loads[heavy] += combined[i]
                    loads[target] -= combined[i]
                    if candidate_peak < chosen_peak - 1e-10:
                        chosen = (i, -1, target)
                        chosen_peak = candidate_peak

                    for j in range(n):
                        if assignment[j] != target:
                            continue
                        loads[heavy] += combined[j] - combined[i]
                        loads[target] += combined[i] - combined[j]
                        candidate_peak = max(loads)
                        loads[heavy] += combined[i] - combined[j]
                        loads[target] += combined[j] - combined[i]
                        if candidate_peak < chosen_peak - 1e-10:
                            chosen = (i, j, target)
                            chosen_peak = candidate_peak
                if time.time() >= stop_time:
                    break

            if chosen is None:
                break
            i, j, target = chosen
            heavy = assignment[i]
            if j < 0:
                loads[heavy] -= combined[i]
                loads[target] += combined[i]
                assignment[i] = target
            else:
                loads[heavy] += combined[j] - combined[i]
                loads[target] += combined[i] - combined[j]
                assignment[i], assignment[j] = assignment[j], assignment[i]
            current_peak = chosen_peak
            if current_peak < best_peak - 1e-10:
                best_peak = current_peak
                best_assignment = list(assignment)
                last_peak_improvement = time.time()

        # Cheap annealing supplies the 2-for-1 and chained exchanges that a
        # relocation/swap local optimum misses.  Its only cap is the wall clock.
        while time.time() < stop_time:
            old_assignment = list(assignment)
            old_loads = list(loads)
            roll = rng.random()
            heavy = max(range(used_machine_count), key=lambda m: loads[m])

            if roll < 0.46:
                heavy_jobs = [j for j in range(n) if assignment[j] == heavy]
                i = heavy_jobs[rng.randrange(len(heavy_jobs))]
                target = rng.randrange(used_machine_count - 1)
                if target >= heavy:
                    target += 1
                target_jobs = [j for j in range(n) if assignment[j] == target]
                if target_jobs:
                    j = target_jobs[rng.randrange(len(target_jobs))]
                    assignment[i], assignment[j] = target, heavy
                    loads[heavy] += combined[j] - combined[i]
                    loads[target] += combined[i] - combined[j]
                else:
                    assignment[i] = target
                    loads[heavy] -= combined[i]
                    loads[target] += combined[i]
            elif roll < 0.76:
                i = rng.randrange(n)
                target = rng.randrange(used_machine_count - 1)
                if target >= assignment[i]:
                    target += 1
                source = assignment[i]
                assignment[i] = target
                loads[source] -= combined[i]
                loads[target] += combined[i]
            else:
                heavy_jobs = [j for j in range(n) if assignment[j] == heavy]
                if len(heavy_jobs) < 2:
                    assignment = old_assignment
                    loads = old_loads
                    continue
                i, j = rng.sample(heavy_jobs, 2)
                target = rng.randrange(used_machine_count - 1)
                if target >= heavy:
                    target += 1
                target_jobs = [k for k in range(n) if assignment[k] == target]
                if not target_jobs:
                    assignment = old_assignment
                    loads = old_loads
                    continue
                k = target_jobs[rng.randrange(len(target_jobs))]
                assignment[i] = target
                assignment[j] = target
                assignment[k] = heavy
                loads[heavy] += combined[k] - combined[i] - combined[j]
                loads[target] += combined[i] + combined[j] - combined[k]

            candidate_peak = max(loads)
            remaining = max(0.0, (stop_time - time.time()) / max(budget, 1e-9))
            temperature = max(1e-8, best_peak * (0.0002 + 0.004 * remaining))
            delta = candidate_peak - current_peak
            if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
                current_peak = candidate_peak
                if current_peak < best_peak - 1e-10:
                    best_peak = current_peak
                    best_assignment = list(assignment)
                    last_peak_improvement = time.time()
            else:
                assignment = old_assignment
                loads = old_loads

            if time.time() - last_peak_improvement > max(0.005, budget * 0.035):
                assignment = list(best_assignment)
                loads = [0.0] * used_machine_count
                for j in range(n):
                    loads[assignment[j]] += combined[j]
                current_peak = max(loads)
                last_peak_improvement = time.time()

        return constructed_state(best_assignment, state[1])

    def improve_sequence(state, stop_time):
        assignment, sequence, value, prepared = state
        sequence = list(sequence)

        while time.time() < stop_time and n > 1:
            chosen_sequence = None
            chosen_value = value
            positions = list(range(n))
            rng.shuffle(positions)

            # Insertion can both reorder ready work and deliberately idle for a
            # soon-to-be-released job with a long downstream tail.
            for old_pos in positions:
                reduced = sequence[:old_pos] + sequence[old_pos + 1 :]
                job = sequence[old_pos]
                start_pos = rng.randrange(n)
                for step in range(n):
                    if time.time() >= stop_time:
                        break
                    new_pos = (start_pos + step) % n
                    if new_pos == old_pos:
                        continue
                    candidate_sequence = reduced[:new_pos] + [job] + reduced[new_pos:]
                    candidate_value = sequence_score(prepared, candidate_sequence)
                    if candidate_value < chosen_value - 1e-10:
                        chosen_sequence = candidate_sequence
                        chosen_value = candidate_value
                if time.time() >= stop_time:
                    break

            if chosen_sequence is None:
                break
            sequence = chosen_sequence
            value = chosen_value
            state = (assignment, sequence, value, prepared)
            consider(state)

        return list(assignment), sequence, value, prepared

    try:
        for assignment in seeds[1:]:
            if time.time() >= deadline:
                break
            state = constructed_state(assignment)
            consider(state)
            if state[2] < current[2]:
                current = state

        if time.time() < balance_phase_end and used_machine_count > 1:
            load_state = improve_load_partition(best, balance_phase_end)
            consider(load_state)
            if load_state[2] < current[2]:
                current = load_state

        if time.time() < phase_one_end and used_machine_count > 1:
            current = improve_assignment(best, phase_one_end)
            consider(current)

        if time.time() < phase_two_end:
            current = improve_sequence(best, phase_two_end)
            consider(current)

        # Joint anytime search.  It alternates assignment and sequencing
        # neighborhoods, accepts controlled uphill moves to cross local-optimum
        # barriers, and restarts from the incumbent after prolonged stagnation.
        current = best
        last_improvement = time.time()
        while time.time() < deadline:
            assignment = list(current[0])
            sequence = list(current[1])
            roll = rng.random()

            if roll < 0.38 and used_machine_count > 1:
                j = rng.randrange(n)
                old_m = assignment[j]
                new_m = rng.randrange(used_machine_count - 1)
                if new_m >= old_m:
                    new_m += 1
                assignment[j] = new_m
                candidate = constructed_state(assignment, sequence)
            elif roll < 0.57 and used_machine_count > 1 and n > 1:
                i = rng.randrange(n)
                j = rng.randrange(n - 1)
                if j >= i:
                    j += 1
                assignment[i], assignment[j] = assignment[j], assignment[i]
                candidate = constructed_state(assignment, sequence)
            else:
                if n <= 1:
                    candidate = current
                elif roll < 0.82:
                    old_pos = rng.randrange(n)
                    new_pos = rng.randrange(n)
                    job = sequence.pop(old_pos)
                    sequence.insert(new_pos, job)
                    candidate = (
                        assignment,
                        sequence,
                        sequence_score(current[3], sequence),
                        current[3],
                    )
                else:
                    left = rng.randrange(n - 1)
                    right = rng.randrange(left + 1, n)
                    sequence[left : right + 1] = reversed(sequence[left : right + 1])
                    candidate = (
                        assignment,
                        sequence,
                        sequence_score(current[3], sequence),
                        current[3],
                    )

            now = time.time()
            remaining_fraction = max(0.0, (deadline - now) / max(budget, 1e-9))
            temperature = max(1e-7, best[2] * (0.0004 + 0.008 * remaining_fraction))
            delta = candidate[2] - current[2]
            if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
                current = candidate

            if candidate[2] < best[2] - 1e-10:
                consider(candidate)
                last_improvement = now
                # Spend a short wall-clock slice polishing every new incumbent.
                slice_end = min(deadline, now + max(0.001, budget * 0.012))
                polished = improve_sequence(best, slice_end)
                consider(polished)
                current = polished

            # Time-based restart, not an iteration counter.
            if now - last_improvement > max(0.01, budget * 0.08):
                current = best
                assignment = list(best[0])
                shake_count = max(1, min(n, 1 + int(4.0 * remaining_fraction)))
                for _ in range(shake_count):
                    if used_machine_count > 1:
                        assignment[rng.randrange(n)] = rng.randrange(used_machine_count)
                current = constructed_state(assignment, best[1])
                last_improvement = now

    except Exception:
        # The complete feasible incumbent created before the search is always
        # available, including when an unusual instance exhausts a neighborhood.
        pass

    final_assignment = list(best[0])
    final_sequence = list(best[1])

    # Rebuild the earliest feasible three-stage schedule from the chosen
    # assignment and server permutation.
    s1 = [0.0] * n
    s2 = [0.0] * n
    s3 = [0.0] * n
    init_free = [0.0] * used_machine_count
    for j in range(n):
        m = final_assignment[j]
        s1[j] = init_free[m]
        init_free[m] += p1[j]

    server_free = 0.0
    for j in final_sequence:
        s2[j] = max(server_free, s1[j] + p1[j])
        server_free = s2[j] + p2[j]

    main_free = list(init_free)
    for j in range(n):
        m = final_assignment[j]
        s3[j] = max(main_free[m], s2[j] + p2[j])
        main_free[m] = s3[j] + p3[j]
    makespan = max(main_free)

    return {
        "machine_assignment": {
            f"job_{j}": int(final_assignment[j]) for j in range(n)
        },
        "server_sequence": [f"job_{j}" for j in final_sequence],
        "start_times": {
            f"job_{j}": [float(s1[j]), float(s2[j]), float(s3[j])]
            for j in range(n)
        },
        "makespan": float(makespan),
    }
