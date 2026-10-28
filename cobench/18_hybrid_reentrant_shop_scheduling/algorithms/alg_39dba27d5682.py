# Library slot 6/10  id=alg_39dba27d5682
# score=310.6087237863664  feasible_rate=1.0
# score_vec=[334.59999999999997, 808.5317789693559, 162.59784767048947, 161.72923292906808, 207.73733873843705, 198.8520395563141, 223.39999999999998, 346.38099999999986, 326.51699999999977, 335.7409999999999]

import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime memetic search for the hybrid reentrant shop problem."""
    t0 = time.time()
    deadline = t0 + max(0.0, float(time_limit_s)) * 0.85

    n = int(instance["num_jobs"])
    machine_count = int(instance["num_machines"])
    p1 = [float(value) for value in instance["init_processing_times"]]
    p2 = [float(value) for value in instance["setup_processing_times"]]
    p3 = [float(value) for value in instance["main_processing_times"]]

    if n == 0:
        return {
            "machine_assignment": {},
            "server_sequence": [],
            "start_times": {},
            "makespan": 0.0,
        }

    # For a fixed assignment and server permutation, this decoder schedules every
    # phase as early as the stated constraints permit.  Jobs sharing a primary
    # machine follow natural job order in both machine-bound phases; the server
    # permutation determines only the setup order.
    def decode(assignment, sequence):
        # Initialization always follows natural job order on each machine.
        # The server permutation controls setup only and must not reorder s1.
        s1 = [0.0] * n
        machine_free = [0.0] * machine_count
        for job in range(n):
            machine = assignment[job]
            s1[job] = machine_free[machine]
            machine_free[machine] += p1[job]

        init_done = machine_free[:]
        s2 = [0.0] * n
        server_free = 0.0
        for job in sequence:
            start = max(server_free, s1[job] + p1[job])
            s2[job] = start
            server_free = start + p2[job]

        s3 = [0.0] * n
        main_free = init_done[:]
        makespan = 0.0
        completion_sum = 0.0
        critical_job = 0
        for job in range(n):
            machine = assignment[job]
            start = max(s2[job] + p2[job], main_free[machine])
            s3[job] = start
            completion = start + p3[job]
            main_free[machine] = completion
            completion_sum += completion
            if completion > makespan:
                makespan = completion
                critical_job = job

        # The second component only guides motion across equal-makespan plateaus.
        fitness = (makespan, completion_sum / n + 0.05 * server_free)
        return fitness, critical_job, s1, s2, s3

    def balanced_assignment(tie_offset=0):
        if machine_count == 1:
            return [0] * n
        loads = [0.0] * machine_count
        assignment = [0] * n
        previous = -1
        for job in range(n):
            choices = [machine for machine in range(machine_count)
                       if machine != previous]
            machine = min(
                choices,
                key=lambda value: (
                    loads[value] + p1[job],
                    (value - tie_offset) % machine_count,
                ),
            )
            assignment[job] = machine
            loads[machine] += p1[job]
            previous = machine
        return assignment

    def tail_aware_assignment(tie_offset=0):
        """Balance initialization load without concentrating large main tails."""
        if machine_count == 1:
            return [0] * n
        loads = [0.0] * machine_count
        largest_tail = [0.0] * machine_count
        assignment = [0] * n
        previous = -1
        for job in range(n):
            choices = [machine for machine in range(machine_count)
                       if machine != previous]
            machine = min(
                choices,
                key=lambda value: (
                    loads[value] + p1[job]
                    + max(largest_tail[value], p3[job]),
                    loads[value],
                    (value - tie_offset) % machine_count,
                ),
            )
            assignment[job] = machine
            loads[machine] += p1[job]
            largest_tail[machine] = max(largest_tail[machine], p3[job])
            previous = machine
        return assignment

    # The fallback is deliberately obtained before any substantial construction.
    base_assignment = balanced_assignment(0)
    base_sequence = sorted(range(n), key=lambda job: (p2[job], -p3[job], job))
    base_fitness, base_critical, _, _, _ = decode(
        base_assignment, base_sequence
    )
    best_assignment = base_assignment[:]
    best_sequence = base_sequence[:]
    best_fitness = base_fitness

    seed_value = n * 1000003 + machine_count * 9176
    for job in range(n):
        seed_value += int((p1[job] * 17.0 + p2[job] * 31.0
                          + p3[job] * 13.0) * 1000.0)
    rng = random.Random(seed_value)

    # Each pool entry is (fitness, assignment, sequence, critical_job).
    pool = [(base_fitness, base_assignment[:], base_sequence[:], base_critical)]
    pool_limit = 20

    def same_genome(left_assignment, left_sequence,
                    right_assignment, right_sequence):
        return (left_assignment == right_assignment
                and left_sequence == right_sequence)

    def offer(assignment, sequence, evaluated=None):
        nonlocal best_assignment, best_sequence, best_fitness
        if evaluated is None:
            fitness, critical, _, _, _ = decode(assignment, sequence)
        else:
            fitness, critical = evaluated

        if fitness < best_fitness:
            best_fitness = fitness
            best_assignment = assignment[:]
            best_sequence = sequence[:]

        for old_fitness, old_assignment, old_sequence, _ in pool:
            if same_genome(assignment, sequence, old_assignment, old_sequence):
                return fitness, critical

        pool.append((fitness, assignment[:], sequence[:], critical))
        pool.sort(key=lambda entry: entry[0])

        if len(pool) > pool_limit:
            # Keep the best half unconditionally.  In the other half, retain
            # structurally different genomes so crossover has useful material.
            elite_count = pool_limit // 2
            retained = pool[:elite_count]
            candidates = pool[elite_count:]
            while candidates and len(retained) < pool_limit:
                best_index = 0
                best_distance = -1
                for index, entry in enumerate(candidates):
                    assignment_distance = sum(
                        1 for job in range(n)
                        if entry[1][job] != retained[0][1][job]
                    )
                    step = max(1, n // 12)
                    sequence_distance = sum(
                        1 for pos in range(0, n, step)
                        if entry[2][pos] != retained[0][2][pos]
                    )
                    distance = assignment_distance + sequence_distance
                    if distance > best_distance:
                        best_distance = distance
                        best_index = index
                retained.append(candidates.pop(best_index))
            pool[:] = sorted(retained, key=lambda entry: entry[0])
        return fitness, critical

    def greedy_sequence(assignment, tail_weight, randomized=False):
        remaining = list(range(n))
        machine_free = [0.0] * machine_count
        server_free = 0.0
        sequence = []
        while remaining:
            if time.time() >= deadline:
                sequence.extend(remaining)
                break
            scored = []
            for job in remaining:
                init_completion = machine_free[assignment[job]] + p1[job]
                server_completion = max(server_free, init_completion) + p2[job]
                score = server_completion - tail_weight * p3[job]
                scored.append((score, server_completion, job))
            scored.sort()
            pick_index = 0
            if randomized and len(scored) > 1:
                width = min(4, len(scored))
                pick_index = rng.randrange(width)
            _, server_completion, job = scored[pick_index]
            sequence.append(job)
            machine_free[assignment[job]] += p1[job]
            server_free = server_completion
            remaining.remove(job)
        return sequence

    # Diverse deterministic rules make the first pool useful immediately.  They
    # also cover the characteristic strengths of SPT and large-tail-first orders.
    try:
        assignment_seeds = [base_assignment]
        if machine_count > 1:
            assignment_seeds.append(balanced_assignment(1 % machine_count))
            assignment_seeds.append(tail_aware_assignment(0))
            assignment_seeds.append(
                [job % machine_count for job in range(n)]
            )

        simple_sequences = [
            list(range(n)),
            sorted(range(n), key=lambda job: (p1[job], p2[job], job)),
            sorted(range(n), key=lambda job: (-p3[job], p2[job], job)),
            sorted(range(n), key=lambda job: (p1[job] + p2[job], -p3[job])),
        ]
        for assignment in assignment_seeds:
            for sequence in simple_sequences:
                if time.time() >= deadline:
                    break
                offer(assignment, sequence)
            for tail_weight in (0.25, 0.6, 1.0, 1.8):
                if time.time() >= deadline:
                    break
                sequence = greedy_sequence(assignment, tail_weight)
                offer(assignment, sequence)

        def tournament():
            sample_size = min(3, len(pool))
            contestants = rng.sample(pool, sample_size)
            return min(contestants, key=lambda entry: entry[0])

        def crossover(first, second):
            first_assignment, first_sequence = first[1], first[2]
            second_assignment, second_sequence = second[1], second[2]

            child_assignment = [
                first_assignment[job] if rng.random() < 0.5
                else second_assignment[job]
                for job in range(n)
            ]
            if machine_count > 1 and rng.random() < 0.85:
                loads = [0.0] * machine_count
                for job in range(n):
                    if job > 0 and child_assignment[job] == child_assignment[job - 1]:
                        alternatives = [
                            machine for machine in range(machine_count)
                            if machine != child_assignment[job - 1]
                        ]
                        parent_choice = second_assignment[job]
                        if parent_choice in alternatives and rng.random() < 0.5:
                            child_assignment[job] = parent_choice
                        else:
                            child_assignment[job] = min(
                                alternatives,
                                key=lambda machine: loads[machine] + p1[job],
                            )
                    loads[child_assignment[job]] += p1[job]

            if n < 2:
                return child_assignment, first_sequence[:]
            left, right = sorted(rng.sample(range(n + 1), 2))
            if left == right:
                right = min(n, left + 1)
            child_sequence = [-1] * n
            used = set()
            for pos in range(left, right):
                job = first_sequence[pos]
                child_sequence[pos] = job
                used.add(job)
            fill = [job for job in second_sequence if job not in used]
            fill_pos = 0
            for pos in range(n):
                if child_sequence[pos] == -1:
                    child_sequence[pos] = fill[fill_pos]
                    fill_pos += 1
            return child_assignment, child_sequence

        def mutate(assignment, sequence, critical):
            new_assignment = assignment[:]
            new_sequence = sequence[:]
            move = rng.random()

            if move < 0.14 and machine_count > 1:
                job = critical if rng.random() < 0.55 else rng.randrange(n)
                forbidden = set()
                if job > 0:
                    forbidden.add(new_assignment[job - 1])
                if job + 1 < n:
                    forbidden.add(new_assignment[job + 1])
                choices = [machine for machine in range(machine_count)
                           if machine != new_assignment[job]
                           and machine not in forbidden]
                if not choices:
                    choices = [machine for machine in range(machine_count)
                               if machine != new_assignment[job]]
                if choices:
                    new_assignment[job] = rng.choice(choices)
            elif move < 0.22 and machine_count > 1 and n > 1:
                first, second = rng.sample(range(n), 2)
                new_assignment[first], new_assignment[second] = (
                    new_assignment[second], new_assignment[first]
                )
            elif move < 0.49 and n > 1:
                old_pos = new_sequence.index(critical)
                if old_pos > 0 and rng.random() < 0.75:
                    new_pos = rng.randrange(old_pos)
                else:
                    new_pos = rng.randrange(n)
                job = new_sequence.pop(old_pos)
                new_sequence.insert(new_pos, job)
            elif move < 0.72 and n > 1:
                first, second = rng.sample(range(n), 2)
                new_sequence[first], new_sequence[second] = (
                    new_sequence[second], new_sequence[first]
                )
            elif move < 0.88 and n > 2:
                first, second = sorted(rng.sample(range(n), 2))
                new_sequence[first:second + 1] = reversed(
                    new_sequence[first:second + 1]
                )
            elif n > 1:
                old_pos = rng.randrange(n)
                job = new_sequence.pop(old_pos)
                new_sequence.insert(rng.randrange(n), job)
                if machine_count > 1 and rng.random() < 0.4:
                    recolor = rng.randrange(n)
                    choices = [machine for machine in range(machine_count)
                               if machine != new_assignment[recolor]]
                    if choices:
                        new_assignment[recolor] = rng.choice(choices)
            return new_assignment, new_sequence

        # Start with an elite, then continuously alternate LAHC intensification
        # and population recombination.  No iteration count terminates the search.
        current = pool[0]
        current_fitness = current[0]
        current_assignment = current[1][:]
        current_sequence = current[2][:]
        current_critical = current[3]
        history_length = max(40, min(180, n * 3))
        history = [current_fitness] * history_length
        history_pos = 0
        stagnant = 0
        restart_after = max(140, n * 6)

        while time.time() < deadline:
            new_assignment, new_sequence = mutate(
                current_assignment, current_sequence, current_critical
            )
            if time.time() >= deadline:
                break
            new_fitness, new_critical, _, _, _ = decode(
                new_assignment, new_sequence
            )

            threshold = history[history_pos]
            if new_fitness <= current_fitness or new_fitness <= threshold:
                current_assignment = new_assignment
                current_sequence = new_sequence
                current_fitness = new_fitness
                current_critical = new_critical
                if new_fitness < best_fitness:
                    offer(
                        new_assignment,
                        new_sequence,
                        (new_fitness, new_critical),
                    )
                    stagnant = 0
                else:
                    stagnant += 1
            else:
                stagnant += 1

            history[history_pos] = current_fitness
            history_pos = (history_pos + 1) % history_length

            # Before abandoning a basin, exhaust the most relevant insertion
            # neighborhood: every position of the job defining the makespan.
            if (stagnant == restart_after // 2 and n > 1
                    and time.time() < deadline):
                job = current_critical
                reduced_sequence = [value for value in current_sequence
                                    if value != job]
                insertion_fitness = current_fitness
                insertion_sequence = current_sequence
                insertion_critical = current_critical
                for pos in range(n):
                    if time.time() >= deadline:
                        break
                    candidate_sequence = reduced_sequence[:]
                    candidate_sequence.insert(pos, job)
                    candidate_fitness, candidate_critical, _, _, _ = decode(
                        current_assignment, candidate_sequence
                    )
                    if candidate_fitness < insertion_fitness:
                        insertion_fitness = candidate_fitness
                        insertion_sequence = candidate_sequence
                        insertion_critical = candidate_critical
                if insertion_fitness < current_fitness:
                    current_sequence = insertion_sequence
                    current_fitness = insertion_fitness
                    current_critical = insertion_critical
                    offer(
                        current_assignment,
                        current_sequence,
                        (current_fitness, current_critical),
                    )
                    history = [current_fitness] * history_length
                    history_pos = 0
                    stagnant = 0

            if stagnant >= restart_after and time.time() < deadline:
                first_parent = tournament()
                second_parent = tournament()
                child_assignment, child_sequence = crossover(
                    first_parent, second_parent
                )

                # A randomized greedy immigrant prevents a converged pool from
                # repeatedly recombining identical local optima.
                if rng.random() < 0.25:
                    child_assignment = balanced_assignment(
                        rng.randrange(machine_count)
                    )
                    child_sequence = greedy_sequence(
                        child_assignment,
                        rng.choice((0.25, 0.6, 1.0, 1.8)),
                        randomized=True,
                    )
                if time.time() >= deadline:
                    break
                child_fitness, child_critical, _, _, _ = decode(
                    child_assignment, child_sequence
                )
                offer(
                    child_assignment,
                    child_sequence,
                    (child_fitness, child_critical),
                )
                current_assignment = child_assignment
                current_sequence = child_sequence
                current_fitness = child_fitness
                current_critical = child_critical
                history = [current_fitness] * history_length
                history_pos = 0
                stagnant = 0
    except Exception:
        # The fully feasible construction made before the search remains valid.
        pass

    final_fitness, _, best_s1, best_s2, best_s3 = decode(
        best_assignment, best_sequence
    )
    makespan = final_fitness[0]
    return {
        "machine_assignment": {
            f"job_{job}": int(best_assignment[job]) for job in range(n)
        },
        "server_sequence": [f"job_{job}" for job in best_sequence],
        "start_times": {
            f"job_{job}": [
                float(best_s1[job]),
                float(best_s2[job]),
                float(best_s3[job]),
            ]
            for job in range(n)
        },
        "makespan": float(makespan),
    }
