# Library slot 3/10  id=alg_dacbf9b1107b
# score=309.4331332913113  feasible_rate=1.0
# score_vec=[334.6, 808.5316615311067, 160.55515161610003, 161.19993231255728, 205.4344589084281, 197.6821285449219, 223.39999999999998, 345.85799999999983, 326.51699999999977, 330.5529999999999]

import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime iterated-greedy solver for the hybrid reentrant shop."""
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

    # For a fixed assignment and server permutation, decode all reported times
    # from the authoritative natural-order and resource constraints.
    def decode(assignment, sequence, keep_times=False):
        # Initialization follows natural job order on each assigned machine.
        s1 = [0.0] * n
        machine_free = [0.0] * machine_count
        for job in range(n):
            machine = assignment[job]
            s1[job] = machine_free[machine]
            machine_free[machine] += p1[job]

        s2 = [0.0] * n
        server_free = 0.0
        for job in sequence:
            start = max(server_free, s1[job] + p1[job])
            s2[job] = start
            server_free = start + p2[job]

        # A machine's main phase starts only after all initialization assigned
        # to it is complete, then proceeds in natural job order.
        s3 = [0.0] * n
        main_free = machine_free[:]
        completions = [0.0] * n
        makespan = 0.0
        critical_job = 0
        completion_sum = 0.0
        completion_square_sum = 0.0
        for job in range(n):
            machine = assignment[job]
            start = max(s2[job] + p2[job], main_free[machine])
            s3[job] = start
            completion = start + p3[job]
            main_free[machine] = completion
            completions[job] = completion
            completion_sum += completion
            completion_square_sum += completion * completion
            if completion > makespan:
                makespan = completion
                critical_job = job

        # Extra components supply a gradient on broad equal-makespan plateaus.
        fitness = (
            makespan,
            completion_square_sum / n,
            completion_sum / n,
            server_free,
        )
        if keep_times:
            return fitness, critical_job, s1, s2, s3, completions
        return fitness, critical_job

    def natural_assignment(mode, offset):
        if machine_count == 1:
            return [0] * n
        loads = [0.0] * machine_count
        tails = [0.0] * machine_count
        assignment = [0] * n
        previous = -1
        for job in range(n):
            choices = [
                machine for machine in range(machine_count)
                if machine != previous
            ]
            if mode == 0:
                key = lambda machine: (
                    loads[machine] + p1[job],
                    (machine - offset) % machine_count,
                )
            else:
                key = lambda machine: (
                    loads[machine] + p1[job]
                    + max(tails[machine], p3[job]),
                    loads[machine] + p1[job],
                    (machine - offset) % machine_count,
                )
            machine = min(choices, key=key)
            assignment[job] = machine
            loads[machine] += p1[job]
            tails[machine] = max(tails[machine], p3[job])
            previous = machine
        return assignment

    def global_assignment(mode, offset):
        """LPT-style assignment, unlike the champion's one-pass index order."""
        if machine_count == 1:
            return [0] * n
        if mode == 0:
            order = sorted(range(n), key=lambda j: (-p1[j], -p3[j], j))
        else:
            order = sorted(
                range(n), key=lambda j: (-(p1[j] + p3[j]), -p1[j], j)
            )
        loads = [0.0] * machine_count
        tails = [0.0] * machine_count
        assignment = [-1] * n
        for job in order:
            forbidden = set()
            if job > 0 and assignment[job - 1] >= 0:
                forbidden.add(assignment[job - 1])
            if job + 1 < n and assignment[job + 1] >= 0:
                forbidden.add(assignment[job + 1])
            choices = [
                machine for machine in range(machine_count)
                if machine not in forbidden
            ]
            if not choices:
                choices = list(range(machine_count))
            machine = min(
                choices,
                key=lambda value: (
                    loads[value] + p1[job]
                    + max(tails[value], p3[job]),
                    loads[value] + p1[job],
                    (value - offset) % machine_count,
                ),
            )
            assignment[job] = machine
            loads[machine] += p1[job]
            tails[machine] = max(tails[machine], p3[job])
        return assignment

    def dispatch_sequence(assignment, slack_factor):
        """Schrage-like construction with endogenous initialization releases."""
        remaining = list(range(n))
        loads = [0.0] * machine_count
        server_free = 0.0
        sequence = []
        mean_setup = sum(p2) / n
        while remaining:
            if time.time() >= deadline:
                sequence.extend(remaining)
                break
            earliest = min(
                loads[assignment[job]] + p1[job] for job in remaining
            )
            threshold = max(server_free, earliest) + slack_factor * mean_setup
            ready = [
                job for job in remaining
                if loads[assignment[job]] + p1[job] <= threshold
            ]
            job = min(
                ready,
                key=lambda value: (
                    -p3[value],
                    p2[value],
                    loads[assignment[value]] + p1[value],
                    value,
                ),
            )
            machine = assignment[job]
            release = loads[machine] + p1[job]
            server_free = max(server_free, release) + p2[job]
            loads[machine] = release
            sequence.append(job)
            remaining.remove(job)
        return sequence

    # Immediate fallback: proper coloring, balanced initialization, long tails
    # early on the server.  It exists before any expensive construction.
    base_assignment = natural_assignment(1, 0)
    base_sequence = sorted(range(n), key=lambda job: (-p3[job], p2[job], job))
    best_fitness, best_critical = decode(base_assignment, base_sequence)
    best_assignment = base_assignment[:]
    best_sequence = base_sequence[:]

    seed_value = n * 1000003 + machine_count * 9176
    for job in range(n):
        seed_value += int(
            (17.0 * p1[job] + 31.0 * p2[job] + 13.0 * p3[job])
            * 1000.0
        )
    rng = random.Random(seed_value)

    # Small elite set supports restarts from distinct locally good structures.
    elite = []

    def offer(assignment, sequence, evaluated=None):
        nonlocal best_fitness, best_assignment, best_sequence, best_critical
        if evaluated is None:
            fitness, critical = decode(assignment, sequence)
        else:
            fitness, critical = evaluated
        if fitness < best_fitness:
            best_fitness = fitness
            best_assignment = assignment[:]
            best_sequence = sequence[:]
            best_critical = critical

        genome = (assignment[:], sequence[:], fitness, critical)
        for old_assignment, old_sequence, _, _ in elite:
            if old_assignment == assignment and old_sequence == sequence:
                return fitness, critical
        elite.append(genome)
        elite.sort(key=lambda entry: entry[2])
        if len(elite) > 10:
            elite.pop()
        return fitness, critical

    assignments = [base_assignment]
    if time.time() < deadline:
        assignments.append(natural_assignment(0, 0))
    for mode in range(2):
        if time.time() >= deadline:
            break
        assignments.append(global_assignment(mode, mode))
    if machine_count > 1 and time.time() < deadline:
        assignments.append(
            [job % machine_count for job in range(n)]
        )

    for assignment in assignments:
        if time.time() >= deadline:
            break
        sequences = [
            base_sequence,
            list(range(n)),
            sorted(range(n), key=lambda job: (p2[job], -p3[job], job)),
            sorted(range(n), key=lambda job: (p1[job], -p3[job], job)),
            dispatch_sequence(assignment, 0.0),
            dispatch_sequence(assignment, 1.5),
            dispatch_sequence(assignment, 4.0),
        ]
        for sequence in sequences:
            if time.time() >= deadline:
                break
            offer(assignment, sequence)

    offer(best_assignment, best_sequence, (best_fitness, best_critical))

    def relocate_descent(assignment, sequence, fitness, stop_time):
        """Best-improvement insertion neighborhood over the server order."""
        while time.time() < stop_time:
            move_fitness = fitness
            move_sequence = None
            for old_pos in range(n):
                if time.time() >= stop_time:
                    break
                job = sequence[old_pos]
                reduced = sequence[:old_pos] + sequence[old_pos + 1:]
                for new_pos in range(n):
                    if new_pos == old_pos:
                        continue
                    if time.time() >= stop_time:
                        break
                    candidate = reduced[:new_pos] + [job] + reduced[new_pos:]
                    candidate_fitness, _ = decode(assignment, candidate)
                    if candidate_fitness < move_fitness:
                        move_fitness = candidate_fitness
                        move_sequence = candidate
            if move_sequence is None:
                break
            sequence = move_sequence
            fitness = move_fitness
            offer(assignment, sequence)
        return assignment, sequence, fitness

    def recolor_descent(assignment, sequence, fitness, stop_time):
        """Best single-job assignment move, including non-proper colorings."""
        while time.time() < stop_time and machine_count > 1:
            move_fitness = fitness
            move_assignment = None
            for job in range(n):
                if time.time() >= stop_time:
                    break
                old_machine = assignment[job]
                for machine in range(machine_count):
                    if machine == old_machine:
                        continue
                    if time.time() >= stop_time:
                        break
                    candidate = assignment[:]
                    candidate[job] = machine
                    candidate_fitness, _ = decode(candidate, sequence)
                    if candidate_fitness < move_fitness:
                        move_fitness = candidate_fitness
                        move_assignment = candidate
            if move_assignment is None:
                break
            assignment = move_assignment
            fitness = move_fitness
            offer(assignment, sequence)
        return assignment, sequence, fitness

    def swap_descent(assignment, sequence, fitness, stop_time):
        """A swap crosses barriers that a single relocate cannot improve."""
        move_fitness = fitness
        move_sequence = None
        for left in range(n - 1):
            if time.time() >= stop_time:
                break
            for right in range(left + 1, n):
                if time.time() >= stop_time:
                    break
                candidate = sequence[:]
                candidate[left], candidate[right] = (
                    candidate[right], candidate[left]
                )
                candidate_fitness, _ = decode(assignment, candidate)
                if candidate_fitness < move_fitness:
                    move_fitness = candidate_fitness
                    move_sequence = candidate
        if move_sequence is not None:
            sequence = move_sequence
            fitness = move_fitness
            offer(assignment, sequence)
        return assignment, sequence, fitness

    def assignment_swap_descent(
        assignment, sequence, fitness, stop_time
    ):
        """Exchange two colors; vital when all one-job recolors are blocked."""
        while time.time() < stop_time and machine_count > 1:
            move_fitness = fitness
            move_assignment = None
            for left in range(n - 1):
                if time.time() >= stop_time:
                    break
                for right in range(left + 1, n):
                    if assignment[left] == assignment[right]:
                        continue
                    if time.time() >= stop_time:
                        break
                    candidate = assignment[:]
                    candidate[left], candidate[right] = (
                        candidate[right], candidate[left]
                    )
                    candidate_fitness, _ = decode(candidate, sequence)
                    if candidate_fitness < move_fitness:
                        move_fitness = candidate_fitness
                        move_assignment = candidate
            if move_assignment is None:
                break
            assignment = move_assignment
            fitness = move_fitness
            offer(assignment, sequence)
        return assignment, sequence, fitness

    def machine_bound(assignment):
        """Exact assignment-induced bound, including equal-color p3 chains."""
        loads = [0.0] * machine_count
        longest_run = [0.0] * machine_count
        last_machine = -1
        run_tail = 0.0
        for job in range(n):
            machine = assignment[job]
            loads[machine] += p1[job]
            if machine == last_machine:
                run_tail += p3[job]
            else:
                if last_machine >= 0 and run_tail > longest_run[last_machine]:
                    longest_run[last_machine] = run_tail
                last_machine = machine
                run_tail = p3[job]
        if run_tail > longest_run[last_machine]:
            longest_run[last_machine] = run_tail
        return max(
            loads[machine] + longest_run[machine]
            for machine in range(machine_count)
        )

    def anneal_assignment(start_assignment, sequence, stop_time):
        """Fast ejection search on the main machine-load bottleneck."""
        if machine_count <= 1 or time.time() >= stop_time:
            return
        phase_start = time.time()
        duration = max(0.000001, stop_time - phase_start)
        current = start_assignment[:]
        current_bound = machine_bound(current)
        bound_best = current[:]
        bound_best_value = current_bound
        last_cycle = -1
        temperature_scale = max(max(p1) * 0.55, 1.0)
        preserve_proper = (
            machine_count > 2
            and all(current[job] != current[job - 1] for job in range(1, n))
        )

        while time.time() < stop_time:
            now = time.time()
            progress = min(0.999999, (now - phase_start) / duration)
            cycle = int(progress * 10.0)
            if cycle != last_cycle:
                current = bound_best[:]
                current_bound = bound_best_value
                last_cycle = cycle
            cycle_progress = progress * 10.0 - cycle
            temperature = temperature_scale * (0.003 ** cycle_progress)

            candidate = current[:]
            if rng.random() < 0.68:
                job = rng.randrange(n)
                old_machine = candidate[job]
                choices = [
                    machine for machine in range(machine_count)
                    if machine != old_machine
                    and (job == 0 or candidate[job - 1] != machine)
                    and (job + 1 == n or candidate[job + 1] != machine)
                ]
                if not choices:
                    choices = [
                        machine for machine in range(machine_count)
                        if machine != old_machine
                    ]
                candidate[job] = choices[rng.randrange(len(choices))]
            else:
                left = rng.randrange(n)
                right = rng.randrange(n)
                if left == right or candidate[left] == candidate[right]:
                    continue
                candidate[left], candidate[right] = (
                    candidate[right], candidate[left]
                )
                if preserve_proper and any(
                    candidate[job] == candidate[job - 1]
                    for job in range(1, n)
                ):
                    continue

            candidate_bound = machine_bound(candidate)
            delta = candidate_bound - current_bound
            if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
                current = candidate
                current_bound = candidate_bound
            if candidate_bound < bound_best_value:
                bound_best = candidate[:]
                bound_best_value = candidate_bound
        # Full decoding is the authority; the bound only proposes assignments.
        offer(bound_best, sequence)
        if time.time() < deadline:
            offer(bound_best, dispatch_sequence(bound_best, 0.0))
        if time.time() < deadline:
            offer(bound_best, dispatch_sequence(bound_best, 1.5))

    # First deepen the best construction.  Later phases use short time slices so
    # one local basin cannot monopolize the anytime budget.
    current_assignment = best_assignment[:]
    current_sequence = best_sequence[:]
    current_fitness = best_fitness
    anneal_remaining = deadline - time.time()
    anneal_stop = min(
        deadline, time.time() + max(0.0, anneal_remaining * 0.22)
    )
    anneal_assignment(current_assignment, current_sequence, anneal_stop)
    current_assignment = best_assignment[:]
    current_sequence = best_sequence[:]
    current_fitness = best_fitness
    initial_remaining = deadline - time.time()
    first_slice = min(
        deadline, time.time() + max(0.01, initial_remaining * 0.06)
    )
    current_assignment, current_sequence, current_fitness = relocate_descent(
        current_assignment, current_sequence, current_fitness, first_slice
    )
    second_slice = min(
        deadline, time.time() + max(0.01, initial_remaining * 0.10)
    )
    current_assignment, current_sequence, current_fitness = recolor_descent(
        current_assignment, current_sequence, current_fitness, second_slice
    )
    third_slice = min(
        deadline, time.time() + max(0.01, initial_remaining * 0.12)
    )
    current_assignment, current_sequence, current_fitness = assignment_swap_descent(
        current_assignment, current_sequence, current_fitness, third_slice
    )

    restart_number = 0
    while time.time() < deadline:
        restart_number += 1
        # Alternate between the incumbent and other elite basins.
        if elite and restart_number % 3 == 0:
            source = elite[rng.randrange(min(len(elite), 6))]
            assignment = source[0][:]
            sequence = source[1][:]
        else:
            assignment = best_assignment[:]
            sequence = best_sequence[:]

        # Ruin several interacting decisions.  The size varies continuously
        # across restarts, providing both fine and coarse neighborhoods.
        if n > 1:
            max_ruin = max(2, min(n, 2 + int(n ** 0.5)))
            ruin_size = rng.randint(2, max_ruin)
            chosen = rng.sample(range(n), ruin_size)
            chosen_set = set(chosen)
            removed = [job for job in sequence if job in chosen_set]
            kept = [job for job in sequence if job not in chosen_set]
            rng.shuffle(removed)
            insert_at = rng.randrange(len(kept) + 1)
            sequence = kept[:insert_at] + removed + kept[insert_at:]

            if machine_count > 1:
                recolors = max(1, ruin_size // 2)
                for job in chosen[:recolors]:
                    assignment[job] = rng.randrange(machine_count)

        fitness, _ = decode(assignment, sequence)
        # Adaptive wall-clock slice: many restarts early, deeper searches late.
        remaining = deadline - time.time()
        slice_seconds = min(remaining, max(0.025, remaining * 0.12))
        phase_start = time.time()
        recolor_stop = min(deadline, phase_start + slice_seconds * 0.30)

        assignment, sequence, fitness = recolor_descent(
            assignment, sequence, fitness, recolor_stop
        )
        assignment_swap_stop = min(
            deadline, phase_start + slice_seconds * 0.60
        )
        assignment, sequence, fitness = assignment_swap_descent(
            assignment, sequence, fitness, assignment_swap_stop
        )
        relocate_stop = min(deadline, phase_start + slice_seconds * 0.88)
        assignment, sequence, fitness = relocate_descent(
            assignment, sequence, fitness, relocate_stop
        )
        stop_time = min(deadline, phase_start + slice_seconds)
        assignment, sequence, fitness = swap_descent(
            assignment, sequence, fitness, stop_time
        )
        offer(assignment, sequence)

    _, _, s1, s2, s3, _ = decode(
        best_assignment, best_sequence, keep_times=True
    )
    machine_assignment = {
        "job_%d" % job: int(best_assignment[job]) for job in range(n)
    }
    server_sequence = ["job_%d" % job for job in best_sequence]
    start_times = {
        "job_%d" % job: [float(s1[job]), float(s2[job]), float(s3[job])]
        for job in range(n)
    }
    makespan = max(s3[job] + p3[job] for job in range(n))
    return {
        "machine_assignment": machine_assignment,
        "server_sequence": server_sequence,
        "start_times": start_times,
        "makespan": float(makespan),
    }
