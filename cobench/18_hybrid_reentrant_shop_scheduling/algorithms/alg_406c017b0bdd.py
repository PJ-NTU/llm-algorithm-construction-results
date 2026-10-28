# Library slot 10/10  id=alg_406c017b0bdd
# score=313.76944299437656  feasible_rate=1.0
# score_vec=[354.20000000000005, 811.469368508452, 161.82272961979476, 162.56897865728666, 214.3771361941078, 203.5672169641244, 223.4, 346.02700000000004, 327.43099999999987, 332.831]

import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime joint search for the hybrid reentrant-shop problem."""
    t0 = time.time()
    deadline = t0 + max(0.0, float(time_limit_s)) * 0.85

    n = int(instance["num_jobs"])
    machine_count = max(1, int(instance["num_machines"]))
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

    jobs = list(range(n))
    eps = 1e-10

    def materialize(assignment, sequence):
        """Build the earliest feasible schedule for an assignment and sequence."""
        s1 = [0.0] * n
        init_end = [0.0] * machine_count
        for job in jobs:
            machine = assignment[job]
            s1[job] = init_end[machine]
            init_end[machine] += p1[job]

        s2 = [0.0] * n
        setup_end = [0.0] * n
        server_clock = 0.0
        for job in sequence:
            server_clock = max(server_clock, s1[job] + p1[job])
            s2[job] = server_clock
            server_clock += p2[job]
            setup_end[job] = server_clock

        s3 = [0.0] * n
        main_end = init_end[:]
        makespan = 0.0
        for job in jobs:
            machine = assignment[job]
            start = max(setup_end[job], main_end[machine])
            s3[job] = start
            main_end[machine] = start + p3[job]
            makespan = max(makespan, main_end[machine])

        return {
            "machine_assignment": {
                "job_{}".format(job): int(assignment[job]) for job in jobs
            },
            "server_sequence": ["job_{}".format(job) for job in sequence],
            "start_times": {
                "job_{}".format(job): [s1[job], s2[job], s3[job]]
                for job in jobs
            },
            "makespan": float(makespan),
        }

    def is_proper(assignment):
        for job in range(1, n):
            if assignment[job] == assignment[job - 1]:
                return False
        return True

    def assignment_data(assignment):
        loads = [0.0] * machine_count
        largest_tail = [0.0] * machine_count
        for job in jobs:
            machine = assignment[job]
            loads[machine] += p1[job]
            if p3[job] > largest_tail[machine]:
                largest_tail[machine] = p3[job]
        barrier = 0.0
        for machine in range(machine_count):
            barrier = max(barrier, loads[machine] + largest_tail[machine])
        return loads, barrier

    def proper_fitness(assignment, sequence, data=None):
        init_end = [0.0] * machine_count
        release = [0.0] * n
        for job in jobs:
            machine = assignment[job]
            init_end[machine] += p1[job]
            release[job] = init_end[machine]

        setup_end = [0.0] * n
        server_clock = 0.0
        for job in sequence:
            server_clock = max(server_clock, release[job]) + p2[job]
            setup_end[job] = server_clock

        main_end = init_end[:]
        makespan = 0.0
        completion_sum = 0.0
        for job in jobs:
            machine = assignment[job]
            completion = max(setup_end[job], main_end[machine]) + p3[job]
            main_end[machine] = completion
            completion_sum += completion
            if completion > makespan:
                makespan = completion
        return makespan, completion_sum / n

    def generic_fitness(assignment, sequence):
        solution = materialize(assignment, sequence)
        completion_sum = 0.0
        for job in jobs:
            completion_sum += solution["start_times"]["job_{}".format(job)][2] + p3[job]
        return solution["makespan"], completion_sum / n

    def fitness(assignment, sequence, data=None):
        if machine_count > 1 and is_proper(assignment):
            return proper_fitness(assignment, sequence, data)
        return generic_fitness(assignment, sequence)

    def better(left, right):
        if left[0] < right[0] - eps:
            return True
        if left[0] > right[0] + eps:
            return False
        return left[1] < right[1] - eps

    def balanced_assignment(mode, offset=0):
        if machine_count == 1:
            return [0] * n
        assignment = [0] * n
        loads = [0.0] * machine_count
        largest_tail = [0.0] * machine_count
        previous = -1
        for job in jobs:
            choices = [machine for machine in range(machine_count)
                       if machine != previous]
            if mode == 0:
                key = lambda machine: (
                    loads[machine] + p1[job],
                    (machine - offset) % machine_count,
                )
            elif mode == 1:
                key = lambda machine: (
                    loads[machine] + p1[job]
                    + max(largest_tail[machine], p3[job]),
                    loads[machine],
                    (machine - offset) % machine_count,
                )
            else:
                key = lambda machine: (
                    max(
                        loads[machine] + p1[job]
                        + max(largest_tail[machine], p3[job]),
                        max(
                            (loads[other] + largest_tail[other]
                             for other in range(machine_count)
                             if other != machine),
                            default=0.0,
                        ),
                    ),
                    loads[machine],
                    (machine - offset) % machine_count,
                )
            machine = min(choices, key=key)
            assignment[job] = machine
            loads[machine] += p1[job]
            largest_tail[machine] = max(largest_tail[machine], p3[job])
            previous = machine
        return assignment

    def colored_lpt(weight_mode, offset=0):
        if machine_count <= 2:
            return [(job + offset) % machine_count for job in jobs]
        if weight_mode == 0:
            weight = [p1[job] for job in jobs]
        elif weight_mode == 1:
            weight = [p1[job] + 0.25 * p3[job] for job in jobs]
        else:
            weight = [p1[job] + p3[job] for job in jobs]
        order = sorted(jobs, key=lambda job: (-weight[job], -p3[job], job))
        assignment = [-1] * n
        loads = [0.0] * machine_count
        largest_tail = [0.0] * machine_count
        for job in order:
            forbidden = set()
            if job > 0 and assignment[job - 1] >= 0:
                forbidden.add(assignment[job - 1])
            if job + 1 < n and assignment[job + 1] >= 0:
                forbidden.add(assignment[job + 1])
            choices = [machine for machine in range(machine_count)
                       if machine not in forbidden]
            machine = min(
                choices,
                key=lambda value: (
                    loads[value] + p1[job]
                    + max(largest_tail[value], p3[job]),
                    loads[value],
                    (value - offset) % machine_count,
                ),
            )
            assignment[job] = machine
            loads[machine] += p1[job]
            largest_tail[machine] = max(largest_tail[machine], p3[job])
        return assignment

    base_assignment = balanced_assignment(1)
    base_sequence = sorted(jobs, key=lambda job: (p2[job], -p3[job], job))
    best_assignment = list(base_assignment)
    best_sequence = list(base_sequence)
    best_fitness = fitness(best_assignment, best_sequence)
    elite = []

    def offer(assignment, sequence, known=None):
        nonlocal best_assignment, best_sequence, best_fitness, elite
        value = fitness(assignment, sequence) if known is None else known
        if better(value, best_fitness):
            best_fitness = value
            best_assignment = list(assignment)
            best_sequence = list(sequence)

        signature = (tuple(assignment), tuple(sequence))
        for old_value, old_assignment, old_sequence in elite:
            if (tuple(old_assignment), tuple(old_sequence)) == signature:
                return value
        elite.append((value, list(assignment), list(sequence)))
        elite.sort(key=lambda item: item[0])
        if len(elite) > 16:
            elite.pop()
        return value

    offer(base_assignment, base_sequence, best_fitness)

    seed_value = n * 1000003 + machine_count * 9176
    for job in jobs:
        seed_value = (
            seed_value * 1000003
            + int((17.0 * p1[job] + 31.0 * p2[job] + 13.0 * p3[job]) * 1000.0)
        ) & 0xFFFFFFFF
    rng = random.Random(seed_value)

    def greedy_sequence(assignment, tail_weight, setup_weight, jitter=0.0):
        total_load, _ = assignment_data(assignment)
        used = [0.0] * machine_count
        remaining = set(jobs)
        sequence = []
        server_clock = 0.0
        while remaining:
            ready = []
            earliest = None
            for job in remaining:
                release = used[assignment[job]] + p1[job]
                if earliest is None or release < earliest:
                    earliest = release
                if release <= server_clock + eps:
                    ready.append(job)
            choices = ready if ready else list(remaining)
            if ready:
                job = max(
                    choices,
                    key=lambda value: (
                        tail_weight * p3[value]
                        + setup_weight * p2[value]
                        + jitter * rng.random(),
                        -p2[value],
                        -value,
                    ),
                )
            else:
                job = min(
                    choices,
                    key=lambda value: (
                        used[assignment[value]] + p1[value],
                        -tail_weight * p3[value],
                        value,
                    ),
                )
            machine = assignment[job]
            used[machine] += p1[job]
            server_clock = max(server_clock, used[machine]) + p2[job]
            sequence.append(job)
            remaining.remove(job)
        return sequence

    def beam_sequence(assignment, width, stop_time):
        if machine_count == 1 or not is_proper(assignment):
            return None
        total_load, barrier = assignment_data(assignment)
        full_mask = (1 << n) - 1
        # rank, mask, clock, peak, completion_sum, loads, sequence
        states = [(barrier, 0, 0.0, barrier, 0.0,
                   tuple([0.0] * machine_count), tuple())]
        checks = 0
        for depth in range(n):
            candidates = []
            for _, mask, clock, peak, completion_sum, loads, sequence in states:
                unseen = full_mask ^ mask
                while unseen:
                    bit = unseen & -unseen
                    job = bit.bit_length() - 1
                    unseen ^= bit
                    machine = assignment[job]
                    new_load = loads[machine] + p1[job]
                    new_clock = max(clock, new_load) + p2[job]
                    completion = max(new_clock, total_load[machine]) + p3[job]
                    new_peak = max(peak, completion)
                    new_sum = completion_sum + completion
                    new_loads = list(loads)
                    new_loads[machine] = new_load
                    remaining_setup = 0.0
                    minimum_tail = None
                    rest = unseen
                    while rest:
                        rest_bit = rest & -rest
                        other = rest_bit.bit_length() - 1
                        rest ^= rest_bit
                        remaining_setup += p2[other]
                        if minimum_tail is None or p3[other] < minimum_tail:
                            minimum_tail = p3[other]
                    finish_bound = new_clock
                    if minimum_tail is not None:
                        finish_bound += remaining_setup + minimum_tail
                    lower = max(new_peak, barrier, finish_bound)
                    rank = lower + 0.0005 * new_sum + 0.0001 * new_clock
                    candidates.append((
                        rank, mask | bit, new_clock, new_peak, new_sum,
                        tuple(new_loads), sequence + (job,),
                    ))
                    checks += 1
                    if checks & 511 == 0 and time.time() >= stop_time:
                        return None
            if not candidates:
                return None
            states = heapq.nsmallest(width, candidates, key=lambda item: item[0])
            if time.time() >= stop_time:
                return None
        return list(min(states, key=lambda item: (item[3], item[4], item[2]))[6])

    def exact_sequence_dp(assignment, stop_time):
        """Pareto DP for small proper-color instances."""
        if n > 15 or machine_count == 1 or not is_proper(assignment):
            return None
        total_load, barrier = assignment_data(assignment)
        # mask -> [(server_clock, peak, completion_sum, loads, sequence)]
        layer = {0: [(0.0, barrier, 0.0,
                      tuple([0.0] * machine_count), tuple())]}
        operations = 0
        for _ in range(n):
            next_layer = {}
            for mask, labels in layer.items():
                for clock, peak, completion_sum, loads, sequence in labels:
                    for job in jobs:
                        bit = 1 << job
                        if mask & bit:
                            continue
                        machine = assignment[job]
                        new_load = loads[machine] + p1[job]
                        new_clock = max(clock, new_load) + p2[job]
                        completion = max(new_clock, total_load[machine]) + p3[job]
                        new_peak = max(peak, completion)
                        new_sum = completion_sum + completion
                        new_mask = mask | bit
                        new_loads = list(loads)
                        new_loads[machine] = new_load
                        new_label = (new_clock, new_peak, new_sum,
                                     tuple(new_loads), sequence + (job,))
                        bucket = next_layer.setdefault(new_mask, [])
                        dominated = False
                        kept = []
                        for old in bucket:
                            if (old[0] <= new_clock + eps
                                    and old[1] <= new_peak + eps
                                    and old[2] <= new_sum + eps):
                                dominated = True
                                break
                            if not (new_clock <= old[0] + eps
                                    and new_peak <= old[1] + eps
                                    and new_sum <= old[2] + eps):
                                kept.append(old)
                        if not dominated:
                            kept.append(new_label)
                            if len(kept) > 40:
                                kept.sort(key=lambda value: (value[1], value[0], value[2]))
                                kept = kept[:40]
                            next_layer[new_mask] = kept
                        operations += 1
                        if operations & 1023 == 0 and time.time() >= stop_time:
                            return None
            layer = next_layer
        labels = layer.get((1 << n) - 1, [])
        if not labels:
            return None
        return list(min(labels, key=lambda value: (value[1], value[2], value[0]))[4])

    def sequence_descent(assignment, sequence, stop_time, do_swaps=True):
        current = list(sequence)
        data = assignment_data(assignment) if is_proper(assignment) else None
        current_value = fitness(assignment, current, data)
        while time.time() < stop_time:
            chosen = None
            chosen_value = current_value
            order = list(range(n))
            rng.shuffle(order)
            for old_position in order:
                if time.time() >= stop_time:
                    break
                job = current[old_position]
                shortened = current[:old_position] + current[old_position + 1:]
                start = rng.randrange(n) if n > 1 else 0
                for step in range(n):
                    new_position = (start + step) % n
                    if new_position == old_position:
                        continue
                    trial = shortened[:new_position] + [job] + shortened[new_position:]
                    value = fitness(assignment, trial, data)
                    if better(value, chosen_value):
                        chosen = trial
                        chosen_value = value
                if time.time() >= stop_time:
                    break
            if chosen is not None:
                current = chosen
                current_value = chosen_value
                offer(assignment, current, current_value)
                continue

            if do_swaps and n > 1:
                for left in range(n - 1):
                    if time.time() >= stop_time:
                        break
                    for right in range(left + 1, n):
                        trial = list(current)
                        trial[left], trial[right] = trial[right], trial[left]
                        value = fitness(assignment, trial, data)
                        if better(value, chosen_value):
                            chosen = trial
                            chosen_value = value
                    if time.time() >= stop_time:
                        break
            if chosen is None:
                break
            current = chosen
            current_value = chosen_value
            offer(assignment, current, current_value)
        return current, current_value

    def assignment_descent(assignment, sequence, stop_time):
        if machine_count <= 2:
            return list(assignment), fitness(assignment, sequence)
        current = list(assignment)
        current_value = fitness(current, sequence)
        while time.time() < stop_time:
            chosen = None
            chosen_value = current_value
            order = list(jobs)
            rng.shuffle(order)
            for job in order:
                old_machine = current[job]
                for machine in range(machine_count):
                    if time.time() >= stop_time:
                        break
                    if machine == old_machine:
                        continue
                    if job > 0 and current[job - 1] == machine:
                        continue
                    if job + 1 < n and current[job + 1] == machine:
                        continue
                    current[job] = machine
                    value = proper_fitness(current, sequence)
                    current[job] = old_machine
                    if better(value, chosen_value):
                        chosen = (job, machine)
                        chosen_value = value
                if time.time() >= stop_time:
                    break
            if chosen is None:
                break
            job, machine = chosen
            current[job] = machine
            current_value = chosen_value
            offer(current, sequence, current_value)
        return current, current_value

    def partition_key(assignment):
        loads = [0.0] * machine_count
        tails = [0.0] * machine_count
        for job in jobs:
            machine = assignment[job]
            loads[machine] += p1[job]
            tails[machine] = max(tails[machine], p3[job])
        costs = sorted(
            (loads[machine] + tails[machine]
             for machine in range(machine_count)),
            reverse=True,
        )
        return tuple(costs) + (sum(value * value for value in costs),)

    def partition_descent(assignment, stop_time, use_swaps=True):
        """Balance sum(p1)+max(p3) while preserving the path coloring."""
        current = list(assignment)
        current_key = partition_key(current)
        while time.time() < stop_time:
            chosen = None
            chosen_key = current_key
            order = list(jobs)
            rng.shuffle(order)

            for job in order:
                old_machine = current[job]
                start = rng.randrange(machine_count)
                for step in range(machine_count):
                    if time.time() >= stop_time:
                        break
                    machine = (start + step) % machine_count
                    if machine == old_machine:
                        continue
                    if job > 0 and current[job - 1] == machine:
                        continue
                    if job + 1 < n and current[job + 1] == machine:
                        continue
                    current[job] = machine
                    key = partition_key(current)
                    current[job] = old_machine
                    if key < chosen_key:
                        chosen_key = key
                        chosen = (0, job, machine)
                if time.time() >= stop_time:
                    break

            if chosen is None and use_swaps:
                for first_index in range(n - 1):
                    if time.time() >= stop_time:
                        break
                    first = order[first_index]
                    for second_index in range(first_index + 1, n):
                        second = order[second_index]
                        first_machine = current[first]
                        second_machine = current[second]
                        if first_machine == second_machine:
                            continue
                        current[first], current[second] = second_machine, first_machine
                        valid = True
                        for edge in (first, first + 1, second, second + 1):
                            if 0 < edge < n and current[edge] == current[edge - 1]:
                                valid = False
                                break
                        if valid:
                            key = partition_key(current)
                            if key < chosen_key:
                                chosen_key = key
                                chosen = (1, first, second)
                        current[first], current[second] = first_machine, second_machine
                    if time.time() >= stop_time:
                        break

            if chosen is None:
                break
            if chosen[0] == 0:
                current[chosen[1]] = chosen[2]
            else:
                first, second = chosen[1], chosen[2]
                current[first], current[second] = current[second], current[first]
            current_key = chosen_key
        return current, current_key

    def fixed_release_sequence(assignment):
        """Schrage seed using natural-order initialization release dates."""
        loads = [0.0] * machine_count
        release = [0.0] * n
        for job in jobs:
            machine = assignment[job]
            loads[machine] += p1[job]
            release[job] = loads[machine]
        pending = sorted(jobs, key=lambda job: (release[job], job))
        available = []
        sequence = []
        cursor = 0
        clock = 0.0
        while cursor < n or available:
            if not available and cursor < n:
                clock = max(clock, release[pending[cursor]])
            while cursor < n and release[pending[cursor]] <= clock + eps:
                job = pending[cursor]
                heapq.heappush(available, (-p3[job], p2[job], job))
                cursor += 1
            if available:
                job = heapq.heappop(available)[2]
                sequence.append(job)
                clock += p2[job]
        return sequence

    def carlier_single_machine(stop_time):
        """Exact branch-and-bound specialization for the one-machine case."""
        release_original = [0.0] * n
        clock = 0.0
        for job in jobs:
            clock += p1[job]
            release_original[job] = clock
        tail_original = [0.0] * n
        tail = 0.0
        for job in range(n - 1, -1, -1):
            tail += p3[job]
            tail_original[job] = tail

        release = list(release_original)
        delivery = list(tail_original)

        def original_value(sequence):
            server = 0.0
            value = 0.0
            for job in sequence:
                server = max(server, release_original[job]) + p2[job]
                value = max(value, server + tail_original[job])
            return value

        def schrage_nonpreemptive():
            pending = sorted(jobs, key=lambda job: (release[job], job))
            available = []
            sequence = []
            completions = []
            cursor = 0
            server = 0.0
            value = 0.0
            while cursor < n or available:
                if not available and cursor < n:
                    server = max(server, release[pending[cursor]])
                while cursor < n and release[pending[cursor]] <= server + eps:
                    job = pending[cursor]
                    heapq.heappush(available, (-delivery[job], p2[job], job))
                    cursor += 1
                if available:
                    job = heapq.heappop(available)[2]
                    server += p2[job]
                    sequence.append(job)
                    completions.append(server)
                    value = max(value, server + delivery[job])
            return value, sequence, completions

        def schrage_preemptive():
            pending = sorted(jobs, key=lambda job: (release[job], job))
            remaining = list(p2)
            available = []
            cursor = 0
            server = release[pending[0]] if pending else 0.0
            value = 0.0
            running = None
            while cursor < n or available:
                while cursor < n and release[pending[cursor]] <= server + eps:
                    job = pending[cursor]
                    heapq.heappush(available, (-delivery[job], job))
                    cursor += 1
                    if running is not None and delivery[job] > delivery[running] + eps:
                        remaining[running] = server - release[job]
                        server = release[job]
                        if remaining[running] > eps:
                            heapq.heappush(
                                available, (-delivery[running], running)
                            )
                if not available:
                    if cursor < n:
                        server = max(server, release[pending[cursor]])
                    continue
                running = heapq.heappop(available)[1]
                server += remaining[running]
                value = max(value, server + delivery[running])
            return value

        root_value, root_sequence, _ = schrage_nonpreemptive()
        best_value = original_value(root_sequence)
        chosen_sequence = list(root_sequence)
        nodes = 0

        def search():
            nonlocal best_value, chosen_sequence, nodes
            nodes += 1
            if nodes & 31 == 0 and time.time() >= stop_time:
                return
            value, sequence, completions = schrage_nonpreemptive()
            candidate = original_value(sequence)
            if candidate < best_value - eps:
                best_value = candidate
                chosen_sequence = list(sequence)

            critical_end = -1
            for index, job in enumerate(sequence):
                if abs(completions[index] + delivery[job] - value) <= 1e-7:
                    critical_end = index
            if critical_end < 0:
                return

            critical_start = -1
            processing = 0.0
            end_job = sequence[critical_end]
            for index in range(critical_end, -1, -1):
                processing += p2[sequence[index]]
                if abs(
                    release[sequence[index]] + processing
                    + delivery[end_job] - value
                ) <= 1e-7:
                    critical_start = index
            if critical_start < 0:
                return

            pivot_index = -1
            for index in range(critical_start, critical_end):
                if delivery[sequence[index]] < delivery[end_job] - eps:
                    pivot_index = index
            if pivot_index < 0:
                return

            pivot = sequence[pivot_index]
            block = sequence[pivot_index + 1:critical_end + 1]
            block_release = min(release[job] for job in block)
            block_delivery = min(delivery[job] for job in block)
            block_processing = sum(p2[job] for job in block)
            h_block = block_release + block_processing + block_delivery

            old_release = release[pivot]
            release[pivot] = max(release[pivot], block_release + block_processing)
            union_h = (
                min(block_release, release[pivot])
                + block_processing + p2[pivot]
                + min(block_delivery, delivery[pivot])
            )
            lower_bound = max(schrage_preemptive(), h_block, union_h)
            if lower_bound < best_value - eps and time.time() < stop_time:
                search()
            release[pivot] = old_release

            old_delivery = delivery[pivot]
            delivery[pivot] = max(
                delivery[pivot], block_delivery + block_processing
            )
            union_h = (
                min(block_release, release[pivot])
                + block_processing + p2[pivot]
                + min(block_delivery, delivery[pivot])
            )
            lower_bound = max(schrage_preemptive(), h_block, union_h)
            if lower_bound < best_value - eps and time.time() < stop_time:
                search()
            delivery[pivot] = old_delivery

        if root_value > 0.0 and time.time() < stop_time:
            search()
        return chosen_sequence

    try:
        assignments = []
        for mode in range(3):
            for offset in range(min(machine_count, 3)):
                assignments.append(balanced_assignment(mode, offset))
        for mode in range(3):
            for offset in range(min(machine_count, 2)):
                assignments.append(colored_lpt(mode, offset))
        assignments.append([job % machine_count for job in jobs])

        unique_assignments = []
        seen_assignments = set()
        for assignment in assignments:
            key = tuple(assignment)
            if key not in seen_assignments:
                seen_assignments.add(key)
                unique_assignments.append(assignment)

        static_sequences = [
            list(jobs),
            list(reversed(jobs)),
            sorted(jobs, key=lambda job: (-p3[job], p2[job], job)),
            sorted(jobs, key=lambda job: (p1[job], -p3[job], job)),
            sorted(jobs, key=lambda job: (p2[job], -p3[job], job)),
            sorted(jobs, key=lambda job: (p1[job] + p2[job], -p3[job], job)),
        ]

        for assignment in unique_assignments:
            if time.time() >= deadline:
                break
            candidates = list(static_sequences)
            candidates.append(fixed_release_sequence(assignment))
            for tail_weight, setup_weight in ((1.0, 0.0), (1.0, 0.25),
                                               (2.0, -0.1), (0.5, 0.5)):
                candidates.append(greedy_sequence(
                    assignment, tail_weight, setup_weight
                ))
            for sequence in candidates:
                offer(assignment, sequence)

        exact_stop = min(deadline, t0 + max(0.0, float(time_limit_s)) * 0.14)
        if machine_count == 1 and time.time() < deadline:
            carlier_stop = min(
                deadline,
                t0 + max(0.0, float(time_limit_s)) * 0.42,
            )
            sequence = carlier_single_machine(carlier_stop)
            if sequence is not None and len(sequence) == n:
                offer([0] * n, sequence)
        if n <= 15 and time.time() < exact_stop:
            exact = exact_sequence_dp(best_assignment, exact_stop)
            if exact is not None:
                offer(best_assignment, exact)

        # When the primary-machine barrier is binding, this dedicated partition
        # neighborhood is much more useful than spending the budget on server
        # permutations.  Keep its pool independent of current sequence quality.
        partition_stop = min(deadline, t0 + max(0.0, float(time_limit_s)) * 0.37)
        partition_pool = []
        if machine_count > 2:
            for assignment in unique_assignments:
                if time.time() >= partition_stop:
                    break
                slice_end = min(
                    partition_stop,
                    time.time() + max(0.015, (partition_stop - time.time()) / 2.0),
                )
                improved, key = partition_descent(
                    assignment, slice_end, use_swaps=True
                )
                partition_pool.append((key, improved))
            partition_pool.sort(key=lambda item: item[0])

            restart = 0
            while time.time() < partition_stop and partition_pool:
                parent = partition_pool[rng.randrange(min(5, len(partition_pool)))][1]
                assignment = list(parent)
                strength = 2 + restart % max(2, min(8, n // 7 + 1))
                for _ in range(strength):
                    job = rng.randrange(n)
                    choices = [machine for machine in range(machine_count)
                               if machine != assignment[job]
                               and (job == 0 or machine != assignment[job - 1])
                               and (job + 1 == n or machine != assignment[job + 1])]
                    if choices:
                        assignment[job] = rng.choice(choices)
                local_end = min(
                    partition_stop,
                    time.time() + max(0.01, (partition_stop - time.time()) * 0.22),
                )
                assignment, key = partition_descent(
                    assignment, local_end, use_swaps=(restart % 2 == 0)
                )
                partition_pool.append((key, assignment))
                partition_pool.sort(key=lambda item: item[0])
                distinct = []
                seen = set()
                for item in partition_pool:
                    signature = tuple(item[1])
                    if signature not in seen:
                        distinct.append(item)
                        seen.add(signature)
                    if len(distinct) >= 12:
                        break
                partition_pool = distinct
                restart += 1

            for _, assignment in partition_pool[:8]:
                if time.time() >= deadline:
                    break
                sequences = [
                    greedy_sequence(assignment, 1.0, 0.0),
                    greedy_sequence(assignment, 1.0, 0.25),
                    fixed_release_sequence(assignment),
                    best_sequence,
                ]
                for sequence in sequences:
                    offer(assignment, sequence)

        beam_stop = min(deadline, t0 + max(0.0, float(time_limit_s)) * 0.46)
        if n > 15 and machine_count > 1 and time.time() < beam_stop:
            beam_assignments = [item[1] for item in elite[:min(4, len(elite))]]
            width = 70 if n <= 80 else 35
            for assignment in beam_assignments:
                if time.time() >= beam_stop:
                    break
                sequence = beam_sequence(assignment, width, beam_stop)
                if sequence is not None:
                    offer(assignment, sequence)

        # Intensify each structurally distinct promising construction.
        intensify_stop = min(deadline, t0 + max(0.0, float(time_limit_s)) * 0.66)
        elite_snapshot = list(elite[:8])
        for _, assignment, sequence in elite_snapshot:
            if time.time() >= intensify_stop:
                break
            local_slice = min(
                intensify_stop,
                time.time() + max(0.015, (intensify_stop - time.time()) / 2.0),
            )
            sequence, _ = sequence_descent(
                assignment, sequence, local_slice, do_swaps=True
            )
            if time.time() < intensify_stop and machine_count > 2:
                assignment, _ = assignment_descent(
                    assignment, sequence, intensify_stop
                )
                offer(assignment, sequence)

        # Cross local-optimum barriers with a short joint annealing phase.  It
        # starts from the strongest incumbent, so even an interrupted phase is
        # harmless, but it may temporarily worsen either the coloring or order.
        anneal_stop = min(deadline, t0 + max(0.0, float(time_limit_s)) * 0.78)
        if time.time() < anneal_stop:
            assignment = list(best_assignment)
            sequence = list(best_sequence)
            current_value = fitness(assignment, sequence)
            anneal_best = current_value
            temperature = max(0.08, current_value[0] * 0.012)
            initial_temperature = temperature
            iteration = 0
            stagnant = 0
            while time.time() < anneal_stop:
                trial_assignment = list(assignment)
                trial_sequence = list(sequence)
                move = rng.random()
                if move < 0.48 and n > 1:
                    old_position = rng.randrange(n)
                    job = trial_sequence.pop(old_position)
                    trial_sequence.insert(rng.randrange(n), job)
                elif move < 0.70 and n > 1:
                    left, right = rng.sample(range(n), 2)
                    trial_sequence[left], trial_sequence[right] = (
                        trial_sequence[right], trial_sequence[left]
                    )
                elif move < 0.82 and n > 3:
                    left, right = sorted(rng.sample(range(n), 2))
                    if right - left > 1:
                        block = trial_sequence[left:right]
                        del trial_sequence[left:right]
                        position = rng.randrange(len(trial_sequence) + 1)
                        trial_sequence[position:position] = block
                elif machine_count > 2:
                    job = rng.randrange(n)
                    choices = [machine for machine in range(machine_count)
                               if machine != trial_assignment[job]
                               and (job == 0 or machine != trial_assignment[job - 1])
                               and (job + 1 == n or machine != trial_assignment[job + 1])]
                    if choices:
                        trial_assignment[job] = rng.choice(choices)
                elif n > 1:
                    left, right = rng.sample(range(n), 2)
                    trial_sequence[left], trial_sequence[right] = (
                        trial_sequence[right], trial_sequence[left]
                    )

                trial_value = fitness(trial_assignment, trial_sequence)
                delta = trial_value[0] - current_value[0]
                accept = better(trial_value, current_value)
                if not accept and delta <= 0.0:
                    accept = True
                if not accept and temperature > 1e-12:
                    accept = rng.random() < math.exp(-delta / temperature)
                if accept:
                    assignment = trial_assignment
                    sequence = trial_sequence
                    current_value = trial_value
                    if better(current_value, anneal_best):
                        anneal_best = current_value
                        offer(assignment, sequence, current_value)
                        stagnant = 0
                    else:
                        stagnant += 1
                else:
                    stagnant += 1

                temperature *= 0.9992
                iteration += 1
                if temperature < initial_temperature * 0.025 or stagnant > 4500:
                    if elite:
                        parent = elite[rng.randrange(min(5, len(elite)))]
                        assignment = list(parent[1])
                        sequence = list(parent[2])
                        current_value = parent[0]
                    temperature = initial_temperature * (0.55 + 0.45 * rng.random())
                    stagnant = 0
                if iteration & 255 == 0 and time.time() >= anneal_stop:
                    break

        # Iterated local search spends every remaining wall-clock slice.
        restart = 0
        while time.time() < deadline:
            if elite:
                limit = min(8, len(elite))
                parent = elite[rng.randrange(limit)]
                assignment = list(parent[1])
                sequence = list(parent[2])
            else:
                assignment = list(best_assignment)
                sequence = list(best_sequence)

            strength = 1 + (restart % max(2, min(7, n // 6 + 1)))
            for _ in range(strength):
                if n > 1:
                    move = rng.randrange(3)
                    if move == 0:
                        old = rng.randrange(n)
                        job = sequence.pop(old)
                        sequence.insert(rng.randrange(n), job)
                    elif move == 1:
                        left, right = rng.sample(range(n), 2)
                        sequence[left], sequence[right] = sequence[right], sequence[left]
                    else:
                        left, right = sorted(rng.sample(range(n), 2))
                        sequence[left:right + 1] = reversed(sequence[left:right + 1])
                if machine_count > 2 and rng.random() < 0.55:
                    job = rng.randrange(n)
                    choices = [machine for machine in range(machine_count)
                               if machine != assignment[job]
                               and (job == 0 or machine != assignment[job - 1])
                               and (job + 1 == n or machine != assignment[job + 1])]
                    if choices:
                        assignment[job] = rng.choice(choices)

            offer(assignment, sequence)
            remaining = deadline - time.time()
            if remaining <= 0.0:
                break
            local_stop = min(deadline, time.time() + max(0.01, min(0.22, remaining * 0.32)))
            sequence, _ = sequence_descent(
                assignment, sequence, local_stop,
                do_swaps=(restart % 3 == 0),
            )
            if machine_count > 2 and time.time() < local_stop:
                assignment, _ = assignment_descent(
                    assignment, sequence, local_stop
                )
            offer(assignment, sequence)
            restart += 1
    except Exception:
        # The complete incumbent installed above is always safe to return.
        pass

    return materialize(best_assignment, best_sequence)
