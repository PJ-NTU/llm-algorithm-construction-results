# Library slot 9/10  id=alg_0ee897cd3cd5
# score=312.512275143856  feasible_rate=1.0
# score_vec=[334.6, 808.5319370720604, 165.276956771257, 164.93542285602592, 206.43467106241812, 198.95476367679845, 223.40000000000003, 356.4559999999999, 329.876, 336.65700000000004]

import heapq
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    deadline = t0 + max(0.0, float(time_limit_s)) * 0.85

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

    # A valid problem has at least one machine.  Keeping this guard here also
    # makes all random machine selections below safe.
    if machine_count <= 0:
        machine_count = 1

    def release_times(assignment):
        loads = [0.0] * machine_count
        releases = [0.0] * n
        for job in range(n):
            machine = assignment[job]
            loads[machine] += p1[job]
            releases[job] = loads[machine]
        return releases, loads

    def evaluate(assignment, sequence):
        releases, loads = release_times(assignment)
        setup_completion = [0.0] * n
        server_time = 0.0
        for job in sequence:
            start = server_time
            if releases[job] > start:
                start = releases[job]
            server_time = start + p2[job]
            setup_completion[job] = server_time

        value = 0.0
        critical = 0
        main_free = loads[:]
        for job in range(n):
            machine = assignment[job]
            start = setup_completion[job]
            if main_free[machine] > start:
                start = main_free[machine]
            completion = start + p3[job]
            main_free[machine] = completion
            if completion > value:
                value = completion
                critical = job
        return value, critical

    def materialize(assignment, sequence):
        s1 = [0.0] * n
        s2 = [0.0] * n
        s3 = [0.0] * n
        loads = [0.0] * machine_count

        for job in range(n):
            machine = assignment[job]
            s1[job] = loads[machine]
            loads[machine] += p1[job]

        server_time = 0.0
        for job in sequence:
            start = server_time
            release = s1[job] + p1[job]
            if release > start:
                start = release
            s2[job] = start
            server_time = start + p2[job]

        makespan = 0.0
        main_free = loads[:]
        for job in range(n):
            machine = assignment[job]
            start = max(s2[job] + p2[job], main_free[machine])
            s3[job] = start
            completion = start + p3[job]
            main_free[machine] = completion
            if completion > makespan:
                makespan = completion

        return {
            "machine_assignment": {
                "job_%d" % job: int(assignment[job]) for job in range(n)
            },
            "server_sequence": ["job_%d" % job for job in sequence],
            "start_times": {
                "job_%d" % job: [s1[job], s2[job], s3[job]]
                for job in range(n)
            },
            "makespan": float(makespan),
        }

    # The fallback is stored before any search.  It is complete and feasible
    # even when the caller grants virtually no search time.
    fallback_assignment = [job % machine_count for job in range(n)]
    fallback_sequence = list(range(n))
    best_assignment = fallback_assignment[:]
    best_sequence = fallback_sequence[:]
    best_value, best_critical = evaluate(best_assignment, best_sequence)

    # Use an instance-derived local seed: reproducible, but different exams do
    # not follow identical search trajectories.
    seed = n * 1000003 + machine_count * 9176
    for job in range(n):
        seed ^= int((p1[job] * 17.0 + p2[job] * 31.0 + p3[job] * 43.0) * 1000.0)
        seed = (seed * 1103515245 + 12345) & 0xFFFFFFFF
    rng = random.Random(seed)

    def schrage_sequence(assignment, noise=0.0):
        releases, unused_loads = release_times(assignment)
        unreleased = sorted(range(n), key=lambda job: (releases[job], job))
        available = []
        sequence = []
        index = 0
        server_time = 0.0
        scale = max(p3) if p3 else 1.0
        while index < n or available:
            if not available and index < n and server_time < releases[unreleased[index]]:
                server_time = releases[unreleased[index]]
            while index < n and releases[unreleased[index]] <= server_time + 1e-12:
                job = unreleased[index]
                priority = p3[job]
                if noise:
                    priority += noise * scale * rng.random()
                heapq.heappush(available, (-priority, releases[job], job))
                index += 1
            unused_priority, unused_release, job = heapq.heappop(available)
            sequence.append(job)
            server_time += p2[job]
        return sequence

    elite = []

    def add_elite(value, assignment, sequence):
        nonlocal elite
        signature = (tuple(assignment), tuple(sequence))
        for old_value, old_assignment, old_sequence in elite:
            if signature == (tuple(old_assignment), tuple(old_sequence)):
                return
        elite.append((value, assignment[:], sequence[:]))
        elite.sort(key=lambda item: item[0])
        if len(elite) > 14:
            # Retain the best ten and four structurally diverse alternatives.
            head = elite[:10]
            tail = elite[10:]
            diverse = []
            while tail and len(diverse) < 4:
                candidate = max(
                    tail,
                    key=lambda item: sum(
                        item[1][job] != head[0][1][job] for job in range(n)
                    ),
                )
                diverse.append(candidate)
                tail.remove(candidate)
            elite = head + diverse

    def consider(assignment, sequence):
        nonlocal best_value, best_assignment, best_sequence, best_critical
        value, critical = evaluate(assignment, sequence)
        add_elite(value, assignment, sequence)
        if value < best_value - 1e-10:
            best_value = value
            best_assignment = assignment[:]
            best_sequence = sequence[:]
            best_critical = critical
        return value, critical

    add_elite(best_value, best_assignment, best_sequence)

    def greedy_assignment(mode, randomized=False):
        assignment = [-1] * n
        loads = [0.0] * machine_count
        largest_tail = [0.0] * machine_count
        previous = -1
        for job in range(n):
            choices = list(range(machine_count))
            if machine_count > 1:
                choices = [machine for machine in choices if machine != previous]
            scored = []
            for machine in choices:
                new_load = loads[machine] + p1[job]
                new_tail = max(largest_tail[machine], p3[job])
                if mode == 0:
                    score = new_load
                elif mode == 1:
                    score = new_load + new_tail
                elif mode == 2:
                    score = new_load + 0.35 * new_tail + 0.08 * p2[job]
                else:
                    score = new_load + new_tail + 0.15 * releases_hint(job, machine, loads)
                if randomized:
                    score *= 0.92 + 0.16 * rng.random()
                scored.append((score, loads[machine], machine))
            machine = min(scored)[2]
            assignment[job] = machine
            loads[machine] += p1[job]
            if p3[job] > largest_tail[machine]:
                largest_tail[machine] = p3[job]
            previous = machine
        return assignment

    def releases_hint(job, machine, loads):
        # A small bias toward early release of large-tail jobs.  It is deliberately
        # subordinate to load balance, which determines the main-phase barrier.
        return (loads[machine] + p1[job]) / (1.0 + p3[job])

    def repair_adjacency(assignment):
        if machine_count <= 1:
            return assignment
        repaired = assignment[:]
        loads = [0.0] * machine_count
        tails = [0.0] * machine_count
        for job in range(n):
            machine = repaired[job]
            if job > 0 and machine == repaired[job - 1]:
                choices = [m for m in range(machine_count) if m != repaired[job - 1]]
                if job + 1 < n and repaired[job + 1] in choices and len(choices) > 1:
                    choices.remove(repaired[job + 1])
                machine = min(
                    choices,
                    key=lambda m: loads[m] + p1[job] + max(tails[m], p3[job]),
                )
                repaired[job] = machine
            loads[machine] += p1[job]
            tails[machine] = max(tails[machine], p3[job])
        return repaired

    try:
        # Seed the memetic pool with complementary colorings and server rules.
        seed_assignments = [fallback_assignment[:]]
        if machine_count > 1:
            for shift in range(min(machine_count, 6)):
                seed_assignments.append([(job + shift) % machine_count for job in range(n)])
        for mode in range(4):
            if time.time() >= deadline:
                return materialize(best_assignment, best_sequence)
            seed_assignments.append(greedy_assignment(mode, False))
            seed_assignments.append(greedy_assignment(mode, True))

        for assignment in seed_assignments:
            if time.time() >= deadline:
                return materialize(best_assignment, best_sequence)
            releases, unused_loads = release_times(assignment)
            candidates = [
                schrage_sequence(assignment),
                sorted(range(n), key=lambda job: (-p3[job], releases[job], job)),
                sorted(range(n), key=lambda job: (releases[job], -p3[job], job)),
            ]
            for sequence in candidates:
                if time.time() >= deadline:
                    return materialize(best_assignment, best_sequence)
                consider(assignment, sequence)

        current_value, current_assignment, current_sequence = elite[0]
        current_assignment = current_assignment[:]
        current_sequence = current_sequence[:]
        current_value, current_critical = evaluate(current_assignment, current_sequence)
        history_length = max(40, min(500, n * 5))
        history = [current_value] * history_length
        history_index = 0
        stagnation = 0
        restart_threshold = max(180, n * 16)

        def complementary_parent(first):
            sample = elite[: min(len(elite), 10)]
            return max(
                sample,
                key=lambda item: sum(
                    first[1][job] != item[1][job] for job in range(n)
                ) + sum(
                    first[2][pos] != item[2][pos] for pos in range(n)
                ) * 0.15,
            )

        def crossover():
            parent1 = elite[rng.randrange(min(len(elite), 6))]
            parent2 = complementary_parent(parent1)
            if n == 1:
                return parent1[1][:], parent1[2][:]

            left = rng.randrange(n)
            right = rng.randrange(left, n)
            child_assignment = parent1[1][:]
            child_assignment[left : right + 1] = parent2[1][left : right + 1]
            if rng.random() < 0.85:
                child_assignment = repair_adjacency(child_assignment)

            # Order crossover transfers a coherent server block and fills the
            # remainder in the other parent's relative order.
            child_sequence = [-1] * n
            used = set()
            for pos in range(left, right + 1):
                job = parent1[2][pos]
                child_sequence[pos] = job
                used.add(job)
            fill = [job for job in parent2[2] if job not in used]
            fill_index = 0
            for pos in range(n):
                if child_sequence[pos] < 0:
                    child_sequence[pos] = fill[fill_index]
                    fill_index += 1

            schrage = schrage_sequence(child_assignment, 0.04)
            if evaluate(child_assignment, schrage)[0] < evaluate(child_assignment, child_sequence)[0]:
                child_sequence = schrage
            return child_assignment, child_sequence

        while time.time() < deadline:
            # Stagnating late-acceptance walks are restarted by B-style genetic
            # recombination of two complementary, already-improved elites.
            if stagnation >= restart_threshold or rng.random() < 0.003:
                new_assignment, new_sequence = crossover()
                new_value, new_critical = evaluate(new_assignment, new_sequence)
                add_elite(new_value, new_assignment, new_sequence)
                current_assignment = new_assignment
                current_sequence = new_sequence
                current_value = new_value
                current_critical = new_critical
                history = [current_value] * history_length
                history_index = 0
                stagnation = 0
                if current_value < best_value - 1e-10:
                    best_value = current_value
                    best_assignment = current_assignment[:]
                    best_sequence = current_sequence[:]
                    best_critical = current_critical
                continue

            if time.time() >= deadline:
                break

            new_assignment = current_assignment[:]
            new_sequence = current_sequence[:]
            move = rng.random()

            if move < 0.30 and machine_count > 1:
                # Machine-color neighborhoods: relocation, exchange, and a
                # segment color swap.  The latter makes useful M=2 moves while
                # creating at most two adjacency conflicts.
                subtype = rng.random()
                if subtype < 0.48:
                    job = current_critical if rng.random() < 0.28 else rng.randrange(n)
                    choices = list(range(machine_count))
                    if machine_count > 2 and rng.random() < 0.85:
                        choices = [
                            machine
                            for machine in choices
                            if (job == 0 or machine != new_assignment[job - 1])
                            and (job + 1 == n or machine != new_assignment[job + 1])
                        ]
                    if choices:
                        new_assignment[job] = choices[rng.randrange(len(choices))]
                elif subtype < 0.72 and n > 1:
                    first = rng.randrange(n)
                    second = rng.randrange(n - 1)
                    if second >= first:
                        second += 1
                    new_assignment[first], new_assignment[second] = (
                        new_assignment[second],
                        new_assignment[first],
                    )
                else:
                    left = rng.randrange(n)
                    right = rng.randrange(left, n)
                    color1 = rng.randrange(machine_count)
                    color2 = rng.randrange(machine_count - 1)
                    if color2 >= color1:
                        color2 += 1
                    for job in range(left, right + 1):
                        if new_assignment[job] == color1:
                            new_assignment[job] = color2
                        elif new_assignment[job] == color2:
                            new_assignment[job] = color1

                if time.time() >= deadline:
                    break
                # Couple the assignment move to the server rather than judging
                # it only under a stale permutation.
                if rng.random() < 0.45:
                    rebuilt = schrage_sequence(new_assignment, 0.02)
                    if evaluate(new_assignment, rebuilt)[0] < evaluate(new_assignment, new_sequence)[0]:
                        new_sequence = rebuilt
            else:
                # Server VNS: insertion is the workhorse; swaps and short block
                # reversals give it escape routes.  Critical jobs are often
                # pulled earlier, which protects their large p3 tail.
                if n > 1:
                    subtype = rng.random()
                    if subtype < 0.57:
                        job = current_critical if rng.random() < 0.34 else new_sequence[rng.randrange(n)]
                        old_pos = new_sequence.index(job)
                        new_sequence.pop(old_pos)
                        if job == current_critical and rng.random() < 0.65:
                            new_pos = rng.randrange(old_pos + 1) if old_pos > 0 else 0
                        else:
                            new_pos = rng.randrange(n)
                        new_sequence.insert(new_pos, job)
                    elif subtype < 0.84:
                        first = rng.randrange(n)
                        second = rng.randrange(n - 1)
                        if second >= first:
                            second += 1
                        new_sequence[first], new_sequence[second] = (
                            new_sequence[second],
                            new_sequence[first],
                        )
                    else:
                        left = rng.randrange(n - 1)
                        max_width = min(n - left, 8)
                        right = left + 1 + rng.randrange(max_width - 1)
                        new_sequence[left : right + 1] = reversed(new_sequence[left : right + 1])

            if time.time() >= deadline:
                break
            new_value, new_critical = evaluate(new_assignment, new_sequence)
            threshold = history[history_index]
            accepted = new_value <= current_value + 1e-10 or new_value <= threshold + 1e-10
            if accepted:
                current_assignment = new_assignment
                current_sequence = new_sequence
                current_value = new_value
                current_critical = new_critical

            history[history_index] = current_value
            history_index = (history_index + 1) % history_length

            if current_value < best_value - 1e-10:
                best_value = current_value
                best_assignment = current_assignment[:]
                best_sequence = current_sequence[:]
                best_critical = current_critical
                add_elite(best_value, best_assignment, best_sequence)
                stagnation = 0
            else:
                stagnation += 1
                if accepted and stagnation % max(25, n) == 0:
                    add_elite(current_value, current_assignment, current_sequence)

    except Exception:
        # The pre-built incumbent is always safe.  Search failures must never
        # turn an otherwise valid exam instance into a missing solution.
        pass

    return materialize(best_assignment, best_sequence)
