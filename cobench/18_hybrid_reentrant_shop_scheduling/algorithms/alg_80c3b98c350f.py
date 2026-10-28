# Library slot 4/10  id=alg_80c3b98c350f
# score=309.56753409855486  feasible_rate=1.0
# score_vec=[334.59999999999997, 808.5317313947484, 160.60560882458412, 161.277105938526, 205.54952062584212, 197.78437420184864, 223.39999999999998, 343.70899999999983, 327.3869999999999, 332.83099999999996]

import heapq
import random
import time


def _safe_solution(instance):
    """Fast, deliberately conservative schedule used as the hard fallback."""
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

    assignment = [job % machine_count for job in range(n)]
    machine_free = [0.0] * machine_count
    s1 = [0.0] * n
    for job in range(n):
        machine = assignment[job]
        s1[job] = machine_free[machine]
        machine_free[machine] += p1[job]

    s2 = [0.0] * n
    server_free = 0.0
    for job in range(n):
        s2[job] = max(server_free, s1[job] + p1[job])
        server_free = s2[job] + p2[job]

    s3 = [0.0] * n
    main_free = [0.0] * machine_count
    for job in range(n):
        machine = assignment[job]
        s3[job] = max(
            s2[job] + p2[job], machine_free[machine], main_free[machine]
        )
        main_free[machine] = s3[job] + p3[job]

    makespan = max(s3[job] + p3[job] for job in range(n))
    return {
        "machine_assignment": {
            "job_{}".format(job): assignment[job] for job in range(n)
        },
        "server_sequence": ["job_{}".format(job) for job in range(n)],
        "start_times": {
            "job_{}".format(job): [s1[job], s2[job], s3[job]]
            for job in range(n)
        },
        "makespan": makespan,
    }


def _search(instance, time_limit_s):
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

    # Keep the legacy init_order component in the search state so the original
    # neighborhood structure stays intact.  It must not affect initialization
    # timing: that stage has a fixed natural order on each assigned machine.
    def decode(assignment, sequence, init_order, keep_times=False):
        s1 = [0.0] * n
        init_done = [0.0] * machine_count
        for job in range(n):
            machine = assignment[job]
            s1[job] = init_done[machine]
            init_done[machine] += p1[job]

        s2 = [0.0] * n
        server_free = 0.0
        for job in sequence:
            start = max(server_free, s1[job] + p1[job])
            s2[job] = start
            server_free = start + p2[job]

        s3 = [0.0] * n
        makespan = 0.0
        completion_sum = 0.0
        critical_job = 0
        main_free = init_done[:]
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

        # The secondary terms provide a slope on broad equal-makespan plateaus.
        fitness = (
            makespan,
            completion_sum / n + 0.03 * server_free
            + 0.002 * sum(init_done),
        )
        if keep_times:
            return fitness, critical_job, s1, s2, s3
        return fitness, critical_job

    def balanced_assignment(offset=0, allow_adjacency=False):
        loads = [0.0] * machine_count
        assignment = [0] * n
        previous = -1
        for job in range(n):
            choices = list(range(machine_count))
            if not allow_adjacency and machine_count > 1:
                if previous in choices:
                    choices.remove(previous)
            machine = min(
                choices,
                key=lambda value: (
                    loads[value] + p1[job],
                    (value - offset) % machine_count,
                ),
            )
            assignment[job] = machine
            loads[machine] += p1[job]
            previous = machine
        return assignment

    def lpt_assignment(offset=0):
        loads = [0.0] * machine_count
        assignment = [0] * n
        jobs = sorted(range(n), key=lambda job: (-p1[job], -p3[job], job))
        for job in jobs:
            machine = min(
                range(machine_count),
                key=lambda value: (
                    loads[value], (value - offset) % machine_count
                ),
            )
            assignment[job] = machine
            loads[machine] += p1[job]
        return assignment

    def tail_assignment(offset=0):
        loads = [0.0] * machine_count
        largest_tail = [0.0] * machine_count
        assignment = [0] * n
        previous = -1
        for job in range(n):
            choices = list(range(machine_count))
            if machine_count > 1 and previous in choices:
                choices.remove(previous)
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
            previous = machine
        return assignment

    def downstream_tails(assignment):
        tails = p3[:]
        for job in range(n - 2, -1, -1):
            if assignment[job] == assignment[job + 1]:
                tails[job] += tails[job + 1]
        return tails

    def initialization_times(assignment, init_order):
        completion = [0.0] * n
        machine_free = [0.0] * machine_count
        for job in range(n):
            machine = assignment[job]
            machine_free[machine] += p1[job]
            completion[job] = machine_free[machine]
        return completion

    def schrage_sequence(assignment, init_order):
        releases = initialization_times(assignment, init_order)
        tails = downstream_tails(assignment)
        unreleased = sorted(range(n), key=lambda job: (releases[job], job))
        available = []
        sequence = []
        index = 0
        current_time = unreleased and releases[unreleased[0]] or 0.0
        while len(sequence) < n:
            while index < n and releases[unreleased[index]] <= current_time + 1e-12:
                job = unreleased[index]
                heapq.heappush(available, (-tails[job], p2[job], job))
                index += 1
            if not available:
                current_time = releases[unreleased[index]]
                continue
            _, _, job = heapq.heappop(available)
            sequence.append(job)
            current_time = max(current_time, releases[job]) + p2[job]
        return sequence

    # Store a complete feasible incumbent before doing any substantial work.
    base_assignment = balanced_assignment(0)
    base_init = list(range(n))
    base_sequence = schrage_sequence(base_assignment, base_init)
    best_fitness, _ = decode(base_assignment, base_sequence, base_init)
    best_assignment = base_assignment[:]
    best_sequence = base_sequence[:]
    best_init = base_init[:]

    seed_value = n * 1000003 + machine_count * 9176
    for job in range(n):
        seed_value += int(
            (p1[job] * 17.0 + p2[job] * 31.0 + p3[job] * 13.0)
            * 1000.0
        )
    rng = random.Random(seed_value)

    # Retain several genuinely different basins for later iterated search.
    pool = []
    pool_limit = 14

    def offer(assignment, sequence, init_order, evaluated=None):
        nonlocal best_fitness, best_assignment, best_sequence, best_init
        if evaluated is None:
            fitness, _ = decode(assignment, sequence, init_order)
        else:
            fitness = evaluated
        if fitness < best_fitness:
            best_fitness = fitness
            best_assignment = assignment[:]
            best_sequence = sequence[:]
            best_init = init_order[:]

        for old in pool:
            if (old[1] == assignment and old[2] == sequence
                    and old[3] == init_order):
                return fitness
        pool.append((fitness, assignment[:], sequence[:], init_order[:]))
        pool.sort(key=lambda entry: entry[0])
        if len(pool) > pool_limit:
            pool.pop()
        return fitness

    offer(base_assignment, base_sequence, base_init, best_fitness)

    # Diverse deterministic starts: server-tail priorities, release-aware
    # Schrage schedules, and independent first-stage priorities.
    assignments = []
    for offset in range(min(machine_count, 4)):
        assignments.append(balanced_assignment(offset))
        assignments.append(tail_assignment(offset))
        assignments.append(lpt_assignment(offset))
    assignments.append(balanced_assignment(0, True))

    # When there are several machines, first optimize a rigorous assignment-side
    # bound: every job on machine k finishes no earlier than that machine's total
    # initialization load plus its own p3 tail.  Searching this cheap surrogate
    # supplies much stronger colorings than perturbing the full schedule alone.
    if machine_count > 2 and time.time() < deadline:
        partition_end = min(
            deadline,
            time.time() + max(0.0, float(time_limit_s)) * 0.10,
        )

        def floor_value(assignment):
            loads = [0.0] * machine_count
            tails = [0.0] * machine_count
            for job in range(n):
                machine = assignment[job]
                loads[machine] += p1[job]
                if p3[job] > tails[machine]:
                    tails[machine] = p3[job]
            return max(
                loads[machine] + tails[machine]
                for machine in range(machine_count)
            )

        current_assignment = balanced_assignment(0)
        current_floor = floor_value(current_assignment)
        floor_pool = [(current_floor, current_assignment[:])]
        stagnation = 0

        while time.time() < partition_end:
            candidate = current_assignment[:]
            if n > 1 and rng.random() < 0.35:
                first = rng.randrange(n)
                second = rng.randrange(n)
                candidate[first], candidate[second] = (
                    candidate[second], candidate[first]
                )
                affected = (first, second)
            else:
                job = rng.randrange(n)
                old_machine = candidate[job]
                new_machine = rng.randrange(machine_count - 1)
                if new_machine >= old_machine:
                    new_machine += 1
                candidate[job] = new_machine
                affected = (job,)

            proper = True
            for job in affected:
                if job > 0 and candidate[job] == candidate[job - 1]:
                    proper = False
                if job + 1 < n and candidate[job] == candidate[job + 1]:
                    proper = False
            if not proper:
                stagnation += 1
                continue

            candidate_floor = floor_value(candidate)
            remaining = max(0.0, partition_end - time.time())
            duration = max(1e-9, max(0.0, float(time_limit_s)) * 0.10)
            temperature = 0.05 + 2.5 * remaining / duration
            delta = candidate_floor - current_floor
            if delta <= 0.0 or rng.random() < temperature / (temperature + delta):
                current_assignment = candidate
                current_floor = candidate_floor

            if candidate_floor < floor_pool[-1][0] or len(floor_pool) < 6:
                if all(old[1] != candidate for old in floor_pool):
                    floor_pool.append((candidate_floor, candidate[:]))
                    floor_pool.sort(key=lambda entry: entry[0])
                    if len(floor_pool) > 6:
                        floor_pool.pop()
            stagnation += 1
            if stagnation >= 1200:
                current_assignment = floor_pool[
                    rng.randrange(min(3, len(floor_pool)))
                ][1][:]
                for _ in range(1 + rng.randrange(4)):
                    job = rng.randrange(n)
                    choices = [
                        machine for machine in range(machine_count)
                        if (job == 0 or machine != current_assignment[job - 1])
                        and (job + 1 == n
                             or machine != current_assignment[job + 1])
                    ]
                    if choices:
                        current_assignment[job] = choices[
                            rng.randrange(len(choices))
                        ]
                current_floor = floor_value(current_assignment)
                stagnation = 0

        assignments.extend(entry[1] for entry in floor_pool)

    init_orders = [
        list(range(n)),
        sorted(range(n), key=lambda job: (p1[job], -p3[job], job)),
        sorted(range(n), key=lambda job: (-p3[job], p1[job], job)),
        sorted(range(n), key=lambda job: (p1[job] + p2[job], -p3[job], job)),
    ]

    def relocate(values, old_position, new_position):
        changed = values[:]
        value = changed.pop(old_position)
        changed.insert(new_position, value)
        return changed

    pipeline_states = []
    for assignment in assignments:
        if time.time() >= deadline:
            break

        first_job = min(range(n), key=lambda job: (p1[job], job))
        last_job = min(
            (job for job in range(n) if job != first_job),
            key=lambda job: (p3[job], job),
            default=first_job,
        )
        forced_middle = [
            job for job in sorted(
                range(n), key=lambda job: (-p3[job], p1[job], job)
            )
            if job != first_job and job != last_job
        ]
        forced_sequence = [first_job] + forced_middle
        if last_job != first_job:
            forced_sequence.append(last_job)
        forced_fitness = offer(
            assignment, forced_sequence, forced_sequence
        )
        pipeline_states.append(
            (forced_fitness, assignment[:], forced_sequence[:])
        )

        for init_order in init_orders:
            if time.time() >= deadline:
                break
            sequence = schrage_sequence(assignment, init_order)
            offer(assignment, sequence, init_order)
            offer(
                assignment,
                sorted(
                    range(n),
                    key=lambda job: (p2[job], -downstream_tails(assignment)[job], job),
                ),
                init_order,
            )
            # Couple the first stage back to the promising server order, then
            # rebuild the server order once more (a cheap coordinate pass).
            coupled_init = sequence[:]
            coupled_sequence = schrage_sequence(assignment, coupled_init)
            offer(assignment, coupled_sequence, coupled_init)

    # Intensify the lower-bound construction without allowing its endpoint
    # choices to drift.  Recoloring repairs prefix loads; synchronized insertion
    # changes which jobs must be initialized early enough to keep the server busy.
    if machine_count > 2 and pipeline_states and time.time() < deadline:
        pipeline_states.sort(key=lambda entry: entry[0])
        pipeline_end = min(
            deadline,
            time.time() + max(0.0, float(time_limit_s)) * 0.12,
        )
        current = pipeline_states[0]
        current_fitness = current[0]
        current_assignment = current[1][:]
        current_sequence = current[2][:]
        stagnation = 0
        while time.time() < pipeline_end:
            candidate_assignment = current_assignment[:]
            candidate_sequence = current_sequence[:]
            if rng.random() < 0.68:
                if n > 1 and rng.random() < 0.30:
                    first = rng.randrange(n)
                    second = rng.randrange(n)
                    candidate_assignment[first], candidate_assignment[second] = (
                        candidate_assignment[second], candidate_assignment[first]
                    )
                    affected = (first, second)
                else:
                    job = rng.randrange(n)
                    old_machine = candidate_assignment[job]
                    new_machine = rng.randrange(machine_count - 1)
                    if new_machine >= old_machine:
                        new_machine += 1
                    candidate_assignment[job] = new_machine
                    affected = (job,)
                proper = True
                for job in affected:
                    if (job > 0 and candidate_assignment[job]
                            == candidate_assignment[job - 1]):
                        proper = False
                    if (job + 1 < n and candidate_assignment[job]
                            == candidate_assignment[job + 1]):
                        proper = False
                if not proper:
                    stagnation += 1
                    continue
            elif n > 3:
                old_position = 1 + rng.randrange(n - 2)
                new_position = 1 + rng.randrange(n - 2)
                candidate_sequence = relocate(
                    candidate_sequence, old_position, new_position
                )

            candidate_fitness, _ = decode(
                candidate_assignment,
                candidate_sequence,
                candidate_sequence,
            )
            delta = candidate_fitness[0] - current_fitness[0]
            remaining = max(0.0, pipeline_end - time.time())
            duration = max(1e-9, max(0.0, float(time_limit_s)) * 0.12)
            temperature = 0.02 + 1.2 * remaining / duration
            if (candidate_fitness < current_fitness
                    or (delta > 0.0
                        and rng.random() < temperature / (temperature + delta))):
                current_assignment = candidate_assignment
                current_sequence = candidate_sequence
                current_fitness = candidate_fitness
            if candidate_fitness < best_fitness:
                offer(
                    candidate_assignment,
                    candidate_sequence,
                    candidate_sequence,
                    candidate_fitness,
                )
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= 1000:
                seed = pipeline_states[
                    rng.randrange(min(5, len(pipeline_states)))
                ]
                current_fitness = seed[0]
                current_assignment = seed[1][:]
                current_sequence = seed[2][:]
                stagnation = 0

    def descent(assignment, sequence, init_order, full=True):
        current_fitness, _ = decode(assignment, sequence, init_order)
        improved_any = False
        keep_going = True
        while keep_going and time.time() < deadline:
            keep_going = False

            # Best insertion on the remote server.  This is the most influential
            # exact neighborhood and subsumes adjacent swaps.
            neighbor_fitness = current_fitness
            neighbor_sequence = None
            if full or n <= 20:
                positions = range(n)
            else:
                positions = [rng.randrange(n) for _ in range(min(n, 20))]
            checked = 0
            for old_position in positions:
                if time.time() >= deadline:
                    break
                if full or n <= 20:
                    targets = range(n)
                else:
                    targets = [rng.randrange(n) for _ in range(min(n, 20))]
                for new_position in targets:
                    if old_position == new_position:
                        continue
                    candidate = relocate(sequence, old_position, new_position)
                    fitness, _ = decode(assignment, candidate, init_order)
                    checked += 1
                    if fitness < neighbor_fitness:
                        neighbor_fitness = fitness
                        neighbor_sequence = candidate
                    if checked & 63 == 0 and time.time() >= deadline:
                        break
            if neighbor_sequence is not None:
                sequence = neighbor_sequence
                current_fitness = neighbor_fitness
                keep_going = True
                improved_any = True
                offer(assignment, sequence, init_order, current_fitness)
                continue

            # The analogous insertion neighborhood for first-stage priorities.
            neighbor_fitness = current_fitness
            neighbor_init = None
            if full or n <= 20:
                positions = range(n)
            else:
                positions = [rng.randrange(n) for _ in range(min(n, 20))]
            checked = 0
            for old_position in positions:
                if time.time() >= deadline:
                    break
                if full or n <= 20:
                    targets = range(n)
                else:
                    targets = [rng.randrange(n) for _ in range(min(n, 20))]
                for new_position in targets:
                    if old_position == new_position:
                        continue
                    candidate = relocate(init_order, old_position, new_position)
                    fitness, _ = decode(assignment, sequence, candidate)
                    checked += 1
                    if fitness < neighbor_fitness:
                        neighbor_fitness = fitness
                        neighbor_init = candidate
                    if checked & 63 == 0 and time.time() >= deadline:
                        break
            if neighbor_init is not None:
                init_order = neighbor_init
                current_fitness = neighbor_fitness
                keep_going = True
                improved_any = True
                offer(assignment, sequence, init_order, current_fitness)
                continue

            # Single-job recoloring directly adjusts initialization load and can
            # create or break the only main-stage precedence arcs that exist.
            neighbor_fitness = current_fitness
            neighbor_assignment = None
            for job in range(n):
                if time.time() >= deadline:
                    break
                old_machine = assignment[job]
                for machine in range(machine_count):
                    if machine == old_machine:
                        continue
                    candidate = assignment[:]
                    candidate[job] = machine
                    fitness, _ = decode(candidate, sequence, init_order)
                    if fitness < neighbor_fitness:
                        neighbor_fitness = fitness
                        neighbor_assignment = candidate
            if neighbor_assignment is not None:
                assignment = neighbor_assignment
                current_fitness = neighbor_fitness
                keep_going = True
                improved_any = True
                offer(assignment, sequence, init_order, current_fitness)
                continue

            # Pair recoloring preserves load much better on the difficult M=2
            # cases.  Sample it on large instances during secondary descents.
            neighbor_fitness = current_fitness
            neighbor_assignment = None
            if full or n <= 20:
                first_jobs = range(n)
                second_jobs = None
            else:
                first_jobs = [rng.randrange(n) for _ in range(min(n, 24))]
                second_jobs = [rng.randrange(n) for _ in range(min(n, 24))]
            checked = 0
            for first in first_jobs:
                if time.time() >= deadline:
                    break
                candidates = range(first + 1, n) if second_jobs is None else second_jobs
                for second in candidates:
                    if first == second or assignment[first] == assignment[second]:
                        continue
                    candidate = assignment[:]
                    candidate[first], candidate[second] = (
                        candidate[second], candidate[first]
                    )
                    fitness, _ = decode(candidate, sequence, init_order)
                    checked += 1
                    if fitness < neighbor_fitness:
                        neighbor_fitness = fitness
                        neighbor_assignment = candidate
                    if checked & 63 == 0 and time.time() >= deadline:
                        break
            if neighbor_assignment is not None:
                assignment = neighbor_assignment
                current_fitness = neighbor_fitness
                keep_going = True
                improved_any = True
                offer(assignment, sequence, init_order, current_fitness)

        return assignment, sequence, init_order, current_fitness, improved_any

    # Give the best deterministic starts exact coordinate descent first.
    start_count = min(1, len(pool))
    for index in range(start_count):
        if time.time() >= deadline:
            break
        entry = pool[index]
        descent(entry[1][:], entry[2][:], entry[3][:], True)

    # Iterated local search uses every remaining wall-clock slice.  Perturbations
    # act on all three decision spaces; sampled descent rapidly settles each new
    # basin and periodic full descent polishes the incumbent.
    restart = 0
    while time.time() < deadline:
        if restart % 5 == 0:
            assignment = best_assignment[:]
            sequence = best_sequence[:]
            init_order = best_init[:]
        else:
            entry = pool[rng.randrange(len(pool))]
            assignment = entry[1][:]
            sequence = entry[2][:]
            init_order = entry[3][:]

        strength = 1 + restart % 7
        for _ in range(strength):
            move = rng.randrange(7)
            if move <= 1 and n > 1:
                old_position = rng.randrange(n)
                new_position = rng.randrange(n)
                sequence = relocate(sequence, old_position, new_position)
            elif move == 2 and n > 1:
                left = rng.randrange(n)
                right = rng.randrange(n)
                sequence[left], sequence[right] = sequence[right], sequence[left]
            elif move == 3 and n > 1:
                old_position = rng.randrange(n)
                new_position = rng.randrange(n)
                init_order = relocate(init_order, old_position, new_position)
            elif move == 4 and n > 1:
                left = rng.randrange(n)
                right = rng.randrange(n)
                init_order[left], init_order[right] = init_order[right], init_order[left]
            elif move == 5:
                job = rng.randrange(n)
                assignment[job] = rng.randrange(machine_count)
            elif n > 1:
                left = rng.randrange(n)
                right = rng.randrange(n)
                assignment[left], assignment[right] = (
                    assignment[right], assignment[left]
                )

        fitness, _ = decode(assignment, sequence, init_order)
        offer(assignment, sequence, init_order, fitness)
        descent(assignment, sequence, init_order, restart % 8 == 7)
        restart += 1

    final_fitness, _, s1, s2, s3 = decode(
        best_assignment, best_sequence, best_init, True
    )
    return {
        "machine_assignment": {
            "job_{}".format(job): best_assignment[job] for job in range(n)
        },
        "server_sequence": ["job_{}".format(job) for job in best_sequence],
        "start_times": {
            "job_{}".format(job): [s1[job], s2[job], s3[job]]
            for job in range(n)
        },
        "makespan": final_fitness[0],
    }


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime three-space search for the hybrid reentrant shop problem."""
    try:
        return _search(instance, time_limit_s)
    except Exception:
        return _safe_solution(instance)
