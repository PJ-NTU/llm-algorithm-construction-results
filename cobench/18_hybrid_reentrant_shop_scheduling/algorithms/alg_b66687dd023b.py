# Library slot 8/10  id=alg_b66687dd023b
# score=311.9329809282914  feasible_rate=1.0
# score_vec=[334.59999999999997, 808.5324499076448, 163.00619627837804, 163.45375400892746, 216.85604525962597, 200.0353638283379, 223.39999999999998, 346.99399999999986, 327.43099999999987, 335.0209999999998]

import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime iterated-greedy solver with exact single-server subcases."""
    t0 = time.time()
    deadline = t0 + max(0.0, float(time_limit_s)) * 0.85

    try:
        n = int(instance["num_jobs"])
        machine_count = max(1, int(instance["num_machines"]))
        p1 = [float(value) for value in instance["init_processing_times"]]
        p2 = [float(value) for value in instance["setup_processing_times"]]
        p3 = [float(value) for value in instance["main_processing_times"]]
    except Exception:
        return {
            "machine_assignment": {},
            "server_sequence": [],
            "start_times": {},
            "makespan": 0.0,
        }

    if n <= 0:
        return {
            "machine_assignment": {},
            "server_sequence": [],
            "start_times": {},
            "makespan": 0.0,
        }

    # For a fixed assignment and server order, initialization follows the
    # mandatory natural job order independently on each primary machine.
    def decode(assignment, sequence, need_times=False):
        s1 = [0.0] * n
        machine_free = [0.0] * machine_count
        for job in range(n):
            machine = assignment[job]
            s1[job] = machine_free[machine]
            machine_free[machine] += p1[job]

        s2 = [0.0] * n
        server_free = 0.0
        for job in sequence:
            server_free = max(server_free, s1[job] + p1[job])
            s2[job] = server_free
            server_free += p2[job]

        s3 = [0.0] * n
        main_free = machine_free[:]
        makespan = 0.0
        completion_sum = 0.0
        for job in range(n):
            machine = assignment[job]
            start = max(s2[job] + p2[job], main_free[machine])
            s3[job] = start
            completion = start + p3[job]
            main_free[machine] = completion
            makespan = max(makespan, completion)
            completion_sum += completion

        # Secondary terms make equal-makespan plateaus navigable without ever
        # overriding the actual objective.
        fitness = (makespan, completion_sum, server_free)
        if need_times:
            return fitness, s1, s2, s3
        return fitness

    def make_solution(assignment, sequence):
        fitness, s1, s2, s3 = decode(assignment, sequence, True)
        return {
            "machine_assignment": {
                "job_%d" % job: int(assignment[job])
                for job in range(n)
            },
            "server_sequence": ["job_%d" % job for job in sequence],
            "start_times": {
                "job_%d" % job: [s1[job], s2[job], s3[job]]
                for job in range(n)
            },
            "makespan": float(fitness[0]),
        }

    # This solution is complete before any expensive work starts.
    if machine_count >= n:
        fallback_assignment = list(range(n))
    elif machine_count == 1:
        fallback_assignment = [0] * n
    else:
        fallback_assignment = [job % machine_count for job in range(n)]
    fallback_sequence = sorted(
        range(n), key=lambda job: (-p3[job], p1[job], p2[job], job)
    )
    best_assignment = fallback_assignment[:]
    best_sequence = fallback_sequence[:]
    best_fitness = decode(best_assignment, best_sequence)

    if time.time() >= deadline:
        return make_solution(best_assignment, best_sequence)

    seed = n * 1000003 + machine_count * 10007
    for job in range(n):
        seed += int((19.0 * p1[job] + 37.0 * p2[job]
                    + 23.0 * p3[job]) * 1000.0)
    rng = random.Random(seed)

    def consider(assignment, sequence):
        nonlocal best_assignment, best_sequence, best_fitness
        fitness = decode(assignment, sequence)
        if fitness < best_fitness:
            best_fitness = fitness
            best_assignment = assignment[:]
            best_sequence = sequence[:]
        return fitness

    def chain_tails(assignment):
        tails = p3[:]
        for job in range(n - 2, -1, -1):
            if assignment[job] == assignment[job + 1]:
                tails[job] += tails[job + 1]
        return tails

    def schrage_order(release, duration, tail):
        unreleased = sorted(range(n), key=lambda job: release[job])
        next_index = 0
        ready = []
        order = []
        now = release[unreleased[0]] if unreleased else 0.0
        while len(order) < n:
            while (next_index < n
                   and release[unreleased[next_index]] <= now + 1e-12):
                job = unreleased[next_index]
                heapq.heappush(ready, (-tail[job], duration[job], job))
                next_index += 1
            if not ready:
                now = release[unreleased[next_index]]
                continue
            unused_tail, unused_duration, job = heapq.heappop(ready)
            order.append(job)
            now += duration[job]
        return order

    # Alternating dispatch rebuilds the server order from releases induced by
    # the previous order. It stops on a fixed point/cycle, not an iteration cap.
    def dispatch_fixed_point(assignment, sequence):
        seen = set()
        current = sequence[:]
        tails = chain_tails(assignment)
        while time.time() < deadline:
            marker = tuple(current)
            if marker in seen:
                break
            seen.add(marker)
            unused_fit, s1, unused_s2, unused_s3 = decode(
                assignment, current, True
            )
            release = [s1[job] + p1[job] for job in range(n)]
            rebuilt = schrage_order(release, p2, tails)
            consider(assignment, rebuilt)
            current = rebuilt
        return current

    # Carlier branch-and-bound for cases that reduce exactly to
    # 1|r_j,q_j|Cmax: one primary machine, or one machine per job.
    def exact_server_subproblem(assignment, release, tail, initial_sequence):
        nonlocal best_assignment, best_sequence, best_fitness
        r_work = release[:]
        q_work = tail[:]
        incumbent_value = float("inf")
        incumbent_order = initial_sequence[:]

        def schrage_with_value():
            unreleased = sorted(range(n), key=lambda job: r_work[job])
            next_index = 0
            ready = []
            order = []
            now = r_work[unreleased[0]] if unreleased else 0.0
            value = 0.0
            while len(order) < n:
                while (next_index < n
                       and r_work[unreleased[next_index]] <= now + 1e-12):
                    job = unreleased[next_index]
                    heapq.heappush(
                        ready, (-q_work[job], p2[job], job)
                    )
                    next_index += 1
                if not ready:
                    now = r_work[unreleased[next_index]]
                    continue
                unused_q, unused_p, job = heapq.heappop(ready)
                order.append(job)
                now += p2[job]
                value = max(value, now + q_work[job])
            return value, order

        def preemptive_bound():
            events = sorted(range(n), key=lambda job: r_work[job])
            event_index = 0
            ready = []
            now = r_work[events[0]] if events else 0.0
            bound = 0.0
            while event_index < n or ready:
                while (event_index < n
                       and r_work[events[event_index]] <= now + 1e-12):
                    job = events[event_index]
                    heapq.heappush(ready, (-q_work[job], job, p2[job]))
                    event_index += 1
                if not ready:
                    now = r_work[events[event_index]]
                    continue
                neg_tail, job, remaining = heapq.heappop(ready)
                next_release = (r_work[events[event_index]]
                                if event_index < n else float("inf"))
                run = min(remaining, next_release - now)
                now += run
                remaining -= run
                if remaining > 1e-12:
                    heapq.heappush(ready, (neg_tail, job, remaining))
                else:
                    bound = max(bound, now - neg_tail)
            return bound

        def branch():
            nonlocal incumbent_value, incumbent_order
            if time.time() >= deadline:
                return
            upper, order = schrage_with_value()
            actual = consider(assignment, order)
            if upper < incumbent_value:
                incumbent_value = upper
                incumbent_order = order[:]

            completion = []
            now = 0.0
            for job in order:
                now = max(now, r_work[job]) + p2[job]
                completion.append(now)
            tolerance = 1e-8 * (1.0 + abs(upper))
            b_pos = -1
            for index, job in enumerate(order):
                if abs(completion[index] + q_work[job] - upper) <= tolerance:
                    b_pos = index
            if b_pos < 0:
                return
            b_job = order[b_pos]
            a_pos = b_pos
            block_duration = 0.0
            for index in range(b_pos, -1, -1):
                block_duration += p2[order[index]]
                if abs(r_work[order[index]] + block_duration
                       + q_work[b_job] - upper) <= tolerance:
                    a_pos = index
            c_pos = -1
            for index in range(a_pos, b_pos):
                if q_work[order[index]] < q_work[b_job] - tolerance:
                    c_pos = index
            if c_pos < 0:
                return

            c_job = order[c_pos]
            suffix_jobs = order[c_pos + 1:b_pos + 1]
            suffix_p = sum(p2[job] for job in suffix_jobs)
            suffix_r = min(r_work[job] for job in suffix_jobs)
            suffix_q = min(q_work[job] for job in suffix_jobs)
            suffix_bound = suffix_r + suffix_p + suffix_q

            old_release = r_work[c_job]
            r_work[c_job] = max(r_work[c_job], suffix_r + suffix_p)
            lower = max(preemptive_bound(), suffix_bound)
            if lower < incumbent_value - tolerance:
                branch()
            r_work[c_job] = old_release

            if time.time() >= deadline:
                return
            old_tail = q_work[c_job]
            q_work[c_job] = max(q_work[c_job], suffix_q + suffix_p)
            lower = max(preemptive_bound(), suffix_bound)
            if lower < incumbent_value - tolerance:
                branch()
            q_work[c_job] = old_tail

        try:
            branch()
        except (RecursionError, IndexError):
            pass
        consider(assignment, incumbent_order)

    # Exact special cases are both structurally important and very cheap to
    # recognize. A unique primary machine per job dominates sharing one.
    if machine_count == 1:
        release = [0.0] * n
        running = 0.0
        for job in range(n):
            running += p1[job]
            release[job] = running
        tails = [0.0] * n
        running = 0.0
        for job in range(n - 1, -1, -1):
            running += p3[job]
            tails[job] = running
        initial = schrage_order(release, p2, tails)
        consider(fallback_assignment, initial)
        exact_server_subproblem(
            fallback_assignment, release, tails, initial
        )
    elif machine_count >= n:
        release = p1[:]
        initial = schrage_order(release, p2, p3)
        consider(fallback_assignment, initial)
        exact_server_subproblem(
            fallback_assignment, release, p3[:], initial
        )

    if time.time() >= deadline:
        return make_solution(best_assignment, best_sequence)

    # Diverse deterministic starting points. Unlike a population algorithm,
    # each is immediately folded into one incumbent and then discarded.
    sequences = [
        list(range(n)),
        sorted(range(n), key=lambda job: (-p3[job], p1[job], job)),
        sorted(range(n), key=lambda job: (p1[job], -p3[job], job)),
        sorted(range(n), key=lambda job: (p2[job], -p3[job], job)),
        sorted(range(n), key=lambda job:
               (p1[job] + p2[job] - p3[job], job)),
    ]
    assignments = [fallback_assignment[:]]
    if 1 < machine_count < n:
        for offset in range(min(machine_count, 4)):
            loads = [0.0] * machine_count
            assignment = [0] * n
            for job in range(n):
                choices = list(range(machine_count))
                machine = min(
                    choices,
                    key=lambda value: (
                        loads[value] + p1[job]
                        + (p3[job] if job > 0
                           and assignment[job - 1] == value else 0.0),
                        (value - offset) % machine_count,
                    ),
                )
                assignment[job] = machine
                loads[machine] += p1[job]
            assignments.append(assignment)

    for assignment in assignments:
        for sequence in sequences:
            if time.time() >= deadline:
                return make_solution(best_assignment, best_sequence)
            consider(assignment, sequence)
        dispatch_fixed_point(assignment, best_sequence)

    current_assignment = best_assignment[:]
    current_sequence = best_sequence[:]
    current_fitness = best_fitness

    def rebuild_assignment(sequence, tail_bias):
        """Jointly decode a permutation into a balanced, mostly chain-free coloring."""
        assignment = [-1] * n
        loads = [0.0] * machine_count
        largest_tail = [0.0] * machine_count
        for job in sequence:
            forbidden = set()
            if job > 0 and assignment[job - 1] >= 0:
                forbidden.add(assignment[job - 1])
            if job + 1 < n and assignment[job + 1] >= 0:
                forbidden.add(assignment[job + 1])
            choices = [machine for machine in range(machine_count)
                       if machine not in forbidden]
            # With two colors, two already colored neighbors can force a chain.
            if not choices:
                choices = list(range(machine_count))
            machine = min(
                choices,
                key=lambda value: (
                    loads[value] + p1[job]
                    + tail_bias * max(largest_tail[value], p3[job]),
                    loads[value],
                    value,
                ),
            )
            assignment[job] = machine
            loads[machine] += p1[job]
            largest_tail[machine] = max(
                largest_tail[machine], p3[job]
            )
        return assignment

    def useful_machines(assignment, job):
        used = set(assignment)
        choices = list(used)
        if len(used) < machine_count:
            for machine in range(machine_count):
                if machine not in used:
                    choices.append(machine)
                    break
        return [machine for machine in choices
                if machine != assignment[job]]

    def best_relocation(assignment, sequence):
        local_best = decode(assignment, sequence)
        local_sequence = sequence
        # Critical-tail jobs receive priority, but every move is examined.
        order_positions = sorted(
            range(n), key=lambda pos: -p3[sequence[pos]]
        )
        for old_pos in order_positions:
            if time.time() >= deadline:
                break
            job = sequence[old_pos]
            reduced = sequence[:old_pos] + sequence[old_pos + 1:]
            for new_pos in range(n):
                if new_pos == old_pos:
                    continue
                candidate = reduced[:new_pos] + [job] + reduced[new_pos:]
                fitness = decode(assignment, candidate)
                if fitness < local_best:
                    local_best = fitness
                    local_sequence = candidate
                if time.time() >= deadline:
                    break
        return local_best, local_sequence

    def best_recolor(assignment, sequence):
        local_best = decode(assignment, sequence)
        local_assignment = assignment
        jobs = sorted(range(n), key=lambda job: -(p1[job] + p3[job]))
        for job in jobs:
            if time.time() >= deadline:
                break
            for machine in useful_machines(assignment, job):
                candidate = assignment[:]
                candidate[job] = machine
                fitness = decode(candidate, sequence)
                if fitness < local_best:
                    local_best = fitness
                    local_assignment = candidate
                if time.time() >= deadline:
                    break
        return local_best, local_assignment

    def joint_descent(sequence, tail_bias):
        """Best insertion where every trial also rebuilds the assignment."""
        assignment = rebuild_assignment(sequence, tail_bias)
        local_best = decode(assignment, sequence)
        local_sequence = sequence
        positions = list(range(n))
        rng.shuffle(positions)
        for old_pos in positions:
            if time.time() >= deadline:
                break
            job = sequence[old_pos]
            reduced = sequence[:old_pos] + sequence[old_pos + 1:]
            for new_pos in range(n):
                if new_pos == old_pos:
                    continue
                candidate_sequence = (
                    reduced[:new_pos] + [job] + reduced[new_pos:]
                )
                candidate_assignment = rebuild_assignment(
                    candidate_sequence, tail_bias
                )
                fitness = decode(candidate_assignment, candidate_sequence)
                if fitness < local_best:
                    local_best = fitness
                    local_sequence = candidate_sequence
                if time.time() >= deadline:
                    break
        local_assignment = rebuild_assignment(local_sequence, tail_bias)
        return local_best, local_assignment, local_sequence

    def best_two_machine_segment(assignment, sequence):
        """Invert one natural-index interval to cross the two-color barrier."""
        local_best = decode(assignment, sequence)
        local_assignment = assignment
        for left in range(n):
            if time.time() >= deadline:
                break
            candidate = assignment[:]
            for right in range(left, n):
                candidate[right] = 1 - candidate[right]
                fitness = decode(candidate, sequence)
                if fitness < local_best:
                    local_best = fitness
                    local_assignment = candidate[:]
                if time.time() >= deadline:
                    break
        return local_best, local_assignment

    # On three or more machines the strongest coupling is between a server
    # insertion and the initialization-machine coloring.  Spend an initial
    # wall-clock phase on this joint landscape; stop early only at a valid
    # analytical lower bound.
    server_lower_bound = min(p1) + sum(p2) + min(p3)
    load_lower_bound = sum(p1) / machine_count + min(p3)
    lower_bound = max(server_lower_bound, load_lower_bound)
    joint_phase_end = t0 + (deadline - t0) * (
        0.72 if machine_count >= 3 else 0.28
    )
    joint_sequence = best_sequence[:]
    joint_fitness = best_fitness
    while time.time() < joint_phase_end:
        bias = rng.choice((0.03, 0.05, 0.08, 0.10, 0.12, 0.16, 0.20))
        if joint_fitness[0] <= lower_bound + 1e-8 * (1.0 + lower_bound):
            break
        if rng.random() < 0.35:
            joint_sequence = list(range(n))
            rng.shuffle(joint_sequence)
        else:
            joint_sequence = best_sequence[:]
            shake_count = 2 + rng.randrange(max(1, min(8, n) - 1))
            for unused_move in range(shake_count):
                old_pos = rng.randrange(n)
                job = joint_sequence.pop(old_pos)
                joint_sequence.insert(rng.randrange(n), job)
        improving = True
        while improving and time.time() < joint_phase_end:
            previous = decode(
                rebuild_assignment(joint_sequence, bias), joint_sequence
            )
            joint_fitness, joint_assignment, rebuilt_sequence = joint_descent(
                joint_sequence, bias
            )
            consider(joint_assignment, rebuilt_sequence)
            improving = joint_fitness < previous
            joint_sequence = rebuilt_sequence

    current_assignment = best_assignment[:]
    current_sequence = best_sequence[:]
    current_fitness = best_fitness

    if machine_count == 2 and time.time() < deadline:
        segment_fitness, segment_assignment = best_two_machine_segment(
            current_assignment, current_sequence
        )
        consider(segment_assignment, current_sequence)
        current_assignment = best_assignment[:]
        current_sequence = best_sequence[:]
        current_fitness = best_fitness

    # Wall-clock driven iterated greedy search. Ruin moves cross valleys; exact
    # best relocation/recolor passes intensify the reconstructed schedule.
    while time.time() < deadline:
        candidate_assignment = current_assignment[:]
        candidate_sequence = current_sequence[:]
        neighborhood = rng.randrange(6)
        strength = 2 + rng.randrange(max(1, min(7, n) - 1))

        if neighborhood in (0, 1, 2) and n > 1:
            if neighborhood == 0:
                left = rng.randrange(n)
                right = rng.randrange(n)
                if left > right:
                    left, right = right, left
                candidate_sequence[left:right + 1] = reversed(
                    candidate_sequence[left:right + 1]
                )
            else:
                for unused_move in range(strength):
                    old_pos = rng.randrange(n)
                    job = candidate_sequence.pop(old_pos)
                    new_pos = rng.randrange(n)
                    candidate_sequence.insert(new_pos, job)
        elif neighborhood in (3, 4) and machine_count > 1:
            for unused_move in range(strength):
                job = rng.randrange(n)
                choices = useful_machines(candidate_assignment, job)
                if choices:
                    candidate_assignment[job] = rng.choice(choices)
        else:
            candidate_sequence = dispatch_fixed_point(
                candidate_assignment, candidate_sequence
            )

        candidate_fitness, candidate_sequence = best_relocation(
            candidate_assignment, candidate_sequence
        )
        if time.time() < deadline and machine_count > 1:
            candidate_fitness, candidate_assignment = best_recolor(
                candidate_assignment, candidate_sequence
            )
        if time.time() < deadline and machine_count == 2:
            candidate_fitness, candidate_assignment = best_two_machine_segment(
                candidate_assignment, candidate_sequence
            )
        candidate_fitness = consider(
            candidate_assignment, candidate_sequence
        )

        elapsed_fraction = min(
            1.0, max(0.0, (time.time() - t0)
                         / max(1e-9, deadline - t0))
        )
        scale = max(1e-9, best_fitness[0] * 0.025
                    * (1.0 - elapsed_fraction) ** 2)
        delta = candidate_fitness[0] - current_fitness[0]
        accept = candidate_fitness <= current_fitness
        if not accept and delta < 50.0 * scale:
            accept = rng.random() < math.exp(-delta / scale)
        if accept:
            current_assignment = candidate_assignment
            current_sequence = candidate_sequence
            current_fitness = candidate_fitness
        else:
            current_assignment = best_assignment[:]
            current_sequence = best_sequence[:]
            current_fitness = best_fitness

    return make_solution(best_assignment, best_sequence)

