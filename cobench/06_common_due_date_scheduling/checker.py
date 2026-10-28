"""Feasibility checker for the common-due-date single-machine schedule."""

from collections import Counter


def _is_int(value):
    """Return whether value is an integer but not a JSON/Python boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def check_solution(instance: dict, solution: dict) -> list:
    """Return semantic or format violations; an empty list means feasible."""
    violations = []

    # Format checks only establish that the authoritative order can be read safely.
    if not isinstance(instance, dict):
        return ["INSTANCE_FORMAT | instance must be a dictionary."]
    if not isinstance(solution, dict):
        return ["FORMAT | solution must be a dictionary."]
    if "order" not in solution:
        return ["FORMAT | required field 'order' is missing."]

    order = solution["order"]
    if not isinstance(order, list):
        return ["FORMAT | 'order' must be a list of integer job indices."]
    bad_order_types = [
        position for position, job in enumerate(order) if not _is_int(job)
    ]
    if bad_order_types:
        return [
            "FORMAT | every 'order' entry must be an integer job index; "
            f"invalid positions: {bad_order_types[:10]}."
        ]

    n = instance.get("n")
    if not _is_int(n):
        return ["INSTANCE_FORMAT | instance field 'n' must be an integer."]

    processing_times = instance.get("processing_times")
    if not isinstance(processing_times, list):
        return [
            "INSTANCE_FORMAT | instance field 'processing_times' must be a list."
        ]
    if n >= 0 and len(processing_times) < n:
        return [
            "INSTANCE_FORMAT | 'processing_times' has fewer entries than are "
            "needed for job indices 0 through n-1."
        ]
    bad_processing_types = [
        job
        for job, processing_time in enumerate(processing_times[: max(n, 0)])
        if not _is_int(processing_time)
    ]
    if bad_processing_types:
        return [
            "INSTANCE_FORMAT | processing times used by the schedule must be "
            f"integers; invalid job indices: {bad_processing_types[:10]}."
        ]

    due_date = instance.get("due_date")
    if not _is_int(due_date):
        return ["INSTANCE_FORMAT | instance field 'due_date' must be an integer."]

    counts = Counter(order)
    invalid_jobs = sorted({job for job in order if job < 0 or job >= n})
    duplicate_jobs = sorted(job for job, count in counts.items() if count > 1)

    # C1 | Each job must be assigned to exactly one position.
    c1_ok = (
        len(order) == n
        and not invalid_jobs
        and len(counts) == n
        and not duplicate_jobs
    )
    if not c1_ok:
        reasons = []
        if len(order) != n:
            reasons.append(f"order has {len(order)} entries but n is {n}")
        if invalid_jobs:
            reasons.append(f"out-of-range job indices are {invalid_jobs[:10]}")
        if duplicate_jobs:
            reasons.append(f"repeated job indices are {duplicate_jobs[:10]}")
        if not reasons:
            reasons.append("the order is not a one-to-one cover of all job indices")
        violations.append("C1 | " + "; ".join(reasons) + ".")

    # C2 | Each position must be occupied by exactly one job.
    # A list gives one scalar entry per represented position; length n covers every
    # required position, and the range check makes each entry an actual job.
    c2_ok = len(order) == n and not invalid_jobs
    if not c2_ok:
        reasons = []
        if len(order) != n:
            reasons.append(
                f"positions 0 through n-1 require {n} entries, got {len(order)}"
            )
        if invalid_jobs:
            reasons.append(
                f"some positions name no valid job: {invalid_jobs[:10]}"
            )
        violations.append("C2 | " + "; ".join(reasons) + ".")

    # C3-C6 need a valid assignment before indexed derived values can be rebuilt.
    if not c1_ok or not c2_ok:
        return violations
    if n == 0:
        return violations

    completion_times = []

    # C3 | C[0] equals the processing time assigned to the first position.
    first_completion = processing_times[order[0]]
    first_assignment_sum = sum(
        processing_times[job] if order[0] == job else 0 for job in range(n)
    )
    if first_completion != first_assignment_sum:
        violations.append(
            "C3 | the recomputed first completion time does not equal the "
            "processing-time assignment sum."
        )
    if first_completion < 0:
        violations.append(
            "C3 | the required first completion time is negative, so it cannot "
            "also satisfy the declared domain C[0] >= 0."
        )
    completion_times.append(first_completion)

    # C4 | Each later C[k] is C[k-1] plus the current job's processing time.
    for position in range(1, n):
        assigned_processing_time = processing_times[order[position]]
        current_completion = completion_times[-1] + assigned_processing_time
        if current_completion != (
            completion_times[position - 1] + assigned_processing_time
        ):
            violations.append(
                f"C4 | completion time at position {position} does not follow "
                "the required no-idle recurrence."
            )
        if current_completion < 0:
            violations.append(
                f"C4 | the required completion time at position {position} is "
                "negative, so it cannot satisfy the declared domain C[k] >= 0."
            )
        completion_times.append(current_completion)

    completion_by_job = {
        job: completion_times[position] for position, job in enumerate(order)
    }

    # C5 | E[j] is max(0, due_date - completion time of job j).
    for job in range(n):
        difference = due_date - completion_by_job[job]
        recomputed_earliness = max(0, difference)
        if recomputed_earliness < 0 or recomputed_earliness < difference:
            violations.append(
                f"C5 | recomputed earliness for job {job} violates either "
                "E[j] >= 0 or E[j] >= due-date minus completion-time."
            )

    # C6 | T[j] is max(0, completion time of job j - due_date).
    for job in range(n):
        difference = completion_by_job[job] - due_date
        recomputed_tardiness = max(0, difference)
        if recomputed_tardiness < 0 or recomputed_tardiness < difference:
            violations.append(
                f"C6 | recomputed tardiness for job {job} violates either "
                "T[j] >= 0 or T[j] >= completion-time minus due-date."
            )

    return violations
