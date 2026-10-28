"""Feasibility checker for the generalized assignment problem."""


def _is_int(value):
    """Return whether value is an integer in the JSON-schema sense."""
    return isinstance(value, int) and not isinstance(value, bool)


def _check_c1(assignments, num_jobs, num_machines):
    """C1: every job is assigned to exactly one machine."""
    violations = []
    if not isinstance(assignments, dict):
        return ["C1 | 'assignments' must be a dict[str, int]."], None

    expected_keys = {str(job) for job in range(num_jobs)}
    actual_keys = set(assignments)

    missing = sorted(expected_keys - actual_keys, key=int)
    if missing:
        violations.append(
            "C1 | missing job keys: " + ", ".join(repr(key) for key in missing)
        )

    unexpected = actual_keys - expected_keys
    if unexpected:
        rendered = sorted((repr(key) for key in unexpected))
        violations.append("C1 | invalid or out-of-range job keys: " + ", ".join(rendered))

    assignment_by_job = {}
    for job in range(num_jobs):
        key = str(job)
        if key not in assignments:
            continue

        machine = assignments[key]
        if not _is_int(machine):
            violations.append(
                f"C1 | job {job} has non-integer machine index {machine!r}."
            )
        elif not 0 <= machine < num_machines:
            violations.append(
                f"C1 | job {job} has machine index {machine}, outside "
                f"0..{num_machines - 1}."
            )
        else:
            assignment_by_job[job] = machine

    if violations:
        return violations, None
    return [], assignment_by_job


def _check_c2(instance, assignment_by_job, num_jobs, num_machines):
    """C2: each machine's assigned resource load is within capacity."""
    violations = []
    resource_matrix = instance.get("resource_matrix")
    capacities = instance.get("capacities")

    if not isinstance(resource_matrix, list):
        violations.append("C2 | instance 'resource_matrix' must be a list of rows.")
    if not isinstance(capacities, list):
        violations.append("C2 | instance 'capacities' must be a list.")
    if violations:
        return violations

    loads = [0 for _ in range(num_machines)]
    machine_data_valid = [True for _ in range(num_machines)]

    for machine in range(num_machines):
        if machine >= len(capacities):
            violations.append(f"C2 | missing capacity for machine {machine}.")
            machine_data_valid[machine] = False
        elif not _is_int(capacities[machine]):
            violations.append(
                f"C2 | capacity for machine {machine} is not an integer: "
                f"{capacities[machine]!r}."
            )
            machine_data_valid[machine] = False

    for job in range(num_jobs):
        machine = assignment_by_job[job]
        if machine >= len(resource_matrix):
            violations.append(
                f"C2 | resource_matrix has no row for assigned machine {machine}."
            )
            machine_data_valid[machine] = False
            continue

        row = resource_matrix[machine]
        if not isinstance(row, list):
            violations.append(
                f"C2 | resource_matrix row {machine} must be a list."
            )
            machine_data_valid[machine] = False
            continue
        if job >= len(row):
            violations.append(
                f"C2 | resource_matrix[{machine}] has no value for job {job}."
            )
            machine_data_valid[machine] = False
            continue

        resource = row[job]
        if not _is_int(resource):
            violations.append(
                f"C2 | resource_matrix[{machine}][{job}] is not an integer: "
                f"{resource!r}."
            )
            machine_data_valid[machine] = False
            continue
        loads[machine] += resource

    for machine in range(num_machines):
        if not machine_data_valid[machine]:
            continue
        capacity = capacities[machine]
        load = loads[machine]
        # Both sides are exact integers, so a tolerance would admit real violations.
        if load > capacity:
            violations.append(
                f"C2 | machine {machine} load {load} exceeds capacity {capacity}."
            )

    return violations


def check_solution(instance: dict, solution: dict) -> list:
    """Return a list of C1/C2 violations; an empty list means feasible."""
    try:
        if not isinstance(instance, dict):
            return ["FORMAT | instance must be a dict."]
        if not isinstance(solution, dict):
            return ["FORMAT | solution must be a dict."]
        if "assignments" not in solution:
            return ["FORMAT | missing required solution field 'assignments'."]

        num_jobs = instance.get("num_jobs")
        num_machines = instance.get("num_machines")
        if not _is_int(num_jobs):
            return ["FORMAT | instance 'num_jobs' must be an integer."]
        if not _is_int(num_machines):
            return ["FORMAT | instance 'num_machines' must be an integer."]

        c1_violations, assignment_by_job = _check_c1(
            solution["assignments"], num_jobs, num_machines
        )
        if c1_violations:
            return c1_violations

        return _check_c2(
            instance, assignment_by_job, num_jobs, num_machines
        )
    except Exception as exc:
        return [f"FORMAT | checker could not safely read the input: {type(exc).__name__}: {exc}"]
