"""Feasibility checker for the fixed-station assembly-line model."""


def _is_int(value):
    """Return True only for integers, excluding booleans."""
    return type(value) is int


def check(instance: dict, solution: dict) -> list:
    violations = []

    if not isinstance(instance, dict):
        return ["structure: instance must be a dict"]
    if not isinstance(solution, dict):
        return ["structure: solution must be a dict"]

    required_instance_fields = (
        "num_tasks",
        "processing_times",
        "cycle_time",
        "num_stations",
        "precedence_arcs",
        "earliest_stations",
        "latest_stations",
    )
    missing_instance_fields = [
        key for key in required_instance_fields if key not in instance
    ]
    if missing_instance_fields:
        return [
            "structure: instance is missing fields: "
            + ", ".join(missing_instance_fields)
        ]

    n = instance["num_tasks"]
    processing_times = instance["processing_times"]
    cycle_time = instance["cycle_time"]
    num_stations = instance["num_stations"]
    precedence_arcs = instance["precedence_arcs"]
    earliest_stations = instance["earliest_stations"]
    latest_stations = instance["latest_stations"]

    instance_errors = []
    if not _is_int(n) or n <= 0:
        instance_errors.append("instance.num_tasks must be a positive integer")
    if not _is_int(cycle_time) or cycle_time <= 0:
        instance_errors.append("instance.cycle_time must be a positive integer")
    if not _is_int(num_stations) or num_stations <= 0:
        instance_errors.append("instance.num_stations must be a positive integer")

    if _is_int(n) and n > 0:
        if (
            not isinstance(processing_times, list)
            or len(processing_times) != n
            or any(not _is_int(value) or value <= 0 for value in processing_times)
        ):
            instance_errors.append(
                "instance.processing_times must contain num_tasks positive integers"
            )
        if (
            not isinstance(earliest_stations, list)
            or len(earliest_stations) != n
            or any(not _is_int(value) for value in earliest_stations)
        ):
            instance_errors.append(
                "instance.earliest_stations must contain num_tasks integers"
            )
        if (
            not isinstance(latest_stations, list)
            or len(latest_stations) != n
            or any(not _is_int(value) for value in latest_stations)
        ):
            instance_errors.append(
                "instance.latest_stations must contain num_tasks integers"
            )

    if not isinstance(precedence_arcs, list):
        instance_errors.append("instance.precedence_arcs must be a list")
    elif _is_int(n) and n > 0:
        for arc_index, arc in enumerate(precedence_arcs):
            if (
                not isinstance(arc, list)
                or len(arc) != 2
                or not _is_int(arc[0])
                or not _is_int(arc[1])
                or not 1 <= arc[0] <= n
                or not 1 <= arc[1] <= n
            ):
                instance_errors.append(
                    f"instance.precedence_arcs[{arc_index}] must be "
                    f"[predecessor, successor] with task IDs in 1..{n}"
                )

    if instance_errors:
        return ["structure: " + error for error in instance_errors]

    if any(
        earliest_stations[i] < 1
        or latest_stations[i] > num_stations
        or earliest_stations[i] > latest_stations[i]
        for i in range(n)
    ):
        return [
            "structure: every input station interval must satisfy "
            "1 <= earliest_stations[i] <= latest_stations[i] <= num_stations"
        ]

    # Solution structure completeness
    if "task_stations" not in solution:
        violations.append("structure: missing solution field 'task_stations'")
        task_stations = None
    else:
        task_stations = solution["task_stations"]
        if not isinstance(task_stations, list):
            violations.append("structure: 'task_stations' must be a list")

    if "worst_case_idle_time" not in solution:
        violations.append(
            "structure: missing solution field 'worst_case_idle_time'"
        )
        worst_case_idle_time = None
        idle_value_valid = False
    else:
        worst_case_idle_time = solution["worst_case_idle_time"]
        idle_value_valid = (
            _is_int(worst_case_idle_time) and worst_case_idle_time >= 0
        )
        if not idle_value_valid:
            violations.append(
                "structure: 'worst_case_idle_time' must be a nonnegative integer"
            )

    # C1: every task has exactly one assignment in its inclusive feasible interval.
    assignment_shape_valid = isinstance(task_stations, list) and len(task_stations) == n
    if isinstance(task_stations, list) and len(task_stations) != n:
        violations.append(
            f"C1: task_stations must have length {n}, got {len(task_stations)}"
        )

    assignments_usable = assignment_shape_valid
    if assignment_shape_valid:
        for i, station in enumerate(task_stations, start=1):
            earliest = earliest_stations[i - 1]
            latest = latest_stations[i - 1]
            if not _is_int(station):
                violations.append(
                    f"C1: task {i} must have one integer station assignment"
                )
                assignments_usable = False
            elif not earliest <= station <= latest:
                violations.append(
                    f"C1: task {i} is assigned to station {station}, outside "
                    f"its feasible interval [{earliest}, {latest}]"
                )
                if not 1 <= station <= num_stations:
                    assignments_usable = False

    # C2: each immediate predecessor is assigned no later than its successor.
    if assignments_usable:
        for predecessor, successor in precedence_arcs:
            predecessor_station = task_stations[predecessor - 1]
            successor_station = task_stations[successor - 1]
            if predecessor_station > successor_station:
                violations.append(
                    f"C2: precedence arc [{predecessor}, {successor}] is violated; "
                    f"stations are {predecessor_station} and {successor_station}"
                )

    station_loads = None
    if assignments_usable:
        station_loads = [0] * num_stations
        for task_index, station in enumerate(task_stations):
            station_loads[station - 1] += processing_times[task_index]

    # C3: the processing load of every station does not exceed the cycle time.
    if station_loads is not None:
        for station, load in enumerate(station_loads, start=1):
            if load > cycle_time:
                violations.append(
                    f"C3: station {station} has load {load}, exceeding "
                    f"cycle_time {cycle_time}"
                )

    # C4: the reported worst-case idle time dominates every station's idle time.
    if station_loads is not None and idle_value_valid:
        for station, load in enumerate(station_loads, start=1):
            idle_time = cycle_time - load
            if worst_case_idle_time < idle_time:
                violations.append(
                    f"C4: worst_case_idle_time {worst_case_idle_time} is below "
                    f"station {station}'s idle time {idle_time}"
                )

    # C5: the reported worst-case idle time satisfies the stated slack lower bound.
    if idle_value_valid:
        total_slack = num_stations * cycle_time - sum(processing_times)
        lower_bound = total_slack // cycle_time
        if worst_case_idle_time < lower_bound:
            violations.append(
                f"C5: worst_case_idle_time {worst_case_idle_time} is below "
                f"the required lower bound {lower_bound}"
            )

    return violations
