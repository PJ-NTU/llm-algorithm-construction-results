import math


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _indexed_value(mapping, index, default):
    mapping = _as_dict(mapping)
    if str(index) in mapping:
        return mapping[str(index)]
    if index in mapping:
        return mapping[index]
    return default


def _num_tasks(instance):
    instance = _as_dict(instance)
    declared = _as_int(instance.get("num_tasks"))
    if declared is not None and declared >= 0:
        return declared

    largest = 0
    for field in ("task_processing_times", "earliest_station", "latest_station"):
        for key in _as_dict(instance.get(field)).keys():
            task = _as_int(key)
            if task is not None and task > largest:
                largest = task

    arcs = instance.get("precedence_arcs", [])
    if isinstance(arcs, (list, tuple)):
        for arc in arcs:
            if isinstance(arc, (list, tuple)):
                for endpoint in arc[:2]:
                    task = _as_int(endpoint)
                    if task is not None and task > largest:
                        largest = task
    return largest


def _instance_numbers(instance):
    instance = _as_dict(instance)
    return (
        _num_tasks(instance),
        max(0, _as_int(instance.get("cycle_time"), 0)),
        max(0, _as_int(instance.get("num_stations"), 0)),
    )


def _processing_time_map(instance):
    result = {}
    raw = _as_dict(_as_dict(instance).get("task_processing_times"))
    for task, processing_time in raw.items():
        task_number = _as_int(task)
        if task_number is not None:
            result[task_number] = _as_int(processing_time, 0)
    return result


def to_ours(official_instance: dict) -> dict:
    instance = _as_dict(official_instance)
    num_tasks, cycle_time, num_stations = _instance_numbers(instance)

    processing = _as_dict(instance.get("task_processing_times"))
    earliest = _as_dict(instance.get("earliest_station"))
    latest = _as_dict(instance.get("latest_station"))
    default_latest = num_stations if num_stations > 0 else 1

    precedence_arcs = []
    raw_arcs = instance.get("precedence_arcs", [])
    if isinstance(raw_arcs, (list, tuple)):
        for arc in raw_arcs:
            if not isinstance(arc, (list, tuple)) or len(arc) < 2:
                continue
            predecessor = _as_int(arc[0])
            successor = _as_int(arc[1])
            if predecessor is not None and successor is not None:
                precedence_arcs.append([predecessor, successor])

    return {
        "num_tasks": num_tasks,
        "processing_times": [
            _as_int(_indexed_value(processing, task, 0), 0)
            for task in range(1, num_tasks + 1)
        ],
        "cycle_time": cycle_time,
        "num_stations": num_stations,
        "precedence_arcs": precedence_arcs,
        "earliest_stations": [
            _as_int(_indexed_value(earliest, task, 1), 1)
            for task in range(1, num_tasks + 1)
        ],
        "latest_stations": [
            _as_int(_indexed_value(latest, task, default_latest), default_latest)
            for task in range(1, num_tasks + 1)
        ],
    }


def to_official(official_instance: dict, our_solution: dict) -> dict:
    instance = _as_dict(official_instance)
    solution = _as_dict(our_solution)
    num_tasks, cycle_time, num_stations = _instance_numbers(instance)
    processing_times = _processing_time_map(instance)
    earliest = _as_dict(instance.get("earliest_station"))

    raw_stations = solution.get("task_stations", [])
    if not isinstance(raw_stations, (list, tuple)):
        raw_stations = []

    task_assignments = {}
    for task in range(1, num_tasks + 1):
        fallback = _as_int(_indexed_value(earliest, task, 1), 1)
        if num_stations > 0:
            fallback = min(max(fallback, 1), num_stations)
        station = _as_int(
            raw_stations[task - 1] if task - 1 < len(raw_stations) else fallback,
            fallback,
        )
        task_assignments[str(task)] = station

    workloads = {station: 0 for station in range(1, num_stations + 1)}
    for task_text, station in task_assignments.items():
        task = _as_int(task_text)
        if task in processing_times and station in workloads:
            workloads[station] += processing_times[task]

    if num_stations > 0:
        computed_objective = max(
            0,
            max(cycle_time - workloads[station] for station in workloads),
        )
    else:
        computed_objective = 0

    objective_value = _as_float(
        solution.get("worst_case_idle_time"),
        float(computed_objective),
    )

    return {
        "objective_value": objective_value,
        "status": "candidate",
        "task_assignments": task_assignments,
        "station_workloads": {
            str(station): workloads[station]
            for station in range(1, num_stations + 1)
        },
        "station_idle_times": {
            str(station): cycle_time - workloads[station]
            for station in range(1, num_stations + 1)
        },
        "z_max": objective_value,
        "num_stations": num_stations,
        "cycle_time": cycle_time,
    }


def official_objective(official_instance: dict, official_solution: dict) -> float:
    instance = _as_dict(official_instance)
    solution = _as_dict(official_solution)
    _, cycle_time, num_stations = _instance_numbers(instance)
    processing_times = _processing_time_map(instance)
    assignments = _as_dict(solution.get("task_assignments"))

    workloads = {station: 0 for station in range(1, num_stations + 1)}
    for raw_task, raw_station in assignments.items():
        task = _as_int(raw_task)
        station = _as_int(raw_station)
        if task in processing_times and station in workloads:
            workloads[station] += processing_times[task]

    max_idle = 0.0
    for station in range(1, num_stations + 1):
        idle = float(cycle_time - workloads[station])
        if idle > max_idle:
            max_idle = idle
    return max_idle
