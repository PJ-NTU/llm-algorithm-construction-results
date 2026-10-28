"""Feasibility checker for the RCPSP/max-pi instance described in MAT_s4.md."""

import re


_DIGIT_RUN = re.compile(r"[0-9]+")


def _is_int(value):
    """JSON booleans are not integer decision values, despite Python's bool subtype."""
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_activity_id(key):
    """Accept any key spelling containing exactly one unambiguous digit run."""
    if _is_int(key):
        return key
    if not isinstance(key, str):
        return None
    matches = _DIGIT_RUN.findall(key)
    if len(matches) != 1:
        return None
    return int(matches[0])


def _read_start_times(instance, solution, violations):
    if not isinstance(solution, dict):
        violations.append("FORMAT | solution must be a dictionary")
        return None
    if "start_times" not in solution:
        violations.append("FORMAT | missing required field 'start_times'")
        return None

    raw = solution["start_times"]
    if not isinstance(raw, dict):
        violations.append("FORMAT | 'start_times' must be a dictionary")
        return None

    n = instance.get("n") if isinstance(instance, dict) else None
    if not _is_int(n):
        violations.append("FORMAT | instance field 'n' must be an integer")
        return None

    last_activity = n + 1
    starts = {}
    ambiguous = set()
    for key, value in raw.items():
        activity = _parse_activity_id(key)
        if activity is None or activity < 0 or activity > last_activity:
            # Only entries representing activities in V are decision values. Extra
            # metadata inside the mapping is not used to judge feasibility.
            continue
        if activity in starts:
            ambiguous.add(activity)
            continue
        if not _is_int(value):
            violations.append(
                f"C1 | start time for activity {activity} must be an integer"
            )
            continue
        starts[activity] = value

    for activity in sorted(ambiguous):
        violations.append(
            f"FORMAT | multiple start_times keys parse to activity {activity}"
        )
        starts.pop(activity, None)

    missing = [activity for activity in range(n + 2) if activity not in starts]
    if missing:
        violations.append(
            "FORMAT | start_times is missing activities: "
            + ", ".join(str(activity) for activity in missing)
        )
    return starts


def _check_c1(instance, starts, violations):
    """C1: all activity starts are non-negative integers and S_0 is exactly zero."""
    if starts is None:
        return
    n = instance["n"]
    for activity in range(n + 2):
        if activity in starts and starts[activity] < 0:
            violations.append(
                f"C1 | activity {activity} has negative start time {starts[activity]}"
            )
    if 0 in starts and starts[0] != 0:
        violations.append(
            f"C1 | project-start activity 0 must start at 0, got {starts[0]}"
        )


def _integer_list(instance, field, constraint, violations):
    value = instance.get(field)
    if not isinstance(value, list):
        violations.append(f"{constraint} | instance field '{field}' must be a list")
        return None
    if any(not _is_int(item) for item in value):
        violations.append(
            f"{constraint} | instance field '{field}' must contain only integers"
        )
        return None
    return value


def _check_c2(instance, starts, violations):
    """C2: check every stated lower-bound temporal lag exactly."""
    if starts is None:
        return
    sources = _integer_list(
        instance, "temporal_constraints_i", "C2", violations
    )
    targets = _integer_list(
        instance, "temporal_constraints_j", "C2", violations
    )
    deltas = _integer_list(
        instance, "temporal_constraints_delta", "C2", violations
    )
    if sources is None or targets is None or deltas is None:
        return
    if not (len(sources) == len(targets) == len(deltas)):
        violations.append(
            "C2 | temporal constraint arrays must have the same length"
        )
        return

    for index, (source, target, delta) in enumerate(
        zip(sources, targets, deltas)
    ):
        if source not in starts or target not in starts:
            violations.append(
                f"C2 | temporal constraint {index} references an activity "
                "without a start time"
            )
            continue
        actual = starts[target] - starts[source]
        if actual < delta:
            violations.append(
                f"C2 | temporal constraint {index} ({source}->{target}) requires "
                f"a start-time difference of at least {delta}, got {actual}"
            )


def _check_c3(instance, starts, violations):
    """C3: recompute each partially renewable resource's total consumption."""
    if starts is None:
        return
    num_resources = instance.get("num_resources")
    if not _is_int(num_resources):
        violations.append("C3 | instance field 'num_resources' must be an integer")
        return

    durations = _integer_list(
        instance, "activities_processing_time", "C3", violations
    )
    capacities = _integer_list(instance, "resources_capacity", "C3", violations)
    periods = instance.get("resources_periods")
    if not isinstance(periods, list):
        violations.append("C3 | instance field 'resources_periods' must be a list")
        periods = None
    if durations is None or capacities is None or periods is None:
        return

    n = instance["n"]
    activity_count = n + 2
    if len(durations) < activity_count:
        violations.append(
            "C3 | activities_processing_time does not cover every activity"
        )
        return
    if len(capacities) < num_resources or len(periods) < num_resources:
        violations.append(
            "C3 | resource capacities/period subsets do not cover every resource"
        )
        return

    for resource in range(num_resources):
        resource_periods = periods[resource]
        if not isinstance(resource_periods, list) or any(
            not _is_int(period) for period in resource_periods
        ):
            violations.append(
                f"C3 | period subset for resource {resource} must be a list of integers"
            )
            continue

        demand_field = f"activities_resource_demands_{resource}"
        demands = _integer_list(instance, demand_field, "C3", violations)
        if demands is None:
            continue
        if len(demands) < activity_count:
            violations.append(
                f"C3 | {demand_field} does not cover every activity"
            )
            continue

        # Pi_k is a set in the description, so repeated serialized periods count once.
        period_set = set(resource_periods)
        total = 0
        for activity in range(activity_count):
            if activity not in starts:
                continue
            start = starts[activity]
            finish = start + durations[activity]
            active_periods = sum(
                1 for period in period_set if start < period <= finish
            )
            total += demands[activity] * active_periods

        # All operands and operations are integral, so exact comparison is required.
        if total > capacities[resource]:
            violations.append(
                f"C3 | resource {resource} consumes {total}, exceeding capacity "
                f"{capacities[resource]}"
            )


def check_solution(instance: dict, solution: dict) -> list:
    """Return concrete violations; an empty list means the solution is feasible."""
    violations = []
    if not isinstance(instance, dict):
        return ["FORMAT | instance must be a dictionary"]

    try:
        starts = _read_start_times(instance, solution, violations)
        _check_c1(instance, starts, violations)
        _check_c2(instance, starts, violations)
        _check_c3(instance, starts, violations)
    except Exception as exc:  # Last-resort robustness for hostile container objects.
        violations.append(
            f"FORMAT | checker could not safely read the supplied data: {type(exc).__name__}"
        )
    return violations
