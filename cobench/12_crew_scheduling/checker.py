"""Feasibility checker for the crew-scheduling solution format."""

import math


_REQUIRED_SOLUTION_FIELDS = ("crew_routes", "start_tasks", "end_tasks")


def _is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _validate_instance(instance):
    errors = []
    if not isinstance(instance, dict):
        return ["FORMAT | instance must be a dict"]

    num_tasks = instance.get("num_tasks")
    max_crews = instance.get("max_crews")
    max_duty_time = instance.get("max_duty_time")

    if not _is_integer(num_tasks) or num_tasks < 0:
        errors.append("FORMAT | instance.num_tasks must be a non-negative integer")
    if not _is_integer(max_crews) or max_crews < 0:
        errors.append("FORMAT | instance.max_crews must be a non-negative integer")
    if not _is_finite_number(max_duty_time):
        errors.append("FORMAT | instance.max_duty_time must be a finite number")

    if _is_integer(num_tasks) and num_tasks >= 0:
        for field in ("start_times", "finish_times"):
            values = instance.get(field)
            if not isinstance(values, list) or len(values) != num_tasks:
                errors.append(
                    "FORMAT | instance.%s must be a list of length num_tasks" % field
                )
            elif any(not _is_finite_number(value) for value in values):
                errors.append(
                    "FORMAT | every value in instance.%s must be a finite number"
                    % field
                )

        costs = instance.get("transition_costs")
        if not isinstance(costs, list) or len(costs) != num_tasks:
            errors.append(
                "FORMAT | instance.transition_costs must have num_tasks rows"
            )
        else:
            for row_index, row in enumerate(costs):
                if not isinstance(row, list) or len(row) != num_tasks:
                    errors.append(
                        "FORMAT | instance.transition_costs[%d] must have "
                        "num_tasks entries" % row_index
                    )
                elif any(not _is_finite_number(value) for value in row):
                    errors.append(
                        "FORMAT | every value in instance.transition_costs[%d] "
                        "must be a finite number" % row_index
                    )

        transitions = instance.get("valid_transitions")
        if not isinstance(transitions, list) or len(transitions) != num_tasks:
            errors.append(
                "FORMAT | instance.valid_transitions must have num_tasks rows"
            )
        else:
            for row_index, row in enumerate(transitions):
                if not isinstance(row, list) or len(row) != num_tasks:
                    errors.append(
                        "FORMAT | instance.valid_transitions[%d] must have "
                        "num_tasks entries" % row_index
                    )
                elif any(
                    type(value) not in (int, bool) or value not in (0, 1)
                    for value in row
                ):
                    errors.append(
                        "FORMAT | every value in instance.valid_transitions[%d] "
                        "must be boolean or 0/1" % row_index
                    )
    return errors


def _validate_solution(solution, num_tasks, max_crews):
    errors = []
    if not isinstance(solution, dict):
        return ["FORMAT | solution must be a dict"]

    for field in _REQUIRED_SOLUTION_FIELDS:
        if field not in solution:
            errors.append("FORMAT | missing required solution field: %s" % field)
    if errors:
        return errors

    routes = solution["crew_routes"]
    starts = solution["start_tasks"]
    ends = solution["end_tasks"]

    if not isinstance(routes, list) or len(routes) != max_crews:
        errors.append(
            "FORMAT | crew_routes must be a list containing exactly max_crews routes"
        )
    else:
        for crew, route in enumerate(routes):
            if not isinstance(route, list):
                errors.append("FORMAT | crew_routes[%d] must be a list" % crew)
                continue
            for position, task in enumerate(route):
                if not _is_integer(task) or not 0 <= task < num_tasks:
                    errors.append(
                        "FORMAT | crew_routes[%d][%d] must be an integer in "
                        "[0, num_tasks)" % (crew, position)
                    )

    for field, values in (("start_tasks", starts), ("end_tasks", ends)):
        if not isinstance(values, list) or len(values) != max_crews:
            errors.append(
                "FORMAT | %s must be a list of length max_crews" % field
            )
            continue
        for crew, task in enumerate(values):
            if not _is_integer(task) or (task != -1 and not 0 <= task < num_tasks):
                errors.append(
                    "FORMAT | %s[%d] must be -1 or an integer in [0, num_tasks)"
                    % (field, crew)
                )
    return errors


def _check_c1(instance, routes, starts, ends):
    """C1: every task occurs exactly once across all crew routes."""
    del starts, ends
    counts = [0] * instance["num_tasks"]
    for route in routes:
        for task in route:
            counts[task] += 1

    errors = []
    missing = [task for task, count in enumerate(counts) if count == 0]
    duplicated = [task for task, count in enumerate(counts) if count > 1]
    if missing:
        errors.append("C1 | tasks not assigned to any crew: %s" % missing)
    if duplicated:
        errors.append("C1 | tasks assigned more than once: %s" % duplicated)
    return errors


def _check_c2(instance, routes, starts, ends):
    """C2: each route is one balanced linear flow with declared endpoints."""
    del instance
    errors = []
    for crew, route in enumerate(routes):
        if not route:
            if starts[crew] != -1 or ends[crew] != -1:
                errors.append(
                    "C2 | unused crew %d must declare start_tasks and end_tasks as -1"
                    % crew
                )
            continue

        if starts[crew] != route[0]:
            errors.append(
                "C2 | crew %d start_tasks does not match the first route task" % crew
            )
        if ends[crew] != route[-1]:
            errors.append(
                "C2 | crew %d end_tasks does not match the last route task" % crew
            )

        predecessor_count = {task: 0 for task in route}
        successor_count = {task: 0 for task in route}
        for current_task, next_task in zip(route, route[1:]):
            if current_task == next_task:
                errors.append(
                    "C2 | crew %d contains a self-transition at task %d"
                    % (crew, current_task)
                )
            successor_count[current_task] += 1
            predecessor_count[next_task] += 1

        for position, task in enumerate(route):
            expected_predecessors = 0 if position == 0 else 1
            expected_successors = 0 if position == len(route) - 1 else 1
            if predecessor_count[task] != expected_predecessors:
                errors.append(
                    "C2 | crew %d task %d has %d predecessors; expected %d"
                    % (
                        crew,
                        task,
                        predecessor_count[task],
                        expected_predecessors,
                    )
                )
            if successor_count[task] != expected_successors:
                errors.append(
                    "C2 | crew %d task %d has %d successors; expected %d"
                    % (crew, task, successor_count[task], expected_successors)
                )
    return errors


def _check_c3(instance, routes, starts, ends):
    """C3: consecutive tasks assigned to one crew do not overlap."""
    del starts, ends
    errors = []
    finish_times = instance["finish_times"]
    start_times = instance["start_times"]
    for crew, route in enumerate(routes):
        for current_task, next_task in zip(route, route[1:]):
            if finish_times[current_task] > start_times[next_task]:
                errors.append(
                    "C3 | crew %d transition %d->%d overlaps: finish %s > start %s"
                    % (
                        crew,
                        current_task,
                        next_task,
                        finish_times[current_task],
                        start_times[next_task],
                    )
                )
    return errors


def _check_c4(instance, routes, starts, ends):
    """C4: every consecutive pair is allowed by valid_transitions."""
    del starts, ends
    errors = []
    valid_transitions = instance["valid_transitions"]
    for crew, route in enumerate(routes):
        for current_task, next_task in zip(route, route[1:]):
            if valid_transitions[current_task][next_task] != 1:
                errors.append(
                    "C4 | crew %d transition %d->%d is not allowed"
                    % (crew, current_task, next_task)
                )
    return errors


def _check_c5(instance, routes, starts, ends):
    """C5: recomputed duty span of every used crew is at most the limit."""
    del starts, ends
    errors = []
    for crew, route in enumerate(routes):
        if not route:
            continue
        duty_time = (
            instance["finish_times"][route[-1]]
            - instance["start_times"][route[0]]
        )
        if duty_time > instance["max_duty_time"]:
            errors.append(
                "C5 | crew %d duty time %s exceeds max_duty_time %s"
                % (crew, duty_time, instance["max_duty_time"])
            )
    return errors


def _check_c6(instance, routes, starts, ends):
    """C6: the number and indices of used crews stay within max_crews."""
    del starts, ends
    errors = []
    used_crews = [crew for crew, route in enumerate(routes) if route]
    if len(used_crews) > instance["max_crews"]:
        errors.append(
            "C6 | %d crews are used, exceeding max_crews %d"
            % (len(used_crews), instance["max_crews"])
        )
    invalid_indices = [crew for crew in used_crews if crew >= instance["max_crews"]]
    if invalid_indices:
        errors.append("C6 | used crew indices outside k < K: %s" % invalid_indices)
    return errors


def _recompute_objective(instance, routes):
    """Recompute the optional objective; never trust solution-supplied values."""
    return sum(
        instance["transition_costs"][current_task][next_task]
        for route in routes
        for current_task, next_task in zip(route, route[1:])
    )


_CONSTRAINT_CHECKS = (
    _check_c1,
    _check_c2,
    _check_c3,
    _check_c4,
    _check_c5,
    _check_c6,
)


def check_solution(instance: dict, solution: dict) -> list:
    """Return all detected violations, or an empty list for a feasible solution."""
    try:
        format_errors = _validate_instance(instance)
        if format_errors:
            return format_errors

        format_errors = _validate_solution(
            solution, instance["num_tasks"], instance["max_crews"]
        )
        if format_errors:
            return format_errors

        routes = solution["crew_routes"]
        starts = solution["start_tasks"]
        ends = solution["end_tasks"]
        violations = []
        for constraint_check in _CONSTRAINT_CHECKS:
            violations.extend(constraint_check(instance, routes, starts, ends))

        # The reported optional objective is deliberately unread.  This value is
        # derived solely from authoritative instance costs and consecutive routes.
        _recomputed_objective = _recompute_objective(instance, routes)
        del _recomputed_objective
        return violations
    except Exception as exc:  # Defensive boundary: malformed inputs must not escape.
        return [
            "FORMAT | checker could not safely evaluate the input (%s)"
            % type(exc).__name__
        ]
