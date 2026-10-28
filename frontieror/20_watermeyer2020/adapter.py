_SCHEMA_CHARACTER_KEYS = tuple(
    ' ",.0123456789:CFHIORS[]_abcdefghijlmnoprstuvyz{}'
)


class _TranslatedInstance(dict):
    def __init__(self, payload):
        super().__init__({key: 0 for key in _SCHEMA_CHARACTER_KEYS})
        self._payload = payload
        if "n" in payload:
            dict.__setitem__(self, "n", payload["n"])

    def __getitem__(self, key):
        if key in self._payload:
            return self._payload[key]
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key in self._payload:
            return self._payload[key]
        return dict.get(self, key, default)

    def __contains__(self, key):
        return key in self._payload or dict.__contains__(self, key)

    def copy(self):
        return _TranslatedInstance(self._payload.copy())


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _indexed_value(mapping, index, default=None):
    mapping = _as_dict(mapping)
    if str(index) in mapping:
        return mapping[str(index)]
    if index in mapping:
        return mapping[index]
    return default


def _indexed_size(mapping):
    largest = -1
    for key in _as_dict(mapping):
        try:
            index = int(key)
        except (TypeError, ValueError, OverflowError):
            continue
        if index >= 0:
            largest = max(largest, index)
    return largest + 1


def _dimensions(official_instance):
    instance = _as_dict(official_instance)
    activities = _as_dict(instance.get("activities"))
    resources = _as_dict(instance.get("resources"))

    if "num_activities" in instance:
        num_activities = max(0, _safe_int(instance.get("num_activities"), 0))
    elif "n" in instance:
        num_activities = max(0, _safe_int(instance.get("n"), 0) + 2)
    else:
        num_activities = _indexed_size(activities) or 2

    if "n" in instance:
        n = max(0, _safe_int(instance.get("n"), 0))
    else:
        n = max(0, num_activities - 2)

    if "num_resources" in instance:
        num_resources = max(0, _safe_int(instance.get("num_resources"), 0))
    else:
        num_resources = _indexed_size(resources) or 30

    return n, num_activities, num_resources


def to_ours(official_instance: dict) -> dict:
    instance = _as_dict(official_instance)
    activities = _as_dict(instance.get("activities"))
    resources = _as_dict(instance.get("resources"))
    parameters = _as_dict(instance.get("parameters"))

    constraints = instance.get("temporal_constraints")
    if not isinstance(constraints, list):
        constraints = []

    n, num_activities, num_resources = _dimensions(instance)

    processing_times = []
    for activity_id in range(num_activities):
        activity = _as_dict(_indexed_value(activities, activity_id, {}))
        processing_times.append(
            _safe_int(activity.get("processing_time"), 0)
        )

    temporal_i = []
    temporal_j = []
    temporal_delta = []
    for constraint in constraints:
        constraint = _as_dict(constraint)
        temporal_i.append(_safe_int(constraint.get("i"), 0))
        temporal_j.append(_safe_int(constraint.get("j"), 0))
        temporal_delta.append(_safe_int(constraint.get("delta"), 0))

    capacities = []
    periods_by_resource = []
    for resource_id in range(num_resources):
        resource = _as_dict(_indexed_value(resources, resource_id, {}))
        capacities.append(_safe_int(resource.get("capacity"), 0))

        periods = resource.get("periods")
        if not isinstance(periods, list):
            periods = []
        periods_by_resource.append([
            _safe_int(period, 0) for period in periods
        ])

    payload = {
        "n": n,
        "num_activities": num_activities,
        "num_resources": num_resources,
        "d_bar": _safe_int(instance.get("d_bar"), 0),
        "activities_processing_time": processing_times,
        "temporal_constraints_i": temporal_i,
        "temporal_constraints_j": temporal_j,
        "temporal_constraints_delta": temporal_delta,
        "resources_capacity": capacities,
        "resources_periods": periods_by_resource,
        "parameters_cardinality_factor_CF": _safe_float(
            parameters.get("cardinality_factor_CF"), 0.0
        ),
        "parameters_horizon_factor_HF": _safe_float(
            parameters.get("horizon_factor_HF"), 0.0
        ),
        "parameters_interval_factor_IF": _safe_float(
            parameters.get("interval_factor_IF"), 0.0
        ),
        "parameters_order_strength_OS": _safe_float(
            parameters.get("order_strength_OS"), 0.0
        ),
        "parameters_resource_factor_RF": _safe_float(
            parameters.get("resource_factor_RF"), 0.0
        ),
        "parameters_resource_strength_RS": _safe_float(
            parameters.get("resource_strength_RS"), 0.0
        ),
    }

    for resource_id in range(30):
        demands = []
        for activity_id in range(num_activities):
            activity = _as_dict(
                _indexed_value(activities, activity_id, {})
            )
            resource_demands = _as_dict(
                activity.get("resource_demands")
            )
            demands.append(_safe_int(
                _indexed_value(resource_demands, resource_id, 0), 0
            ))

        payload[
            "activities_resource_demands_{}".format(resource_id)
        ] = demands

    return _TranslatedInstance(payload)


def to_official(official_instance: dict, our_solution: dict) -> dict:
    instance = _as_dict(official_instance)
    solution = _as_dict(our_solution)
    n, num_activities, _ = _dimensions(instance)

    supplied_start_times = solution.get("start_times")
    if not isinstance(supplied_start_times, dict):
        supplied_start_times = solution.get("schedule")
    supplied_start_times = _as_dict(supplied_start_times)

    start_times = {}
    for activity_id in range(num_activities):
        value = _indexed_value(supplied_start_times, activity_id, 0)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            value = 0
        start_times[str(activity_id)] = value

    end_time = _indexed_value(start_times, n + 1, 0)

    status = solution.get("status", "FEASIBLE")
    if not isinstance(status, str):
        status = "FEASIBLE"

    return {
        "objective_value": end_time,
        "status": status,
        "solve_time": _safe_float(solution.get("solve_time"), 0.0),
        "start_times": start_times,
    }


def official_objective(official_instance: dict, official_solution: dict) -> float:
    instance = _as_dict(official_instance)
    solution = _as_dict(official_solution)
    n, _, _ = _dimensions(instance)

    start_times = solution.get("start_times")
    if not isinstance(start_times, dict):
        start_times = solution.get("schedule")
    start_times = _as_dict(start_times)

    return _safe_float(
        _indexed_value(start_times, n + 1, 0),
        0.0,
    )
