def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, (list, tuple)) else []


def _number(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if result != result or result in (float("inf"), float("-inf")):
        return float(default)
    return result


def _integer(value, default=0):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _lookup(mapping, key, default=None):
    mapping = _as_dict(mapping)
    if key in mapping:
        return mapping[key]
    text = str(key)
    if text in mapping:
        return mapping[text]
    try:
        integer_key = int(key)
    except (TypeError, ValueError, OverflowError):
        return default
    return mapping.get(integer_key, default)


def _vector_value(vector, index, default):
    vector = _as_list(vector)
    if 0 <= index < len(vector):
        return vector[index]
    return default


def _matrix_value(matrix, row, column, default):
    matrix = _as_list(matrix)
    if not (0 <= row < len(matrix)):
        return default
    values = _as_list(matrix[row])
    if not (0 <= column < len(values)):
        return default
    return values[column]


def _instance_flights(instance):
    instance = _as_dict(instance)
    raw = list(_as_list(instance.get("flights")))
    count = max(len(raw), _integer(instance.get("num_flights"), len(raw)), 0)
    flights = []
    for index in range(count):
        if index < len(raw):
            flights.append(_as_dict(raw[index]))
        else:
            flights.append({"id": index + 1})
    return flights


def _instance_airports(instance):
    instance = _as_dict(instance)
    raw = list(_as_list(instance.get("airports")))
    count = max(len(raw), _integer(instance.get("num_airports"), len(raw)), 0)
    for index in range(len(raw), count):
        raw.append("__airport_%d" % index)
    return raw


def _instance_scenarios(instance):
    instance = _as_dict(instance)
    raw = list(_as_list(instance.get("scenarios")))
    count = max(len(raw), _integer(instance.get("num_scenarios"), len(raw)), 0)
    scenarios = []
    for index in range(count):
        if index < len(raw):
            scenarios.append(_as_dict(raw[index]))
        else:
            scenarios.append({"id": index + 1})
    return scenarios


def _merge_connections(items, flight_index, target):
    for raw_connection in _as_list(items):
        connection = _as_dict(raw_connection)
        first_key = str(connection.get("flight_i"))
        second_key = str(connection.get("flight_j"))
        if first_key not in flight_index or second_key not in flight_index:
            continue
        pair = (flight_index[first_key], flight_index[second_key])
        minimum = _integer(connection.get("min_connection_time"), 0)
        if pair not in target or minimum > target[pair]:
            target[pair] = minimum


def to_ours(official_instance: dict) -> dict:
    instance = _as_dict(official_instance)
    flights = _instance_flights(instance)
    airports = _instance_airports(instance)
    scenarios = _instance_scenarios(instance)

    n_flights = len(flights)
    n_airports = len(airports)
    n_scenarios = len(scenarios)

    inferred_periods = 1
    for flight in flights:
        inferred_periods = max(
            inferred_periods,
            _integer(flight.get("dep_period"), 1),
            _integer(flight.get("arr_period"), 1),
        )
    n_periods = max(1, _integer(instance.get("num_time_periods"), inferred_periods))

    airport_index = {str(airport): index for index, airport in enumerate(airports)}
    flight_index = {}
    for index, flight in enumerate(flights):
        flight_index[str(flight.get("id", index + 1))] = index

    requested_departure = []
    requested_arrival = []
    scheduled_duration = []
    min_duration = []
    max_duration = []
    departure_airport = []
    arrival_airport = []
    max_departure_delay = []
    max_arrival_delay = []
    departure_cost = []
    arrival_cost = []

    default_dep_cost = _number(instance.get("unit_dep_delay_cost"), 0.0)
    default_arr_cost = _number(instance.get("unit_arr_delay_cost"), 0.0)

    for flight in flights:
        dep = _integer(flight.get("dep_period"), 1)
        arr = _integer(flight.get("arr_period"), dep)
        duration = _integer(flight.get("delta_sch"), arr - dep)
        minimum = _integer(flight.get("delta_min"), duration)
        maximum = _integer(flight.get("delta_max"), max(duration, minimum))

        requested_departure.append(dep)
        requested_arrival.append(arr)
        scheduled_duration.append(duration)
        min_duration.append(minimum)
        max_duration.append(maximum)
        departure_airport.append(airport_index.get(str(flight.get("dep_airport")), 0))
        arrival_airport.append(airport_index.get(str(flight.get("arr_airport")), 0))
        max_departure_delay.append(max(0, _integer(flight.get("max_dep_delay"), 0)))
        max_arrival_delay.append(max(0, _integer(flight.get("max_arr_delay"), 0)))
        departure_cost.append(max(0.0, _number(flight.get("cost_dep_delay"), default_dep_cost)))
        arrival_cost.append(max(0.0, _number(flight.get("cost_arr_delay"), default_arr_cost)))

    aircraft_connections = {}
    _merge_connections(instance.get("aircraft_connections"), flight_index, aircraft_connections)

    schedule_connections = dict(aircraft_connections)
    _merge_connections(instance.get("passenger_connections"), flight_index, schedule_connections)

    schedule_pairs = [[first, second] for first, second in schedule_connections]
    schedule_times = [schedule_connections[tuple(pair)] for pair in schedule_pairs]
    aircraft_pairs = [[first, second] for first, second in aircraft_connections]
    aircraft_times = [aircraft_connections[tuple(pair)] for pair in aircraft_pairs]

    probabilities = [max(0.0, _number(scenario.get("probability"), 0.0)) for scenario in scenarios]
    probability_sum = sum(probabilities)
    if probabilities:
        if probability_sum > 0.0:
            probabilities = [value / probability_sum for value in probabilities]
        else:
            probabilities = [1.0 / len(probabilities) for _ in probabilities]

    weather = []
    for scenario in scenarios:
        operating_conditions = _as_dict(scenario.get("operating_conditions"))
        scenario_weather = []
        for airport in airports:
            values = _as_list(operating_conditions.get(airport))
            scenario_weather.append([
                1 if str(_vector_value(values, period, "VMC")).upper() == "IMC" else 0
                for period in range(n_periods)
            ])
        weather.append(scenario_weather)

    capacity_envelopes = _as_dict(instance.get("capacity_envelopes"))
    capacity_departure = [[], []]
    capacity_arrival = [[], []]
    capacity_limits = [[], []]
    for condition_index, condition_name in enumerate(("VMC", "IMC")):
        for airport in airports:
            airport_envelope = _as_dict(capacity_envelopes.get(airport))
            segments = list(_as_list(airport_envelope.get(condition_name)))
            if not segments:
                segments = [{}, {}, {}]
            dep_coefficients = []
            arr_coefficients = []
            limits = []
            for raw_segment in segments:
                segment = _as_dict(raw_segment)
                dep_coefficients.append(_number(segment.get("a"), 0.0))
                arr_coefficients.append(_number(segment.get("b"), 0.0))
                limits.append(_number(segment.get("Q"), 0.0))
            capacity_departure[condition_index].append(dep_coefficients)
            capacity_arrival[condition_index].append(arr_coefficients)
            capacity_limits[condition_index].append(limits)

    rho = _number(instance.get("rho"), 0.0)
    rho = min(1.0, max(0.0, rho))

    return {
        "n_flights": n_flights,
        "n_airports": n_airports,
        "n_periods": n_periods,
        "n_scenarios": n_scenarios,
        "flight_departure_airport": departure_airport,
        "flight_arrival_airport": arrival_airport,
        "requested_departure_period": requested_departure,
        "requested_arrival_period": requested_arrival,
        "scheduled_enroute_duration": scheduled_duration,
        "min_enroute_duration": min_duration,
        "max_enroute_duration": max_duration,
        "max_departure_delay": max_departure_delay,
        "max_arrival_delay": max_arrival_delay,
        "departure_delay_cost": departure_cost,
        "arrival_delay_cost": arrival_cost,
        "max_schedule_displacement": max(0, _integer(instance.get("maximum_displacement_delta"), 0)),
        "schedule_connection_pairs": schedule_pairs,
        "schedule_min_connection_time": schedule_times,
        "aircraft_connection_pairs": aircraft_pairs,
        "aircraft_min_connection_time": aircraft_times,
        "scenario_probabilities": probabilities,
        "weather_condition": weather,
        "capacity_departure_coefficients": capacity_departure,
        "capacity_arrival_coefficients": capacity_arrival,
        "capacity_limits": capacity_limits,
        "rho": rho,
    }


def to_official(official_instance: dict, our_solution: dict) -> dict:
    instance = _as_dict(official_instance)
    solution = _as_dict(our_solution)
    flights = _instance_flights(instance)
    scenarios = _instance_scenarios(instance)
    delta = max(0, _integer(instance.get("maximum_displacement_delta"), 0))

    scheduled_dep_values = _as_list(solution.get("scheduled_departure_period"))
    scheduled_arr_values = _as_list(solution.get("scheduled_arrival_period"))
    operated_dep_values = _as_list(solution.get("operated_departure_period"))
    operated_arr_values = _as_list(solution.get("operated_arrival_period"))
    departure_delay_values = _as_list(solution.get("departure_delay"))
    arrival_delay_values = _as_list(solution.get("arrival_delay"))

    schedule = {}
    scheduled_dep = []
    scheduled_arr = []

    for index, flight in enumerate(flights):
        flight_id = flight.get("id", index + 1)
        requested_dep = _integer(flight.get("dep_period"), 1)
        requested_arr = _integer(flight.get("arr_period"), requested_dep)
        duration = _integer(flight.get("delta_sch"), requested_arr - requested_dep)

        dep = _integer(_vector_value(scheduled_dep_values, index, requested_dep), requested_dep)
        arr_default = dep + duration
        arr = _integer(_vector_value(scheduled_arr_values, index, arr_default), arr_default)

        scheduled_dep.append(dep)
        scheduled_arr.append(arr)
        schedule[str(flight_id)] = {
            "scheduled_dep": dep,
            "scheduled_arr": arr,
        }

    scenario_solutions = []
    for scenario_index, scenario in enumerate(scenarios):
        scenario_id = scenario.get("id", scenario_index + 1)
        x_dep = {}
        x_arr = {}
        v_dep = {}
        v_arr = {}

        for flight_index, flight in enumerate(flights):
            flight_id = flight.get("id", flight_index + 1)
            flight_key = str(flight_id)
            requested_dep = _integer(flight.get("dep_period"), 1)
            requested_arr = _integer(flight.get("arr_period"), requested_dep)
            minimum_duration = _integer(
                flight.get("delta_min"),
                _integer(flight.get("delta_sch"), requested_arr - requested_dep),
            )
            max_dep_delay = max(0, _integer(flight.get("max_dep_delay"), 0))
            max_arr_delay = max(0, _integer(flight.get("max_arr_delay"), 0))

            operated_dep = _integer(
                _matrix_value(operated_dep_values, scenario_index, flight_index, scheduled_dep[flight_index]),
                scheduled_dep[flight_index],
            )
            operated_arr = _integer(
                _matrix_value(operated_arr_values, scenario_index, flight_index, scheduled_arr[flight_index]),
                scheduled_arr[flight_index],
            )

            for period in range(
                requested_dep - delta + 1,
                requested_dep + delta + max_dep_delay + 1,
            ):
                x_dep["%s_%d" % (flight_key, period)] = 1 if period <= operated_dep else 0

            for period in range(
                requested_dep - delta + minimum_duration + 1,
                requested_arr + delta + max_arr_delay + 1,
            ):
                x_arr["%s_%d" % (flight_key, period)] = 1 if period <= operated_arr else 0

            supplied_dep_delay = _matrix_value(
                departure_delay_values, scenario_index, flight_index, None
            )
            supplied_arr_delay = _matrix_value(
                arrival_delay_values, scenario_index, flight_index, None
            )

            dep_delay_default = max(0.0, float(operated_dep - scheduled_dep[flight_index]))
            arr_delay_default = max(0.0, float(operated_arr - scheduled_arr[flight_index]))
            v_dep[flight_key] = _number(supplied_dep_delay, dep_delay_default)
            v_arr[flight_key] = _number(supplied_arr_delay, arr_delay_default)

        scenario_solutions.append({
            "scenario_id": scenario_id,
            "x_dep": x_dep,
            "x_arr": x_arr,
            "v_dep": v_dep,
            "v_arr": v_arr,
        })

    official_solution = {
        "objective_value": 0.0,
        "schedule": schedule,
        "scenario_solutions": scenario_solutions,
    }
    official_solution["objective_value"] = official_objective(instance, official_solution)
    return official_solution


def _w_at(period, requested_departure, scheduled_departure, delta):
    if period <= requested_departure - delta:
        return 1
    if period > requested_departure + delta:
        return 0
    return 1 if period <= scheduled_departure else 0


def official_objective(official_instance: dict, official_solution: dict) -> float:
    instance = _as_dict(official_instance)
    solution = _as_dict(official_solution)
    flights = _instance_flights(instance)
    delta = max(0, _integer(instance.get("maximum_displacement_delta"), 0))
    schedule = _as_dict(solution.get("schedule"))

    displacement_cost = 0.0
    for index, flight in enumerate(flights):
        flight_id = flight.get("id", index + 1)
        scheduled = _as_dict(_lookup(schedule, flight_id, {}))
        if not scheduled:
            continue
        requested_dep = _integer(flight.get("dep_period"), 1)
        scheduled_dep = _number(scheduled.get("scheduled_dep"), requested_dep)
        for period in range(
            requested_dep - delta + 1,
            requested_dep + delta + 1,
        ):
            coefficient = abs(period - requested_dep)
            displacement_cost += coefficient * (
                _w_at(period - 1, requested_dep, scheduled_dep, delta)
                - _w_at(period, requested_dep, scheduled_dep, delta)
            )

    probability_by_id = {}
    for index, scenario in enumerate(_instance_scenarios(instance)):
        scenario_id = scenario.get("id", index + 1)
        probability_by_id[str(scenario_id)] = max(
            0.0, _number(scenario.get("probability"), 0.0)
        )

    default_dep_cost = _number(instance.get("unit_dep_delay_cost"), 0.0)
    default_arr_cost = _number(instance.get("unit_arr_delay_cost"), 0.0)
    delay_cost = 0.0

    for raw_scenario_solution in _as_list(solution.get("scenario_solutions")):
        scenario_solution = _as_dict(raw_scenario_solution)
        scenario_id = scenario_solution.get("scenario_id")
        probability = probability_by_id.get(str(scenario_id), 0.0)
        if probability == 0.0:
            continue

        v_dep = _as_dict(scenario_solution.get("v_dep"))
        v_arr = _as_dict(scenario_solution.get("v_arr"))
        scenario_cost = 0.0
        for index, flight in enumerate(flights):
            flight_id = flight.get("id", index + 1)
            dep_cost = max(
                0.0, _number(flight.get("cost_dep_delay"), default_dep_cost)
            )
            arr_cost = max(
                0.0, _number(flight.get("cost_arr_delay"), default_arr_cost)
            )
            dep_delay = _number(_lookup(v_dep, flight_id, 0.0), 0.0)
            arr_delay = _number(_lookup(v_arr, flight_id, 0.0), 0.0)
            scenario_cost += dep_cost * dep_delay + arr_cost * arr_delay
        delay_cost += probability * scenario_cost

    rho = _number(instance.get("rho"), 0.0)
    rho = min(1.0, max(0.0, rho))
    return float(rho * displacement_cost + (1.0 - rho) * delay_cost)
