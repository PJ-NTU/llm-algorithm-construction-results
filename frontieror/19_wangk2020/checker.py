import math

EPS = 1e-9

_REQUIRED_INSTANCE_FIELDS = {
    'n_flights',
    'n_airports',
    'n_periods',
    'n_scenarios',
    'flight_departure_airport',
    'flight_arrival_airport',
    'requested_departure_period',
    'requested_arrival_period',
    'scheduled_enroute_duration',
    'min_enroute_duration',
    'max_enroute_duration',
    'max_departure_delay',
    'max_arrival_delay',
    'departure_delay_cost',
    'arrival_delay_cost',
    'max_schedule_displacement',
    'schedule_connection_pairs',
    'schedule_min_connection_time',
    'aircraft_connection_pairs',
    'aircraft_min_connection_time',
    'scenario_probabilities',
    'weather_condition',
    'capacity_departure_coefficients',
    'capacity_arrival_coefficients',
    'capacity_limits',
    'rho',
}

_REQUIRED_SOLUTION_FIELDS = {
    'scheduled_departure_period',
    'scheduled_arrival_period',
    'operated_departure_period',
    'operated_arrival_period',
    'departure_delay',
    'arrival_delay',
}


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _check_vector(value, expected_length, name, predicate, violations):
    if not isinstance(value, list):
        violations.append(f'structure: {name} must be a list')
        return False
    if len(value) != expected_length:
        violations.append(
            f'structure: {name} must have length {expected_length}, got {len(value)}'
        )
        return False
    valid = True
    for index, item in enumerate(value):
        if not predicate(item):
            violations.append(f'structure: {name}[{index}] has an invalid value or type')
            valid = False
    return valid


def _check_matrix(value, rows, columns, name, predicate, violations):
    if not isinstance(value, list):
        violations.append(f'structure: {name} must be a list of lists')
        return False
    if len(value) != rows:
        violations.append(
            f'structure: {name} must have {rows} rows, got {len(value)}'
        )
        return False
    valid = True
    for row_index, row in enumerate(value):
        if not isinstance(row, list):
            violations.append(f'structure: {name}[{row_index}] must be a list')
            valid = False
            continue
        if len(row) != columns:
            violations.append(
                f'structure: {name}[{row_index}] must have length {columns}, '
                f'got {len(row)}'
            )
            valid = False
            continue
        for column_index, item in enumerate(row):
            if not predicate(item):
                violations.append(
                    f'structure: {name}[{row_index}][{column_index}] '
                    'has an invalid value or type'
                )
                valid = False
    return valid


def _indicator(event_period, threshold_period):
    return 1 if event_period >= threshold_period else 0


def _drop(event_period, period):
    return _indicator(event_period, period) - _indicator(event_period, period + 1)


def check(instance: dict, solution: dict) -> list:
    violations = []

    if not isinstance(instance, dict):
        return ['instance structure: instance must be a dict']
    missing_instance = sorted(_REQUIRED_INSTANCE_FIELDS - set(instance))
    if missing_instance:
        return [
            'instance structure: missing fields: ' + ', '.join(missing_instance)
        ]

    count_names = ('n_flights', 'n_airports', 'n_periods', 'n_scenarios')
    bad_counts = [
        name
        for name in count_names
        if not _is_int(instance[name]) or instance[name] <= 0
    ]
    if bad_counts:
        return [
            'instance structure: these counts must be positive integers: '
            + ', '.join(bad_counts)
        ]

    if not isinstance(solution, dict):
        return ['structure: solution must be a dict']
    missing_solution = sorted(_REQUIRED_SOLUTION_FIELDS - set(solution))
    if missing_solution:
        return ['structure: missing solution fields: ' + ', '.join(missing_solution)]

    n_flights = instance['n_flights']
    n_airports = instance['n_airports']
    n_periods = instance['n_periods']
    n_scenarios = instance['n_scenarios']

    structure_ok = True
    structure_ok &= _check_vector(
        solution['scheduled_departure_period'],
        n_flights,
        'scheduled_departure_period',
        _is_int,
        violations,
    )
    structure_ok &= _check_vector(
        solution['scheduled_arrival_period'],
        n_flights,
        'scheduled_arrival_period',
        _is_int,
        violations,
    )
    structure_ok &= _check_matrix(
        solution['operated_departure_period'],
        n_scenarios,
        n_flights,
        'operated_departure_period',
        _is_int,
        violations,
    )
    structure_ok &= _check_matrix(
        solution['operated_arrival_period'],
        n_scenarios,
        n_flights,
        'operated_arrival_period',
        _is_int,
        violations,
    )
    structure_ok &= _check_matrix(
        solution['departure_delay'],
        n_scenarios,
        n_flights,
        'departure_delay',
        _is_number,
        violations,
    )
    structure_ok &= _check_matrix(
        solution['arrival_delay'],
        n_scenarios,
        n_flights,
        'arrival_delay',
        _is_number,
        violations,
    )
    if not structure_ok:
        return violations

    scheduled_departure = solution['scheduled_departure_period']
    scheduled_arrival = solution['scheduled_arrival_period']
    operated_departure = solution['operated_departure_period']
    operated_arrival = solution['operated_arrival_period']
    departure_delay = solution['departure_delay']
    arrival_delay = solution['arrival_delay']

    # Decision-variable domains: delay variables are finite and nonnegative.
    for scenario in range(n_scenarios):
        for flight in range(n_flights):
            if departure_delay[scenario][flight] < -EPS:
                violations.append(
                    f'domain: departure_delay[{scenario}][{flight}] is negative'
                )
            if arrival_delay[scenario][flight] < -EPS:
                violations.append(
                    f'domain: arrival_delay[{scenario}][{flight}] is negative'
                )

    # C1: scheduled indicator sequences start at one and end at the zero sentinel.
    for field_name, periods in (
        ('scheduled_departure_period', scheduled_departure),
        ('scheduled_arrival_period', scheduled_arrival),
    ):
        for flight, event_period in enumerate(periods):
            if (
                _indicator(event_period, 1) != 1
                or _indicator(event_period, n_periods + 1) != 0
            ):
                violations.append(
                    f'C1: {field_name}[{flight}]={event_period} is outside '
                    f'the actual period range 1..{n_periods}'
                )

    # C2: reconstructed scheduled indicators are non-increasing in time.
    for field_name, periods in (
        ('scheduled_departure_period', scheduled_departure),
        ('scheduled_arrival_period', scheduled_arrival),
    ):
        for flight, event_period in enumerate(periods):
            for period in range(1, n_periods + 1):
                if _indicator(event_period, period) < _indicator(
                    event_period, period + 1
                ):
                    violations.append(
                        f'C2: {field_name}[{flight}] produces an increasing '
                        f'indicator between periods {period} and {period + 1}'
                    )

    # C3: operated indicator sequences start at one and end at the zero sentinel.
    for field_name, matrix in (
        ('operated_departure_period', operated_departure),
        ('operated_arrival_period', operated_arrival),
    ):
        for scenario in range(n_scenarios):
            for flight in range(n_flights):
                event_period = matrix[scenario][flight]
                if (
                    _indicator(event_period, 1) != 1
                    or _indicator(event_period, n_periods + 1) != 0
                ):
                    violations.append(
                        f'C3: {field_name}[{scenario}][{flight}]={event_period} '
                        f'is outside the actual period range 1..{n_periods}'
                    )

    # C4: reconstructed operated indicators are non-increasing in time.
    for field_name, matrix in (
        ('operated_departure_period', operated_departure),
        ('operated_arrival_period', operated_arrival),
    ):
        for scenario in range(n_scenarios):
            for flight in range(n_flights):
                event_period = matrix[scenario][flight]
                for period in range(1, n_periods + 1):
                    if _indicator(event_period, period) < _indicator(
                        event_period, period + 1
                    ):
                        violations.append(
                            f'C4: {field_name}[{scenario}][{flight}] produces '
                            f'an increasing indicator between periods {period} '
                            f'and {period + 1}'
                        )

    # C5: scheduled departure displacement is bounded by the common limit.
    max_displacement = instance['max_schedule_displacement']
    requested_departure = instance['requested_departure_period']
    for flight in range(n_flights):
        displacement = scheduled_departure[flight] - requested_departure[flight]
        if displacement < -max_displacement or displacement > max_displacement:
            violations.append(
                f'C5: flight {flight} has scheduled departure displacement '
                f'{displacement}, outside [-{max_displacement}, {max_displacement}]'
            )

    # C6: scheduled arrival minus scheduled departure equals scheduled duration.
    scheduled_duration = instance['scheduled_enroute_duration']
    for flight in range(n_flights):
        actual_difference = scheduled_arrival[flight] - scheduled_departure[flight]
        if actual_difference != scheduled_duration[flight]:
            violations.append(
                f'C6: flight {flight} has scheduled duration {actual_difference}, '
                f'expected {scheduled_duration[flight]}'
            )

    # C7: all first-stage schedule connections meet their minimum connection time.
    for connection_index, pair in enumerate(instance['schedule_connection_pairs']):
        first_flight, second_flight = pair
        minimum = instance['schedule_min_connection_time'][connection_index]
        separation = (
            scheduled_departure[second_flight] - scheduled_arrival[first_flight]
        )
        if separation < minimum:
            violations.append(
                f'C7: schedule connection {connection_index} ({first_flight}, '
                f'{second_flight}) has separation {separation}, below {minimum}'
            )

    # C8: departure delay equals operated departure minus scheduled departure.
    for scenario in range(n_scenarios):
        for flight in range(n_flights):
            required_delay = (
                operated_departure[scenario][flight] - scheduled_departure[flight]
            )
            if abs(departure_delay[scenario][flight] - required_delay) > EPS:
                violations.append(
                    f'C8: departure_delay[{scenario}][{flight}] is '
                    f'{departure_delay[scenario][flight]}, expected {required_delay}'
                )

    # C9: arrival delay is at least operated arrival minus scheduled arrival.
    for scenario in range(n_scenarios):
        for flight in range(n_flights):
            lower_bound = (
                operated_arrival[scenario][flight] - scheduled_arrival[flight]
            )
            if arrival_delay[scenario][flight] + EPS < lower_bound:
                violations.append(
                    f'C9: arrival_delay[{scenario}][{flight}] is '
                    f'{arrival_delay[scenario][flight]}, below {lower_bound}'
                )

    # C10: departure delay does not exceed its flight-specific maximum.
    for scenario in range(n_scenarios):
        for flight in range(n_flights):
            maximum = instance['max_departure_delay'][flight]
            if departure_delay[scenario][flight] > maximum + EPS:
                violations.append(
                    f'C10: departure_delay[{scenario}][{flight}] is '
                    f'{departure_delay[scenario][flight]}, above {maximum}'
                )

    # C11: arrival delay does not exceed its flight-specific maximum.
    for scenario in range(n_scenarios):
        for flight in range(n_flights):
            maximum = instance['max_arrival_delay'][flight]
            if arrival_delay[scenario][flight] > maximum + EPS:
                violations.append(
                    f'C11: arrival_delay[{scenario}][{flight}] is '
                    f'{arrival_delay[scenario][flight]}, above {maximum}'
                )

    # C12: aircraft-only operated connections meet their minimum connection time.
    for connection_index, pair in enumerate(instance['aircraft_connection_pairs']):
        first_flight, second_flight = pair
        minimum = instance['aircraft_min_connection_time'][connection_index]
        for scenario in range(n_scenarios):
            separation = (
                operated_departure[scenario][second_flight]
                - operated_arrival[scenario][first_flight]
            )
            if separation < minimum:
                violations.append(
                    f'C12: scenario {scenario}, aircraft connection '
                    f'{connection_index} ({first_flight}, {second_flight}) has '
                    f'separation {separation}, below {minimum}'
                )

    # C13: operated en-route duration lies within its flight-specific bounds.
    for scenario in range(n_scenarios):
        for flight in range(n_flights):
            duration = (
                operated_arrival[scenario][flight]
                - operated_departure[scenario][flight]
            )
            minimum = instance['min_enroute_duration'][flight]
            maximum = instance['max_enroute_duration'][flight]
            if duration < minimum or duration > maximum:
                violations.append(
                    f'C13: scenario {scenario}, flight {flight} has operated '
                    f'duration {duration}, outside [{minimum}, {maximum}]'
                )

    # C14: every active airport-capacity-envelope segment is respected.
    departure_airport = instance['flight_departure_airport']
    arrival_airport = instance['flight_arrival_airport']
    weather = instance['weather_condition']
    departure_coefficients = instance['capacity_departure_coefficients']
    arrival_coefficients = instance['capacity_arrival_coefficients']
    capacity_limits = instance['capacity_limits']

    for scenario in range(n_scenarios):
        for airport in range(n_airports):
            for period_index in range(n_periods):
                period = period_index + 1
                condition = weather[scenario][airport][period_index]
                departure_count = sum(
                    _drop(operated_departure[scenario][flight], period)
                    for flight in range(n_flights)
                    if departure_airport[flight] == airport
                )
                arrival_count = sum(
                    _drop(operated_arrival[scenario][flight], period)
                    for flight in range(n_flights)
                    if arrival_airport[flight] == airport
                )
                limits = capacity_limits[condition][airport]
                for segment, limit in enumerate(limits):
                    lhs = (
                        departure_coefficients[condition][airport][segment]
                        * departure_count
                        + arrival_coefficients[condition][airport][segment]
                        * arrival_count
                    )
                    if lhs > limit + EPS:
                        violations.append(
                            f'C14: scenario {scenario}, airport {airport}, period '
                            f'{period}, condition {condition}, segment {segment} '
                            f'has load {lhs}, above capacity {limit}'
                        )

    return violations
