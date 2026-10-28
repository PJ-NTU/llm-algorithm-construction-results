def objective(instance: dict, solution: dict) -> float:
    '''Return the minimization objective: schedule displacement plus expected delay.'''
    n_flights = instance['n_flights']
    n_scenarios = instance['n_scenarios']
    rho = float(instance['rho'])

    scheduled_departure = solution['scheduled_departure_period']
    requested_departure = instance['requested_departure_period']
    departure_delay = solution['departure_delay']
    arrival_delay = solution['arrival_delay']

    schedule_displacement_cost = sum(
        abs(scheduled_departure[flight] - requested_departure[flight])
        for flight in range(n_flights)
    )

    expected_operational_delay_cost = 0.0
    for scenario in range(n_scenarios):
        scenario_cost = sum(
            instance['departure_delay_cost'][flight]
            * departure_delay[scenario][flight]
            + instance['arrival_delay_cost'][flight]
            * arrival_delay[scenario][flight]
            for flight in range(n_flights)
        )
        expected_operational_delay_cost += (
            instance['scenario_probabilities'][scenario] * scenario_cost
        )

    return float(
        rho * schedule_displacement_cost
        + (1.0 - rho) * expected_operational_delay_cost
    )
