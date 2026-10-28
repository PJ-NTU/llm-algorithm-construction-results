def objective(instance: dict, solution: dict) -> float:
    """Return the expected total cost; smaller values are better."""
    h = instance["num_periods"]
    n = instance["num_customers"]
    m = instance["num_vehicles"]
    s_count = instance["num_scenarios"]

    setup = solution["production_setup"]
    routes = solution["vehicle_routes"]
    production = solution["production_quantities"]
    inventories = solution["end_inventories"]
    unmet = solution["unmet_demand_quantities"]

    total = instance["setup_cost"] * sum(setup)

    transportation_costs = instance["transportation_costs"]
    for period in range(h):
        for vehicle in range(m):
            route = routes[period][vehicle]
            for start, end in zip(route, route[1:]):
                total += transportation_costs[start][end]

    probabilities = instance["scenario_probabilities"]
    production_cost = instance["unit_production_cost"]
    holding_costs = instance["holding_costs"]
    penalties = instance["unmet_demand_penalties"]

    for scenario in range(s_count):
        scenario_cost = 0.0
        for period in range(h):
            scenario_cost += production_cost * production[scenario][period]
            scenario_cost += sum(
                holding_costs[location]
                * inventories[scenario][period][location]
                for location in range(n + 1)
            )
            scenario_cost += sum(
                penalties[customer] * unmet[scenario][period][customer]
                for customer in range(n)
            )
        total += probabilities[scenario] * scenario_cost

    return float(total)
