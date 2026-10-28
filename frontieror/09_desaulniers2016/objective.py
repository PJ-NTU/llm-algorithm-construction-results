def objective(instance: dict, solution: dict) -> float:
    travel_costs = instance['travel_costs']
    holding_costs = instance['holding_costs']
    routes = solution['routes']
    ending = solution['ending_inventories']

    travel_total = 0.0
    for period_routes in routes:
        for route in period_routes:
            if not route:
                continue
            travel_total += travel_costs[0][route[0]]
            for origin, destination in zip(route, route[1:]):
                travel_total += travel_costs[origin][destination]
            travel_total += travel_costs[route[-1]][0]

    holding_total = 0.0
    for period_inventory in ending:
        for location, inventory in enumerate(period_inventory):
            holding_total += holding_costs[location] * inventory

    return float(travel_total + holding_total)
