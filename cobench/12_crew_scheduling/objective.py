def objective(instance, solution):
    transition_costs = instance['transition_costs']
    crew_routes = solution['crew_routes']
    
    total_cost = 0.0
    for route in crew_routes:
        if not route or len(route) < 2:
            continue
        for i in range(len(route) - 1):
            u = route[i]
            v = route[i+1]
            total_cost += transition_costs[u][v]
            
    return float(total_cost)