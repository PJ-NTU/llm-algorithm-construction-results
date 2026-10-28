def objective(instance, solution):
    costs = instance['costs']
    selection_status = solution['selection_status']
    total_cost = 0.0
    for j in range(len(costs)):
        total_cost += costs[j] * selection_status[j]
    return float(total_cost)