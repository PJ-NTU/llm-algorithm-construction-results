def objective(instance, solution):
    profit = instance['profit']
    selection_status = solution['selection_status']
    total_profit = 0.0
    for j in range(instance['n']):
        total_profit += profit[j] * selection_status[j]
    return float(total_profit)

# Legacy raw utility is maximized; cycle4 scores are minimized.
_legacy_raw_objective = objective
def objective(instance, solution):
    return -_legacy_raw_objective(instance, solution)
