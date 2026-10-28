def objective(instance, solution):
    return sum(instance['part_values'][i] * solution['selection'][i] for i in range(instance['num_parts']))

# Legacy raw utility is maximized; cycle4 scores are minimized.
_legacy_raw_objective = objective
def objective(instance, solution):
    return -_legacy_raw_objective(instance, solution)
