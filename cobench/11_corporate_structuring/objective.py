def objective(instance, solution):
    target = str(instance['target'])
    net_dividends = solution['net_dividends']
    # The objective is to maximize the total net income reaching the root country.
    # Based on the solver code and description, this is F[target].
    return float(net_dividends.get(target, 0.0))

# Legacy raw utility is maximized; cycle4 scores are minimized.
_legacy_raw_objective = objective
def objective(instance, solution):
    return -_legacy_raw_objective(instance, solution)
