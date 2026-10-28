def objective(instance, solution):
    return float(sum(solution["bin_activation"]))