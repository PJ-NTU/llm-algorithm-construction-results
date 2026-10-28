def objective(instance, solution):
    return float(sum(solution["vertex_selection"]))