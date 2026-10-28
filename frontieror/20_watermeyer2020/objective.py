def objective(instance, solution):
    n = instance['n']
    start_times = solution['start_times']
    # The objective is the start time of the project-end activity (n+1)
    # Activity indices are 0-indexed, so the end activity is at index n+1
    return float(start_times[str(n + 1)])