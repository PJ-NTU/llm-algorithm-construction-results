def objective(instance, solution):
    cost_matrix = instance['cost_matrix']
    assignments = solution['assignments']
    total_cost = 0.0
    for job_str, machine_idx in assignments.items():
        job_idx = int(job_str)
        total_cost += cost_matrix[machine_idx][job_idx]
    return float(total_cost)