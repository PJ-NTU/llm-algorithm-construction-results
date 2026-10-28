def objective(instance, solution):
    import math
    num_machines = instance['num_machines']
    avg_C = instance['average_machine_completion_time']
    machine_completion_times = solution['machine_completion_times']
    
    sum_sq_diff = 0.0
    for i in range(num_machines):
        C_i = machine_completion_times[str(i)]
        sum_sq_diff += (C_i - avg_C) ** 2
        
    actual_objective = (1.0 / avg_C) * math.sqrt(max(0.0, sum_sq_diff))
    return float(actual_objective)