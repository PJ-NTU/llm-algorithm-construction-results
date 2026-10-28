def objective(instance, solution):
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    jobs_data = instance['jobs']
    start_times = solution['start_times']
    
    max_completion_time = 0.0
    for j in range(n_jobs):
        # The makespan is defined by the completion time of the last operation of each job
        # Last operation index is n_machines - 1
        last_op_start = start_times[j][n_machines - 1]
        last_op_duration = jobs_data[j][n_machines - 1][1]
        completion_time = last_op_start + last_op_duration
        if completion_time > max_completion_time:
            max_completion_time = completion_time
            
    return float(max_completion_time)