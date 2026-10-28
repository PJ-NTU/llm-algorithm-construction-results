def objective(instance, solution):
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    ptimes = instance['ptimes']
    start_times = solution['start']
    
    cmax = 0.0
    for j in range(n_jobs):
        for m in range(n_machines):
            completion_time = start_times[j][m] + ptimes[j][m]
            if completion_time > cmax:
                cmax = float(completion_time)
                
    return cmax