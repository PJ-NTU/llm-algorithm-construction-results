def objective(instance, solution):
    p3 = instance['main_processing_times']
    start_times = solution['start_times']
    
    max_completion_time = 0.0
    for j in range(instance['num_jobs']):
        job_key = f"job_{j}"
        # s3 is the third element in the start_times list for each job
        s3_j = start_times[job_key][2]
        completion_time = s3_j + p3[j]
        if completion_time > max_completion_time:
            max_completion_time = completion_time
            
    return float(max_completion_time)