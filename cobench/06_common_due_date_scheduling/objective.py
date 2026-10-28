def objective(instance, solution):
    n = instance['n']
    due_date = instance['due_date']
    earliness_penalties = instance['earliness_penalties']
    tardiness_penalties = instance['tardiness_penalties']
    processing_times = instance['processing_times']
    
    order = solution['order']
    
    # Calculate completion times based on the order
    completion_times_by_job = {}
    current_time = 0
    for job_index in order:
        current_time += processing_times[job_index]
        completion_times_by_job[job_index] = current_time
        
    total_cost = 0.0
    for j in range(n):
        c_j = completion_times_by_job[j]
        earliness = max(0.0, float(due_date - c_j))
        tardiness = max(0.0, float(c_j - due_date))
        total_cost += earliness_penalties[j] * earliness + tardiness_penalties[j] * tardiness
        
    return float(total_cost)