def objective(instance, solution):
    n = instance['n']
    n_machines = instance['n_machines']
    proc = instance['proc']
    perm = solution['perm']
    
    # C[k][i] is the completion time of the job at position k on machine i
    # Initialize a 2D array for completion times
    C = [[0.0 for _ in range(n_machines)] for _ in range(n)]
    
    for k in range(n):
        job_idx = perm[k]
        for i in range(n_machines):
            p_ki = proc[job_idx][i]
            
            # Time from previous machine for the same job
            prev_machine_time = C[k][i-1] if i > 0 else 0.0
            
            # Time from previous job on the same machine
            prev_job_time = C[k-1][i] if k > 0 else 0.0
            
            # Completion time is the max of both dependencies plus processing time
            C[k][i] = max(prev_machine_time, prev_job_time) + p_ki
            
    # The objective is the completion time of the last job on the last machine
    return float(C[n-1][n_machines-1])