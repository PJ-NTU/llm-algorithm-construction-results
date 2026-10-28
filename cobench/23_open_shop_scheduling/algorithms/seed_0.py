import time
import random
import copy

def solve(instance, time_limit_s):
    t0 = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    ptimes = instance['ptimes']
    
    def get_schedule(job_sequences, mach_sequences):
        """
        Builds a feasible semi-active schedule using a Gift-Wrapping/Simulation approach.
        A job_sequence[j] is the order of machines for job j.
        A mach_sequence[m] is the order of jobs for machine m.
        """
        start = [[0] * n_machines for _ in range(n_jobs)]
        completion = [[0] * n_machines for _ in range(n_jobs)]
        
        # Pointers for which operation is next in each sequence
        j_ptr = [0] * n_jobs
        m_ptr = [0] * n_machines
        
        # Earliest time job/machine is free
        job_ready = [0] * n_jobs
        mach_ready = [0] * n_machines
        
        finished_count = 0
        total_ops = n_jobs * n_machines
        
        # We simulate step-by-step to resolve dependencies.
        # In Open Shop, if we follow fixed job-machine priorities, deadlock can occur 
        # unless we are careful. However, simply picking any operation (j, m) that 
        # is next in both its job sequence and machine sequence works if the sequences 
        # are consistent. A safer way is a list-scheduling approach.
        
        # To ensure we don't deadlock and always produce a feasible schedule:
        # We iterate until all operations are scheduled.
        while finished_count < total_ops:
            progress = False
            for j in range(n_jobs):
                if j_ptr[j] < n_machines:
                    m = job_sequences[j][j_ptr[j]]
                    if mach_sequences[m][m_ptr[m]] == j:
                        # This operation is the next for both job j and machine m
                        st = max(job_ready[j], mach_ready[m])
                        start[j][m] = st
                        dur = ptimes[j][m]
                        finish = st + dur
                        completion[j][m] = finish
                        job_ready[j] = finish
                        mach_ready[m] = finish
                        j_ptr[j] += 1
                        m_ptr[m] += 1
                        finished_count += 1
                        progress = True
            
            if not progress:
                # Deadlock detected due to incompatible sequences. 
                # This can happen in Open Shop if we use two independent permutations.
                # Repair: If stalled, force the next available op for some job.
                # But for Open Shop, a simpler way to get a feasible schedule is 
                # to just use a fixed priority for machines and a permutation for jobs.
                return None, None
                
        cmax = max(max(row) for row in completion) if completion else 0
        return cmax, start

    def get_random_valid_sequences():
        # To avoid deadlocks, we can use a "Latin Square" inspired initial state 
        # or simply ensure mach_sequences are derived from job_sequences.
        # However, a robust way is to use a priority-based list scheduler.
        job_seqs = [list(range(n_machines)) for _ in range(n_jobs)]
        for s in job_seqs:
            random.shuffle(s)
        
        # Instead of a random mach_seq, we use a construction that avoids deadlocks.
        # A simple way: machine m processes jobs in order 0, 1, ..., n-1 (shifted).
        mach_seqs = [[] for _ in range(n_machines)]
        for m in range(n_machines):
            mach_seqs[m] = [(m + j) % n_jobs for j in range(n_jobs)]
            random.shuffle(mach_seqs[m])
            
        return job_seqs, mach_seqs

    def list_schedule(job_priority_list):
        """
        A robust list scheduler for Open Shop. 
        job_priority_list: a permutation of all N*M operations.
        """
        start = [[-1] * n_machines for _ in range(n_jobs)]
        job_ready = [0] * n_jobs
        mach_ready = [0] * n_machines
        
        # We use a simple greedy: for each operation in the priority list,
        # schedule it as early as possible given its job and machine availability.
        # Note: This specific greedy doesn't guarantee the best, but it's ALWAYS feasible.
        # However, to be truly competitive, we should consider the "available" ops.
        
        ops = job_priority_list # list of (j, m)
        scheduled = [[False] * n_machines for _ in range(n_jobs)]
        
        # To respect the "no overlap" constraints, we must schedule ops such that
        # at any time t, a job is on at most one machine and a machine handles one job.
        # The simplest way to do this is to simulate time or use a greedy insertion.
        
        for j, m in ops:
            st = max(job_ready[j], mach_ready[m])
            # But wait, this doesn't work if we schedule an op that starts at 10
            # then later schedule an op that should have started at 5.
            # We must schedule in increasing order of start times.
            pass
        
        # Correct Greedy:
        start = [[0] * n_machines for _ in range(n_jobs)]
        j_busy_until = [0] * n_jobs
        m_busy_until = [0] * n_machines
        
        # We'll use a simple "Available Task First" strategy
        remaining_ops = set((j, m) for j in range(n_jobs) for m in range(n_machines))
        while remaining_ops:
            best_op = None
            earliest_start = float('inf')
            
            # Heuristic: pick the operation that can start earliest
            for j, m in remaining_ops:
                t = max(j_busy_until[j], m_busy_until[m])
                if t < earliest_start:
                    earliest_start = t
                    best_op = (j, m)
            
            j, m = best_op
            start[j][m] = earliest_start
            finish = earliest_start + ptimes[j][m]
            j_busy_until[j] = finish
            m_busy_until[m] = finish
            remaining_ops.remove(best_op)
            
        cmax = max(j_busy_until)
        return cmax, start

    # Initial solution
    best_cmax = float('inf')
    best_start = None

    margin = 0.05
    while time.time() - t0 < time_limit_s - margin:
        # Create a priority sequence
        ops = [(j, m) for j in range(n_jobs) for m in range(n_machines)]
        random.shuffle(ops)
        
        # We can improve the greedy by trying different orderings
        # 1. Shortest Processing Time first
        # 2. Longest Processing Time first
        # 3. Random
        
        # Let's use a Randomized Priority Rule
        # Assign random weights to each operation, then sort.
        weights = {}
        for j in range(n_jobs):
            for m in range(n_machines):
                # Mix of LPT and randomness
                weights[(j, m)] = ptimes[j][m] * random.uniform(0.5, 2.0) + random.uniform(0, 10)
        
        sorted_ops = sorted(ops, key=lambda x: weights[x], reverse=True)
        
        # Generate schedule
        curr_start = [[0] * n_machines for _ in range(n_jobs)]
        j_busy_until = [0] * n_jobs
        m_busy_until = [0] * n_machines
        
        remaining = set(ops)
        while remaining:
            best_op = None
            earliest_start = float('inf')
            
            # To make it fast, we only check a subset of remaining ops
            # or use the pre-sorted priority to break ties.
            # But the most important thing is feasibility.
            
            # Simple list scheduling: find op that can start earliest.
            # If multiple, use the priority weights.
            candidates = []
            min_t = float('inf')
            
            for j, m in remaining:
                t = max(j_busy_until[j], m_busy_until[m])
                if t < min_t:
                    min_t = t
                    candidates = [(j, m)]
                elif t == min_t:
                    candidates.append((j, m))
            
            # Break ties using weights
            best_op = max(candidates, key=lambda x: weights[x])
            
            j, m = best_op
            curr_start[j][m] = min_t
            finish = min_t + ptimes[j][m]
            j_busy_until[j] = finish
            m_busy_until[m] = finish
            remaining.remove(best_op)
            
        curr_cmax = max(j_busy_until)
        
        if curr_cmax < best_cmax:
            best_cmax = curr_cmax
            best_start = copy.deepcopy(curr_start)

    return {"start": best_start}