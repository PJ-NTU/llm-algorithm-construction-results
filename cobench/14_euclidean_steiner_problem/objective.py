def objective(instance, solution):
    import math
    
    terminals = instance['points']
    steiner_points = solution['steiner_points']
    edges = solution['edges']
    
    # Combine all points into a single list
    # Indices 0 to N-1 are terminals, N to N+M-1 are Steiner points
    all_points = terminals + steiner_points
    
    total_length = 0.0
    for u_idx, v_idx in edges:
        p1 = all_points[u_idx]
        p2 = all_points[v_idx]
        
        # Euclidean distance: sqrt((x1-x2)^2 + (y1-y2)^2)
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        total_length += dist
        
    return float(total_length)