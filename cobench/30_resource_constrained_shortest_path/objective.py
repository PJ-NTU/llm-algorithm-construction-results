def objective(instance, solution):
    path = solution.get("path", [])
    if not path or len(path) < 2:
        return 0.0
    
    arcs = instance["arcs"]
    arc_costs = instance["arc_costs"]
    
    # Create a lookup for arc costs based on (start_node, end_node)
    # Note: Input arcs are 1-indexed as per the description and solver code
    arc_cost_map = {}
    for idx, (u, v) in enumerate(arcs):
        arc_cost_map[(u, v)] = arc_costs[idx]
    
    total_cost = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        total_cost += arc_cost_map.get((u, v), 0.0)
        
    return float(total_cost)