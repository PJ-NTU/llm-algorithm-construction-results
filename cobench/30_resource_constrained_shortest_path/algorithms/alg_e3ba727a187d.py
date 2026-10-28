# Library slot 3/3  id=alg_e3ba727a187d
# score=10000000000.0  feasible_rate=0.0
# score_vec=[10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0]

import time
import heapq
import random

def solve(instance, time_limit_s):
    t0 = time.time()
    n = instance["n"]
    k_res = instance["resource_types"]
    arcs_raw = instance["arcs"]
    arc_costs = instance["arc_costs"]
    v_res = instance["vertex_resource_consumption"]
    a_res = instance["arc_resource_consumption"]
    l_bounds = instance["lower_bounds"]
    u_bounds = instance["upper_bounds"]

    # Adjacency list and cost/resource mapping
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(arcs_raw):
        adj[u].append((v, arc_costs[idx], a_res[idx]))

    def get_path_info(path):
        """Calculates total cost and resource consumption along a path."""
        if not path:
            return float('inf'), None
        
        current_res = [0.0] * k_res
        # Initial resource at start node
        for k in range(k_res):
            current_res[k] = v_res[path[0]][k]
            if current_res[k] < l_bounds[path[0]][k]:
                current_res[k] = l_bounds[path[0]][k]
            if current_res[k] > u_bounds[path[0]][k]:
                return float('inf'), None
        
        accumulations = {str(i): [0.0] * k_res for i in range(n)}
        accumulations[str(path[0])] = list(current_res)
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            # Find arc
            found = False
            for neighbor, cost, res_inc in adj[u]:
                if neighbor == v:
                    total_cost += cost
                    for k in range(k_res):
                        current_res[k] += res_inc[k] + v_res[v][k]
                        if current_res[k] < l_bounds[v][k]:
                            current_res[k] = l_bounds[v][k]
                        if current_res[k] > u_bounds[v][k]:
                            return float('inf'), None
                    accumulations[str(v)] = list(current_res)
                    found = True
                    break
            if not found:
                return float('inf'), None
        return total_cost, accumulations

    def shortest_path(multipliers):
        """Standard Dijkstra using modified costs (Lagrangian relaxation)."""
        dist = [float('inf')] * n
        parent = [-1] * n
        dist[0] = 0
        pq = [(0, 0)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, cost, res_inc in adj[u]:
                # Modified cost = original_cost + sum(multiplier * violation_contribution)
                # Violation contribution here is simplified to resource consumption
                mod_cost = cost
                for k in range(k_res):
                    mod_cost += multipliers[k] * (res_inc[k] + v_res[v][k])
                
                if dist[u] + mod_cost < dist[v]:
                    dist[v] = dist[u] + mod_cost
                    parent[v] = u
                    heapq.heappush(pq, (dist[v], v))
        
        if parent[n-1] == -1:
            return []
        path = []
        curr = n-1
        while curr != -1:
            path.append(curr)
            curr = parent[curr]
        return path[::-1]

    best_path = []
    best_cost = float('inf')
    best_accumulations = {str(i): [0.0] * k_res for i in range(n)}
    
    # Lagrangian Multipliers
    multipliers = [0.0] * k_res
    step_size = 1.0
    
    margin = 0.05
    while time.time() - t0 < time_limit_s - margin:
        # 1. Solve relaxed problem
        path = shortest_path(multipliers)
        
        if path:
            # 2. Evaluate and try to improve via Local Search (Node Reinsertion)
            cost, accs = get_path_info(path)
            
            if cost < float('inf'):
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    best_accumulations = accs
            
            # Local Search: Try removing a node and re-routing
            if len(path) > 3:
                for _ in range(5): # Small fixed number of LS steps to keep loop tight
                    idx = random.randint(1, len(path) - 2)
                    new_path = path[:idx] + path[idx+1:]
                    c, a = get_path_info(new_path)
                    if c < cost:
                        path = new_path
                        cost, accs = c, a
                        if cost < best_cost:
                            best_cost = cost
                            best_path = path
                            best_accumulations = accs

            # 3. Update Multipliers based on feasibility of 'path'
            # (Heuristic update: increase penalty for resources that tend to exceed bounds)
            # We don't have a perfect subgradient here, but we can nudge multipliers
            res_usage = [0.0] * k_res
            curr_path_feasible = True
            temp_res = list(v_res[path[0]])
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                for neighbor, cost_arc, res_inc in adj[u]:
                    if neighbor == v:
                        for k in range(k_res):
                            temp_res[k] += res_inc[k] + v_res[v][k]
                        break
                for k in range(k_res):
                    if temp_res[k] > u_bounds[path[i+1]][k]:
                        multipliers[k] += step_size * (temp_res[k] - u_bounds[path[i+1]][k])
                        curr_path_feasible = False
            
            if curr_path_feasible:
                step_size *= 0.95 # Convergence
            else:
                step_size *= 1.05 # Diversification
        else:
            # If no path found, jitter multipliers
            for k in range(k_res):
                multipliers[k] = random.uniform(0, 100)
            step_size = 1.0

        # Periodic restart of multipliers to explore new regions
        if random.random() < 0.05:
            multipliers = [random.uniform(0, 10) for _ in range(k_res)]

    # Final check on best_accumulations to ensure all vertices are present
    if not best_path:
        return {"path": [], "resource_accumulations": {str(i): [0.0] * k_res for i in range(n)}}
    
    # Ensure every node is in the output dict as per schema
    final_accs = {str(i): [0.0] * k_res for i in range(n)}
    for node_str, vals in best_accumulations.items():
        final_accs[node_str] = vals

    return {
        "path": best_path,
        "resource_accumulations": final_accs
    }