# Library slot 2/3  id=alg_9c1d885458bc
# score=10000000000.0  feasible_rate=0.0
# score_vec=[10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0, 10000000000.0]

import time
import heapq
import numpy as np

def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    
    n = instance['n']
    num_res = instance['resource_types']
    arcs_data = instance['arcs']
    arc_costs = instance['arc_costs']
    v_res = instance['vertex_resource_consumption']
    a_res = instance['arc_resource_consumption']
    L = instance['lower_bounds']
    U = instance['upper_bounds']
    
    # Adjacency list: adj[u] = [(v, cost, res_vec, arc_idx), ...]
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(arcs_data):
        adj[u].append((v, arc_costs[idx], a_res[idx], idx))
        
    source = 0
    target = n - 1
    
    # Label structure: (cost, resources, path, current_v)
    # Dominance: label A dominates B if cost_A <= cost_B and res_A <= res_B (all k)
    # and at least one inequality is strict.
    
    # labels[v] = list of (cost, tuple(resources), path)
    labels = [[] for _ in range(n)]
    
    # Initial label at source
    init_res = tuple(v_res[source][k] for k in range(num_res))
    
    # Check source feasibility
    for k in range(num_res):
        if init_res[k] < L[source][k] or init_res[k] > U[source][k]:
            return {
                "path": [],
                "resource_accumulations": {str(i): [0.0] * num_res for i in range(n)}
            }
            
    # Priority Queue: (cost, current_v, resources, path)
    pq = [(0.0, source, init_res, [source])]
    labels[source].append((0.0, init_res))
    
    best_path = []
    min_cost = float('inf')
    best_res_acc = None

    margin = 0.05
    
    while pq and (time.time() - t0 < time_limit_s - margin):
        c, u, res_u, path_u = heapq.heappop(pq)
        
        # Pruning: if already found a better path to target
        if c >= min_cost:
            continue
            
        # Explore neighbors
        for v, edge_cost, edge_res, _ in adj[u]:
            new_cost = c + edge_cost
            if new_cost >= min_cost:
                continue
                
            # Resource accumulation
            new_res = []
            possible = True
            for k in range(num_res):
                val = res_u[k] + v_res[u][k] + edge_res[k]
                # Vertex v resource constraints
                if val < L[v][k]:
                    val = L[v][k] # Lower bound is usually treated as 'at least', but here strictly w[j] >= w[i]...
                if val > U[v][k]:
                    possible = False
                    break
                new_res.append(val)
            
            if not possible:
                continue
            
            new_res_tuple = tuple(new_res)
            
            # Simple cycle prevention
            if v in path_u:
                continue
                
            # Dominance check
            dominated = False
            new_labels_v = []
            for (old_c, old_res) in labels[v]:
                # If existing label is better or equal in everything
                if old_c <= new_cost and all(old_res[k] <= new_res_tuple[k] for k in range(num_res)):
                    dominated = True
                    break
            
            if dominated:
                continue
                
            # If not dominated, add and prune existing ones dominated by new
            for (old_c, old_res) in labels[v]:
                if not (new_cost <= old_c and all(new_res_tuple[k] <= old_res[k] for k in range(num_res))):
                    new_labels_v.append((old_c, old_res))
            
            new_labels_v.append((new_cost, new_res_tuple))
            labels[v] = new_labels_v
            
            new_path = path_u + [v]
            if v == target:
                if new_cost < min_cost:
                    min_cost = new_cost
                    best_path = new_path
                    # For the target node, we must also ensure lower bounds are met (already checked in loop)
                    # We store the accumulations for the output
            
            heapq.heappush(pq, (new_cost, v, new_res_tuple, new_path))

    # Prepare final resource accumulations for the best path
    res_accumulations = {str(i): [0.0] * num_res for i in range(n)}
    if best_path:
        curr_res = [v_res[best_path[0]][k] for k in range(num_res)]
        for k in range(num_res):
            res_accumulations[str(best_path[0])][k] = curr_res[k]
            
        for idx in range(len(best_path) - 1):
            u = best_path[idx]
            v = best_path[idx+1]
            # Find the arc used
            arc_res_vec = None
            for v_adj, _, a_r, _ in adj[u]:
                if v_adj == v:
                    arc_res_vec = a_r # Simplification: taking first matching arc
                    break
            
            for k in range(num_res):
                curr_res[k] = curr_res[k] + v_res[u][k] + arc_res_vec[k]
                if curr_res[k] < L[v][k]:
                    curr_res[k] = L[v][k]
                res_accumulations[str(v)][k] = curr_res[k]

    return {
        "path": best_path,
        "resource_accumulations": res_accumulations
    }