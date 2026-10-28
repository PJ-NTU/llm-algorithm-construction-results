# Library slot 1/3  id=alg_0922d7877f66
# score=196.445  feasible_rate=1.0
# score_vec=[218.74999999999994, 326.37999999999994, 142.98999999999998, 185.16, 259.28000000000003, 247.57999999999998, 112.03000000000002, 121.26, 165.37, 185.65000000000003]

import time
import heapq
import random

def solve(instance, time_limit_s):
    t0 = time.time()
    
    n = instance['n']
    num_res = instance['resource_types']
    arcs_raw = instance['arcs']
    arc_costs = instance['arc_costs']
    v_res = instance['vertex_resource_consumption']
    a_res = instance['arc_resource_consumption']
    L = instance['lower_bounds']
    U = instance['upper_bounds']
    
    # Pre-process graph (Input is 1-indexed for path, but 0-indexed in arrays)
    # The problem says: "vertex 1 starts, vertex n ends" and "indices start from 0"
    # Looking at the checker: path[0] == 1, path[-1] == n.
    # So the data arrays for v_res[0] corresponds to vertex 1.
    adj = [[] for _ in range(n)]
    for idx, (u_raw, v_raw) in enumerate(arcs_raw):
        # The input arcs are typically 1-indexed based on the schema description
        # but the logic must be robust. Let's assume indices in arcs are 1..n
        u, v = u_raw - 1, v_raw - 1
        adj[u].append({
            'to': v,
            'cost': arc_costs[idx],
            'res': a_res[idx],
            'idx': idx
        })
        
    source = 0 # Vertex 1
    target = n - 1 # Vertex n
    
    # Dijkstra on reverse graph for pruning
    rev_adj = [[] for _ in range(n)]
    for idx, (u_raw, v_raw) in enumerate(arcs_raw):
        u, v = u_raw - 1, v_raw - 1
        rev_adj[v].append((u, arc_costs[idx]))
        
    min_dist_to_target = [float('inf')] * n
    min_dist_to_target[target] = 0
    pq = [(0, target)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > min_dist_to_target[u]:
            continue
        for v, cost in rev_adj[u]:
            if min_dist_to_target[v] > d + cost:
                min_dist_to_target[v] = d + cost
                heapq.heappush(pq, (min_dist_to_target[v], v))

    best_cost = float('inf')
    best_path = []
    best_res_acc_at_nodes = None

    # State: (node, current_res_tuple)
    # Dominance: for a node, a state is dominated if cost is higher and ALL resources are higher
    # Using a simpler heuristic: Label-setting approach with limited states
    
    # priority queue: (estimated_total_cost, current_cost, current_node, current_res_tuple, path_tuple)
    init_res = tuple(v_res[source])
    
    # Check if source itself is feasible
    source_feasible = True
    for k in range(num_res):
        if init_res[k] < L[source][k] or init_res[k] > U[source][k]:
            source_feasible = False
    
    if source_feasible:
        # (cost, node, res_tuple, path_tuple)
        pq = [(min_dist_to_target[source], 0, source, init_res, (source + 1,))]
        
        # visited[node] = list of (res_tuple, cost)
        visited = [[] for _ in range(n)]
        
        while pq and time.time() - t0 < time_limit_s - 0.1:
            est, c_cost, u, c_res, path = heapq.heappop(pq)
            
            if c_cost + min_dist_to_target[u] >= best_cost:
                continue
            
            # Dominance check
            is_dom = False
            for seen_res, seen_cost in visited[u]:
                if seen_cost <= c_cost and all(seen_res[k] <= c_res[k] for k in range(num_res)):
                    is_dom = True
                    break
            if is_dom:
                continue
            
            # Add to visited and filter existing
            new_visited = []
            for seen_res, seen_cost in visited[u]:
                if c_cost <= seen_cost and all(c_res[k] <= seen_res[k] for k in range(num_res)):
                    continue
                new_visited.append((seen_res, seen_cost))
            new_visited.append((c_res, c_cost))
            visited[u] = new_visited

            if u == target:
                if c_cost < best_cost:
                    best_cost = c_cost
                    best_path = list(path)
                    # We'll re-calculate full accumulations for the return dict
                    continue

            for edge in adj[u]:
                v = edge['to']
                if (v + 1) in path: continue
                
                # Resource accumulation: w[j][k] >= w[i][k] + v_res[i][k] + a_res[i][j][k]
                # Also must be >= L[v][k]
                next_res = []
                possible = True
                for k in range(num_res):
                    val = c_res[k] + v_res[u][k] + edge['res'][k]
                    if val < L[v][k]: val = L[v][k]
                    if val > U[v][k]:
                        possible = False
                        break
                    next_res.append(val)
                
                if possible:
                    t_next_res = tuple(next_res)
                    new_cost = c_cost + edge['cost']
                    heapq.heappush(pq, (new_cost + min_dist_to_target[v], new_cost, v, t_next_res, path + (v + 1,)))

    # Finalize result
    res_out = {str(i + 1): [0.0] * num_res for i in range(n)}
    if best_path:
        # Re-traverse best path to get exact accumulations
        curr_res = [float(x) for x in v_res[best_path[0]-1]]
        res_out[str(best_path[0])] = list(curr_res)
        
        for i in range(len(best_path) - 1):
            u_idx = best_path[i] - 1
            v_idx = best_path[i+1] - 1
            # Find the specific arc (handle multiple arcs between same nodes)
            best_arc = None
            min_arc_cost = float('inf')
            for edge in adj[u_idx]:
                if edge['to'] == v_idx:
                    # Check if this arc was feasible
                    test_res = []
                    arc_possible = True
                    for k in range(num_res):
                        val = curr_res[k] + v_res[u_idx][k] + edge['res'][k]
                        if val < L[v_idx][k]: val = L[v_idx][k]
                        if val > U[v_idx][k]:
                            arc_possible = False
                            break
                        test_res.append(val)
                    if arc_possible:
                        # In the pulse we didn't store which arc, so we pick the cheapest feasible one
                        if edge['cost'] < min_arc_cost:
                            min_arc_cost = edge['cost']
                            best_arc = test_res
            
            if best_arc:
                curr_res = best_arc
                res_out[str(v_idx + 1)] = [float(x) for x in curr_res]

    # Ensure all vertices are in the dict as strings "1" to "n"
    for i in range(1, n + 1):
        if str(i) not in res_out:
            res_out[str(i)] = [0.0] * num_res

    return {
        "path": best_path,
        "resource_accumulations": res_out
    }