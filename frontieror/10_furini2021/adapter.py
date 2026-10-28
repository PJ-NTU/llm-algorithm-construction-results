"""Format adapter for furini2021 (edge interdiction clique problem).

Official instance: {"num_vertices", "num_edges", "interdiction_budget_k", "vertices", "edges", plus generator
                    metadata}. Task format: the same fields under the same names, with vertices renumbered 0..n-1 in
                    the order of "vertices" (the official labels already are 0..n-1) and edges in the official order.

Official solution: {"objective_value", "interdicted_edges": [[u, v], ...]}. The task solution flags every edge
("interdiction_status", aligned with "edges"); the flagged edges are the interdicted edges. The objective, the
clique number of the graph that remains, is recomputed exactly (branch and bound with a colouring bound), as the
official checker does.
"""

FIELDS = ("actual_edge_density", "budget_mu_percent", "budget_rule", "generation_seed", "generation_style",
          "interdiction_budget_k", "problem", "target_edge_density")


def to_ours(official_instance: dict) -> dict:
    labels = official_instance["vertices"]
    index = {v: i for i, v in enumerate(labels)}
    edges = [[index[u], index[v]] for u, v in official_instance["edges"]]
    out = {k: official_instance.get(k) for k in FIELDS}
    out.update(vertices=list(range(len(labels))), edges=edges, num_vertices=len(labels), num_edges=len(edges))
    return out


def to_official(official_instance: dict, our_solution: dict) -> dict:
    edges = official_instance["edges"]
    status = our_solution["interdiction_status"]
    cut = [list(edges[e]) for e in range(min(len(edges), len(status))) if status[e] == 1]
    solution = {"interdicted_edges": cut}
    solution["objective_value"] = int(official_objective(official_instance, solution))
    return solution


def _clique_number(n, adj):
    """Maximum clique size of a graph given as bitsets adj[v]."""
    best = 0

    def expand(cand, size):
        nonlocal best
        order, bound, rest, colour = [], [], cand, 0
        while rest:                                   # greedy colouring gives an upper bound per vertex
            colour += 1
            q = rest
            while q:
                v = (q & -q).bit_length() - 1
                q &= ~adj[v] & ~(1 << v)
                rest &= ~(1 << v)
                order.append(v)
                bound.append(colour)
        for k in range(len(order) - 1, -1, -1):
            if size + bound[k] <= best:
                return
            v = order[k]
            sub = cand & adj[v]
            if sub:
                expand(sub, size + 1)
            elif size + 1 > best:
                best = size + 1
            cand &= ~(1 << v)

    expand((1 << n) - 1, 0)
    return best


def official_objective(official_instance: dict, official_solution: dict) -> float:
    labels = official_instance["vertices"]
    index = {v: i for i, v in enumerate(labels)}
    cut = {frozenset(e) for e in official_solution["interdicted_edges"]}
    adj = [0] * len(labels)
    for u, v in official_instance["edges"]:
        if frozenset((u, v)) not in cut and u != v:
            adj[index[u]] |= 1 << index[v]
            adj[index[v]] |= 1 << index[u]
    return float(_clique_number(len(labels), adj))
