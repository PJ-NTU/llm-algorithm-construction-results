"""Format adapter for mehrotra1996 (graph colouring).

Official instance: {"graph": {"num_nodes", "num_edges", "nodes": [label, ...], "edges": [[u, v], ...]}, ...}.
Task format:       {"n", "edges"}; vertex v of the task is the v-th entry of graph.nodes (the official labels are
                   0..num_nodes-1 in every instance, so the two numberings coincide).

Official solution: {"objective_value", "vertex_colors": {"<node>": colour}, "color_classes": [[node, ...], ...]}.
The task solution lists the colour of every vertex ("coloring"); colours are renumbered 0..K-1 in increasing order
of the task's colour labels, color_classes[c] lists the vertices of colour c, and the objective is K.
"""


def to_ours(official_instance: dict) -> dict:
    g = official_instance["graph"]
    index = {v: i for i, v in enumerate(g["nodes"])}
    return {"n": len(g["nodes"]), "edges": [[index[u], index[v]] for u, v in g["edges"]]}


def to_official(official_instance: dict, our_solution: dict) -> dict:
    nodes = official_instance["graph"]["nodes"]
    coloring = our_solution["coloring"][:len(nodes)]
    relabel = {c: k for k, c in enumerate(sorted(set(coloring)))}
    classes = [[] for _ in relabel]
    vertex_colors = {}
    for i, c in enumerate(coloring):
        vertex_colors[str(nodes[i])] = relabel[c]
        classes[relabel[c]].append(nodes[i])
    return {"objective_value": len(classes), "vertex_colors": vertex_colors, "color_classes": classes}


def official_objective(official_instance: dict, official_solution: dict) -> float:
    return float(len(set(official_solution["vertex_colors"].values())))
