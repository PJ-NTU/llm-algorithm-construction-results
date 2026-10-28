"""Format adapter for carvalho1999 (one-dimensional bin packing).

Official instance: {"bin_capacity", "num_items", "items": [size, ...]}.
Task format:       {"n", "capacity", "weights"}. Items are listed by decreasing size (ties in official order), i.e.
                   grouped by size the way the official checker and the arc-flow model read them; task item k is
                   official item order[k]. This is the order the recorded results were produced with; several
                   algorithms are order-sensitive (e.g. alg_9775d125b4 on large_1: 8409 bins sorted, about 9100 in the
                   official order).

Official solution: {"objective_value", "num_bins", "bin_assignments": [{"items": [sizes], "total_size"}, ...]}, one
entry per used bin. The task solution gives the bin index of every item ("item_assignments"); every bin that
receives at least one item becomes one official bin, in increasing bin index. The objective is the number of bins.
"""


def _order(official_instance):
    items = official_instance["items"]
    return sorted(range(len(items)), key=lambda i: -items[i])


def to_ours(official_instance: dict) -> dict:
    items = official_instance["items"]
    return {"n": len(items), "capacity": int(official_instance["bin_capacity"]),
            "weights": [int(items[i]) for i in _order(official_instance)]}


def to_official(official_instance: dict, our_solution: dict) -> dict:
    items = official_instance["items"]
    bins = {}
    for k, b in zip(_order(official_instance), our_solution["item_assignments"]):
        bins.setdefault(b, []).append(items[k])
    bin_assignments = [{"items": bins[b], "total_size": sum(bins[b])} for b in sorted(bins)]
    return {"objective_value": len(bin_assignments), "num_bins": len(bin_assignments),
            "bin_assignments": bin_assignments}


def official_objective(official_instance: dict, official_solution: dict) -> float:
    return float(len(official_solution["bin_assignments"]))
