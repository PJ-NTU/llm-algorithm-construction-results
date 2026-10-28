"""Format adapter for bront2009 (choice-based network revenue management, CDLP).

Official instance: {"network": {"num_cities", "city_names", "num_legs", "legs": [{"leg_id", "origin",
                    "destination", "capacity"}]}, "products": [{"product_id", "origin_destination", "class", "fare",
                    "legs_used"}], "segments": [{"segment_id", "lambda_l", "consideration_set", "preference_vector",
                    "no_purchase_preference", "description"}], "booking_horizon": {"T"}, "alpha", "lambda", ...}.
Task format:       the same data flattened to "<block>_<field>" lists (network_legs_capacity, products_fare, ...).
                   Legs, products and segments are renumbered 0..count-1 in the official list order, and
                   products_legs_used / segments_consideration_set are rewritten with the new numbers (the official
                   ids are 1..count in list order, which is also how the official checker reads them).

Arrival rates: the official model uses p_l = lambda_l / lambda, i.e. segment l arrives at rate lambda_l per period;
the task model uses lambda * lambda_l / sum_l(lambda_l). The task's "lambda" is therefore set to sum_l(lambda_l),
which makes capacity use and revenue identical in both models.

Official solution: {"objective_value", "active_columns": [{"offer_set": [product_id, ...], "time_allocated"}]}.
Each task allocation becomes one column with its products mapped back to the official ids; the objective,
sum_S lambda * R(S) * t(S), is recomputed with the official MNL formulas.
"""


def to_ours(official_instance: dict) -> dict:
    net = official_instance["network"]
    legs = net["legs"]
    products = official_instance["products"]
    segments = official_instance["segments"]
    leg_index = {leg["leg_id"]: i for i, leg in enumerate(legs)}
    product_index = {p["product_id"]: j for j, p in enumerate(products)}
    return {
        "alpha": official_instance.get("alpha"),
        "booking_horizon_T": official_instance["booking_horizon"]["T"],
        "lambda": sum(s["lambda_l"] for s in segments),
        "network_city_names": list(net.get("city_names", [])),
        "network_legs_capacity": [leg["capacity"] for leg in legs],
        "network_legs_destination": [leg.get("destination") for leg in legs],
        "network_legs_leg_id": list(range(len(legs))),
        "network_legs_origin": [leg.get("origin") for leg in legs],
        "network_num_cities": net.get("num_cities", len(net.get("city_names", []))),
        "network_num_legs": len(legs),
        "products_class": [p.get("class") for p in products],
        "products_fare": [float(p["fare"]) for p in products],
        "products_legs_used": [[leg_index[l] for l in p["legs_used"]] for p in products],
        "products_origin_destination": [p.get("origin_destination") for p in products],
        "products_product_id": list(range(len(products))),
        "segments_consideration_set": [[product_index[j] for j in s["consideration_set"]] for s in segments],
        "segments_description": [s.get("description", "") for s in segments],
        "segments_lambda_l": [s["lambda_l"] for s in segments],
        "segments_no_purchase_preference": [s["no_purchase_preference"] for s in segments],
        "segments_preference_vector": [list(s["preference_vector"]) for s in segments],
        "segments_segment_id": list(range(len(segments))),
        "source": official_instance.get("source", ""),
    }


def to_official(official_instance: dict, our_solution: dict) -> dict:
    ids = [p["product_id"] for p in official_instance["products"]]
    columns = [{"offer_set": sorted(ids[j] for j in set(a["offer_set"])), "time_allocated": a["time_allocated"]}
               for a in our_solution["offer_set_allocations"]]
    solution = {"active_columns": columns}
    solution["objective_value"] = official_objective(official_instance, solution)
    return solution


def official_objective(official_instance: dict, official_solution: dict) -> float:
    lam = official_instance["lambda"]
    fare = {p["product_id"]: float(p["fare"]) for p in official_instance["products"]}
    segments = [(s["lambda_l"] / lam, dict(zip(s["consideration_set"], s["preference_vector"])),
                 s["no_purchase_preference"]) for s in official_instance["segments"]]
    total = 0.0
    for col in official_solution["active_columns"]:
        offer = set(col["offer_set"])
        revenue = 0.0
        for share, weights, outside in segments:
            avail = [j for j in offer if j in weights]
            if not avail:
                continue
            denom = outside + sum(weights[j] for j in avail)
            revenue += sum(fare[j] * share * weights[j] / denom for j in avail)
        total += lam * revenue * col["time_allocated"]
    return total
