"""Feasibility checker for the choice-based network revenue problem."""

import math


_REL = 1e-9


def _leq(a, b):
    """Relative-tolerance comparison for values produced by float arithmetic."""
    if not math.isfinite(a) or not math.isfinite(b):
        return False
    return a - b <= _REL * max(1.0, abs(a), abs(b))


def _is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fsum(values):
    """Sum floats without letting adversarial magnitudes escape as an exception."""
    try:
        return math.fsum(values)
    except (OverflowError, ValueError):
        return math.inf


def _instance_data(instance):
    """Read only the data needed to recompute C1 and C2 safely."""
    if not isinstance(instance, dict):
        return None, ["INSTANCE | instance must be an object"]

    required = (
        "lambda",
        "booking_horizon_T",
        "network_legs_capacity",
        "network_legs_leg_id",
        "products_product_id",
        "products_legs_used",
        "segments_lambda_l",
        "segments_consideration_set",
        "segments_preference_vector",
        "segments_no_purchase_preference",
    )
    missing = [key for key in required if key not in instance]
    if missing:
        return None, ["INSTANCE | missing data needed by the described model: " + ", ".join(missing)]

    arrival_rate = instance["lambda"]
    horizon = instance["booking_horizon_T"]
    leg_ids = instance["network_legs_leg_id"]
    capacities = instance["network_legs_capacity"]
    product_ids = instance["products_product_id"]
    product_legs = instance["products_legs_used"]
    segment_rates = instance["segments_lambda_l"]
    consideration_sets = instance["segments_consideration_set"]
    preferences = instance["segments_preference_vector"]
    no_purchase = instance["segments_no_purchase_preference"]

    if not _is_finite_number(arrival_rate) or not _is_finite_number(horizon):
        return None, ["INSTANCE | lambda and booking_horizon_T must be finite numbers to evaluate C1/C2"]

    arrays = (leg_ids, capacities, product_ids, product_legs, segment_rates,
              consideration_sets, preferences, no_purchase)
    if any(not isinstance(values, list) for values in arrays):
        return None, ["INSTANCE | model axis fields needed by C1 must be lists"]
    if len(leg_ids) != len(capacities):
        return None, ["INSTANCE | leg ids and capacities have different lengths"]
    if len(product_ids) != len(product_legs):
        return None, ["INSTANCE | product ids and products_legs_used have different lengths"]
    if not (len(segment_rates) == len(consideration_sets) == len(preferences) == len(no_purchase)):
        return None, ["INSTANCE | segment axis fields have different lengths"]
    if any(not _is_finite_number(value) for value in capacities):
        return None, ["INSTANCE | leg capacities must be finite numbers to evaluate C1"]
    if any(not _is_finite_number(value) for value in segment_rates):
        return None, ["INSTANCE | segment_lambda_l values must be finite numbers to evaluate C1"]
    if any(not _is_finite_number(value) for value in no_purchase):
        return None, ["INSTANCE | no-purchase preferences must be finite numbers to evaluate C1"]

    try:
        leg_index = {leg_id: index for index, leg_id in enumerate(leg_ids)}
        product_index = {product_id: index for index, product_id in enumerate(product_ids)}
    except TypeError:
        return None, ["INSTANCE | leg and product ids must be usable as identifiers"]
    if len(leg_index) != len(leg_ids) or len(product_index) != len(product_ids):
        return None, ["INSTANCE | duplicate leg or product ids make the model axes ambiguous"]

    for legs in product_legs:
        if not isinstance(legs, list):
            return None, ["INSTANCE | each products_legs_used entry must be a list"]
        try:
            if any(leg_id not in leg_index for leg_id in legs):
                return None, ["INSTANCE | products_legs_used refers to an unknown leg id"]
        except TypeError:
            return None, ["INSTANCE | products_legs_used contains an unusable leg id"]

    segment_models = []
    for considered, weights, outside_weight in zip(consideration_sets, preferences, no_purchase):
        if not isinstance(considered, list) or not isinstance(weights, list):
            return None, ["INSTANCE | each segment consideration set and preference vector must be a list"]
        if len(considered) != len(weights):
            return None, ["INSTANCE | a segment preference vector does not align with its consideration set"]
        if any(not _is_finite_number(weight) for weight in weights):
            return None, ["INSTANCE | segment preferences must be finite numbers to evaluate C1"]
        try:
            if any(product_id not in product_index for product_id in considered):
                return None, ["INSTANCE | a segment consideration set refers to an unknown product id"]
            segment_models.append((dict(zip(considered, weights)), outside_weight))
        except TypeError:
            return None, ["INSTANCE | a segment consideration set contains an unusable product id"]

    segment_total = _fsum(segment_rates)
    if not math.isfinite(segment_total):
        return None, ["INSTANCE | segment shares cannot be derived from segment_lambda_l"]
    if not segment_rates:
        segment_shares = []
    elif segment_total == 0 and arrival_rate == 0:
        segment_shares = [0.0 for _ in segment_rates]
    elif segment_total == 0:
        return None, ["INSTANCE | segment shares cannot be derived because segment_lambda_l sums to zero"]
    else:
        segment_shares = [rate / segment_total for rate in segment_rates]

    return {
        "arrival_rate": arrival_rate,
        "horizon": horizon,
        "leg_ids": leg_ids,
        "capacities": capacities,
        "leg_index": leg_index,
        "product_ids": product_ids,
        "product_index": product_index,
        "product_legs": product_legs,
        "segment_shares": segment_shares,
        "segment_models": segment_models,
    }, []


def check_solution(instance: dict, solution: dict) -> list:
    """Return described-model violations; an empty list means feasible."""
    violations = []
    data, instance_errors = _instance_data(instance)
    if instance_errors:
        return instance_errors

    if not isinstance(solution, dict):
        return ["FORMAT | solution must be an object"]
    if "offer_set_allocations" not in solution:
        return ["FORMAT | missing required field offer_set_allocations"]

    records = solution["offer_set_allocations"]
    if not isinstance(records, list):
        return ["FORMAT | offer_set_allocations must be a list"]

    durations_by_offer = {}
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            violations.append(f"FORMAT | allocation {record_index} must be an object")
            continue
        if "offer_set" not in record or "time_allocated" not in record:
            violations.append(
                f"FORMAT | allocation {record_index} must contain offer_set and time_allocated"
            )
            continue

        offer_set = record["offer_set"]
        duration = record["time_allocated"]
        if not isinstance(offer_set, list):
            violations.append(f"FORMAT | allocation {record_index} offer_set must be a list")
            continue
        if not _is_finite_number(duration):
            violations.append(f"FORMAT | allocation {record_index} time_allocated must be a finite number")
            continue

        # Decision-variable domain: t[S] >= 0. The submitted value is compared exactly.
        if duration < 0:
            violations.append(f"D1 | allocation {record_index} has negative time_allocated {duration!r}")
            continue

        normalized_offer = set()
        malformed_offer = False
        for product_id in offer_set:
            try:
                known_product = not isinstance(product_id, bool) and product_id in data["product_index"]
            except TypeError:
                known_product = False
            if not known_product:
                violations.append(
                    f"FORMAT | allocation {record_index} offer_set contains unknown product id {product_id!r}"
                )
                malformed_offer = True
                break
            normalized_offer.add(product_id)
        if malformed_offer:
            continue

        offer_key = frozenset(normalized_offer)
        durations_by_offer.setdefault(offer_key, []).append(duration)

    if violations:
        return violations

    merged_allocations = [
        (offer_set, _fsum(durations))
        for offer_set, durations in durations_by_offer.items()
    ]

    # C1: sum_S(lambda * Q_i(S) * t[S]) <= c_i for every flight leg i.
    leg_terms = [[] for _ in data["leg_ids"]]
    for offer_set, duration in merged_allocations:
        if duration == 0:
            continue
        for share, (segment_weights, outside_weight) in zip(
            data["segment_shares"], data["segment_models"]
        ):
            available = offer_set.intersection(segment_weights)
            if not available:
                continue
            denominator = outside_weight + _fsum(segment_weights[p] for p in available)
            if denominator == 0 or not math.isfinite(denominator):
                return ["C1 | MNL choice probability is undefined for an allocated offer set"]
            for product_id in available:
                probability_term = share * segment_weights[product_id] / denominator
                consumption = data["arrival_rate"] * probability_term * duration
                if not math.isfinite(consumption):
                    return ["C1 | computed capacity consumption is not finite"]
                product_pos = data["product_index"][product_id]
                for leg_id in data["product_legs"][product_pos]:
                    leg_terms[data["leg_index"][leg_id]].append(consumption)

    for leg_pos, (leg_id, capacity) in enumerate(zip(data["leg_ids"], data["capacities"])):
        consumption = _fsum(leg_terms[leg_pos])
        if not _leq(consumption, capacity):
            violations.append(
                f"C1 | leg {leg_id!r} consumes {consumption!r}, exceeding capacity {capacity!r}"
            )

    # C2: sum_S(t[S]) <= T. The sum is float arithmetic, so use relative tolerance.
    total_duration = _fsum(duration for _, duration in merged_allocations)
    if not _leq(total_duration, data["horizon"]):
        violations.append(
            f"C2 | total allocated time {total_duration!r} exceeds booking horizon {data['horizon']!r}"
        )

    return violations
