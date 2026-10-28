def _mapping(value):
    return value if isinstance(value, dict) else {}


def _sequence(value):
    return value if isinstance(value, (list, tuple)) else []


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _instance_size(instance, fallback=0):
    n = _as_int(instance.get("n"), 0)
    if n > 0:
        return n

    lengths = [
        len(_sequence(instance.get("mu"))),
        len(_sequence(instance.get("factor_loadings"))),
        len(_sequence(instance.get("idiosyncratic_variance"))),
        len(_sequence(instance.get("covariance_matrix"))),
        _as_int(fallback, 0),
    ]
    return max(lengths) if lengths else 0


def _vector(value, length, default=0.0):
    source = _sequence(value)
    return [
        _as_float(source[i], default) if i < len(source) else float(default)
        for i in range(length)
    ]


def _matrix(value, rows, columns, default=0.0):
    source = _sequence(value)
    result = []
    for i in range(rows):
        row = _sequence(source[i]) if i < len(source) else []
        result.append([
            _as_float(row[j], default) if j < len(row) else float(default)
            for j in range(columns)
        ])
    return result


def _factor_count(instance, factor_rows):
    declared = _as_int(instance.get("r"), 0)
    if declared > 0:
        return declared
    return max((len(_sequence(row)) for row in factor_rows), default=0)


def to_ours(official_instance: dict) -> dict:
    instance = _mapping(official_instance)
    n = _instance_size(instance)
    mu = _vector(instance.get("mu"), n, 0.0)

    raw_covariance = _sequence(instance.get("covariance_matrix"))
    raw_factors = _sequence(instance.get("factor_loadings"))
    declared_format = instance.get("covariance_format")
    use_direct = declared_format == "direct" or (
        declared_format != "factor" and bool(raw_covariance) and not raw_factors
    )

    if use_direct:
        covariance_format = "direct"
        covariance_matrix = _matrix(raw_covariance, n, n, 0.0)
        factor_loadings = []
        idiosyncratic_variances = []
    else:
        covariance_format = "factor"
        covariance_matrix = []
        factor_count = _factor_count(instance, raw_factors)
        official_factors = _matrix(raw_factors, n, factor_count, 0.0)
        factor_loadings = [
            [official_factors[security][factor] for security in range(n)]
            for factor in range(factor_count)
        ]
        idiosyncratic_variances = _vector(
            instance.get("idiosyncratic_variance"), n, 0.0
        )

    constraints = _mapping(instance.get("constraints"))
    raw_rules = _sequence(constraints.get("A_lin"))
    rule_coefficients = _matrix(raw_rules, len(raw_rules), n, 0.0)
    raw_lower = _sequence(constraints.get("l_lin"))
    raw_upper = _sequence(constraints.get("u_lin"))
    lower_bounds = []
    upper_bounds = []

    for index, row in enumerate(rule_coefficients):
        natural_lower = min(row) if row else 0.0
        natural_upper = max(row) if row else 0.0
        lower_bounds.append(
            _as_float(raw_lower[index], natural_lower)
            if index < len(raw_lower)
            else natural_lower
        )
        upper_bounds.append(
            _as_float(raw_upper[index], natural_upper)
            if index < len(raw_upper)
            else natural_upper
        )

    if (
        _as_bool(constraints.get("has_min_return_constraint", False))
        and constraints.get("r_bar") is not None
    ):
        minimum_return = _as_float(constraints.get("r_bar"), 0.0)
        rule_coefficients.append(list(mu))
        lower_bounds.append(minimum_return)
        upper_bounds.append(max([minimum_return] + mu) if mu else minimum_return)

    has_minimum = (
        _as_bool(constraints.get("has_min_investment_constraint", False))
        and constraints.get("l_min_investment") is not None
    )
    has_maximum = constraints.get("u_max_investment") is not None
    if has_minimum or has_maximum:
        minimum_weights = (
            _vector(constraints.get("l_min_investment"), n, 0.0)
            if has_minimum
            else [0.0] * n
        )
        maximum_weights = (
            _vector(constraints.get("u_max_investment"), n, 1.0)
            if has_maximum
            else [1.0] * n
        )
    else:
        minimum_weights = []
        maximum_weights = []

    return {
        "num_securities": n,
        "expected_returns": mu,
        "covariance_format": covariance_format,
        "covariance_matrix": covariance_matrix,
        "factor_loadings": factor_loadings,
        "idiosyncratic_variances": idiosyncratic_variances,
        "return_weight": _as_float(instance.get("kappa"), 0.0),
        "concentration_parameter": _as_float(instance.get("gamma"), 1.0),
        "max_securities": _as_int(instance.get("k"), n),
        "linear_rule_coefficients": rule_coefficients,
        "linear_rule_lower_bounds": lower_bounds,
        "linear_rule_upper_bounds": upper_bounds,
        "minimum_weights": minimum_weights,
        "maximum_weights": maximum_weights,
    }


def to_official(official_instance: dict, our_solution: dict) -> dict:
    instance = _mapping(official_instance)
    solution = _mapping(our_solution)
    raw_weights = _sequence(solution.get("weights"))
    n = _instance_size(instance, len(raw_weights))
    x = _vector(raw_weights, n, 0.0)

    official_solution = {
        "instance_id": instance.get("instance_id", ""),
        "x": x,
        "support": [i for i, weight in enumerate(x) if abs(weight) > 1e-5],
    }
    official_solution["objective_value"] = official_objective(
        instance, official_solution
    )
    return official_solution


def official_objective(official_instance: dict, official_solution: dict) -> float:
    instance = _mapping(official_instance)
    solution = _mapping(official_solution)
    raw_x = _sequence(solution.get("x"))
    n = _instance_size(instance, len(raw_x))
    x = _vector(raw_x, n, 0.0)
    mu = _vector(instance.get("mu"), n, 0.0)
    idiosyncratic = _vector(
        instance.get("idiosyncratic_variance"), n, 0.0
    )

    raw_factors = _sequence(instance.get("factor_loadings"))
    raw_covariance = _sequence(instance.get("covariance_matrix"))

    if raw_factors:
        factor_count = _factor_count(instance, raw_factors)
        factors = _matrix(raw_factors, n, factor_count, 0.0)
        common_risk = 0.0
        for factor in range(factor_count):
            exposure = sum(
                factors[security][factor] * x[security]
                for security in range(n)
            )
            common_risk += exposure * exposure
        variance = common_risk + sum(
            idiosyncratic[i] * x[i] * x[i] for i in range(n)
        )
    elif raw_covariance:
        covariance = _matrix(raw_covariance, n, n, 0.0)
        variance = sum(
            x[i] * covariance[i][j] * x[j]
            for i in range(n)
            for j in range(n)
        )
    else:
        variance = sum(
            idiosyncratic[i] * x[i] * x[i] for i in range(n)
        )

    gamma = _as_float(instance.get("gamma"), 1.0)
    if gamma <= 0.0:
        gamma = 1.0
    kappa = _as_float(instance.get("kappa"), 0.0)
    squared_norm = sum(weight * weight for weight in x)
    expected_return = sum(mu[i] * x[i] for i in range(n))

    return float(
        0.5 * variance
        + squared_norm / (2.0 * gamma)
        - kappa * expected_return
    )