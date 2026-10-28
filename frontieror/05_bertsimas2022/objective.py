import math


def _finite_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _vector(value, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of length {length}")
    return [_finite_float(item, f"{name}[{i}]") for i, item in enumerate(value)]


def objective(instance: dict, solution: dict) -> float:
    """Return the model objective value; smaller values are always better."""
    if not isinstance(instance, dict):
        raise ValueError("instance must be a dictionary")
    if not isinstance(solution, dict):
        raise ValueError("solution must be a dictionary")

    n = instance.get("num_securities")
    if type(n) is not int or n <= 0:
        raise ValueError("num_securities must be a positive integer")

    expected_returns = _vector(instance.get("expected_returns"), n, "expected_returns")
    weights = _vector(solution.get("weights"), n, "weights")

    return_weight = _finite_float(instance.get("return_weight"), "return_weight")
    if return_weight < 0.0:
        raise ValueError("return_weight must be non-negative")

    concentration_parameter = _finite_float(
        instance.get("concentration_parameter"), "concentration_parameter"
    )
    if concentration_parameter <= 0.0:
        raise ValueError("concentration_parameter must be strictly positive")

    covariance_format = instance.get("covariance_format")
    if covariance_format == "direct":
        raw_covariance = instance.get("covariance_matrix")
        if not isinstance(raw_covariance, list) or len(raw_covariance) != n:
            raise ValueError("covariance_matrix must have num_securities rows")
        covariance = [
            _vector(row, n, f"covariance_matrix[{i}]")
            for i, row in enumerate(raw_covariance)
        ]
        if instance.get("factor_loadings") != []:
            raise ValueError("factor_loadings must be empty in direct covariance mode")
        if instance.get("idiosyncratic_variances") != []:
            raise ValueError(
                "idiosyncratic_variances must be empty in direct covariance mode"
            )

        variance = math.fsum(
            covariance[i][j] * weights[i] * weights[j]
            for i in range(n)
            for j in range(n)
        )

    elif covariance_format == "factor":
        if instance.get("covariance_matrix") != []:
            raise ValueError("covariance_matrix must be empty in factor mode")

        raw_loadings = instance.get("factor_loadings")
        if not isinstance(raw_loadings, list):
            raise ValueError("factor_loadings must be a list of factor rows")
        loadings = [
            _vector(row, n, f"factor_loadings[{factor}]")
            for factor, row in enumerate(raw_loadings)
        ]
        idiosyncratic_variances = _vector(
            instance.get("idiosyncratic_variances"),
            n,
            "idiosyncratic_variances",
        )
        for i, value in enumerate(idiosyncratic_variances):
            if value < 0.0:
                raise ValueError(
                    f"idiosyncratic_variances[{i}] must be non-negative"
                )

        factor_variance = math.fsum(
            math.fsum(row[i] * weights[i] for i in range(n)) ** 2
            for row in loadings
        )
        idiosyncratic_variance = math.fsum(
            idiosyncratic_variances[i] * weights[i] ** 2 for i in range(n)
        )
        variance = factor_variance + idiosyncratic_variance

    else:
        raise ValueError("covariance_format must be 'direct' or 'factor'")

    squared_weight_sum = math.fsum(weight * weight for weight in weights)
    expected_return = math.fsum(
        expected_returns[i] * weights[i] for i in range(n)
    )

    return float(
        0.5 * variance
        + squared_weight_sum / (2.0 * concentration_parameter)
        - return_weight * expected_return
    )
