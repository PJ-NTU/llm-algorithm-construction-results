import math

ABS_TOL = 1e-8
REL_TOL = 1e-8


def _margin(a: float, b: float) -> float:
    return ABS_TOL + REL_TOL * max(abs(a), abs(b))


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _margin(a, b)


def _below(a: float, b: float) -> bool:
    return a < b - _margin(a, b)


def _above(a: float, b: float) -> bool:
    return a > b + _margin(a, b)


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


def _read_feasibility_data(instance: dict):
    if not isinstance(instance, dict):
        raise ValueError("instance must be a dictionary")

    n = instance.get("num_securities")
    if type(n) is not int or n <= 0:
        raise ValueError("num_securities must be a positive integer")

    max_securities = instance.get("max_securities")
    if type(max_securities) is not int or not 1 <= max_securities <= n:
        raise ValueError("max_securities must be an integer between 1 and num_securities")

    raw_minimums = instance.get("minimum_weights")
    raw_maximums = instance.get("maximum_weights")
    if raw_minimums == [] and raw_maximums == []:
        minimums = [0.0] * n
        maximums = [1.0] * n
    else:
        minimums = _vector(raw_minimums, n, "minimum_weights")
        maximums = _vector(raw_maximums, n, "maximum_weights")
        for i, (minimum, maximum) in enumerate(zip(minimums, maximums)):
            if minimum < 0.0:
                raise ValueError(f"minimum_weights[{i}] must be non-negative")
            if maximum < minimum:
                raise ValueError(
                    f"maximum_weights[{i}] must be at least minimum_weights[{i}]"
                )

    raw_coefficients = instance.get("linear_rule_coefficients")
    raw_lowers = instance.get("linear_rule_lower_bounds")
    raw_uppers = instance.get("linear_rule_upper_bounds")
    if raw_coefficients == [] and raw_lowers == [] and raw_uppers == []:
        coefficients = []
        lowers = []
        uppers = []
    else:
        if not isinstance(raw_coefficients, list):
            raise ValueError("linear_rule_coefficients must be a list")
        rule_count = len(raw_coefficients)
        lowers = _vector(raw_lowers, rule_count, "linear_rule_lower_bounds")
        uppers = _vector(raw_uppers, rule_count, "linear_rule_upper_bounds")
        coefficients = [
            _vector(row, n, f"linear_rule_coefficients[{r}]")
            for r, row in enumerate(raw_coefficients)
        ]
        for r, (lower, upper) in enumerate(zip(lowers, uppers)):
            if lower > upper:
                raise ValueError(
                    f"linear rule {r} has lower bound greater than upper bound"
                )

    return n, max_securities, minimums, maximums, coefficients, lowers, uppers


def check(instance: dict, solution: dict) -> list:
    """Return feasibility violations; an empty list means the solution is feasible."""
    violations = []

    try:
        (
            n,
            max_securities,
            minimums,
            maximums,
            coefficients,
            rule_lowers,
            rule_uppers,
        ) = _read_feasibility_data(instance)
    except ValueError as exc:
        return [f"instance structure: {exc}"]

    if not isinstance(solution, dict):
        return ["solution structure: solution must be a dictionary"]

    raw_weights = solution.get("weights")
    raw_selected = solution.get("selected")
    structure_ok = True

    if not isinstance(raw_weights, list) or len(raw_weights) != n:
        violations.append(f"solution structure: weights must be a list of length {n}")
        structure_ok = False
    if not isinstance(raw_selected, list) or len(raw_selected) != n:
        violations.append(f"solution structure: selected must be a list of length {n}")
        structure_ok = False
    if not structure_ok:
        return violations

    weights = []
    for i, value in enumerate(raw_weights):
        try:
            weights.append(_finite_float(value, f"weights[{i}]"))
        except ValueError as exc:
            violations.append(f"solution structure: {exc}")
            structure_ok = False

    selected = []
    for i, value in enumerate(raw_selected):
        if type(value) is not int or value not in (0, 1):
            violations.append(f"solution structure: selected[{i}] must be 0 or 1")
            structure_ok = False
        else:
            selected.append(value)

    if not structure_ok:
        return violations

    for i, weight in enumerate(weights):
        if _below(weight, 0.0):
            violations.append(f"decision domain: weights[{i}] is negative ({weight})")
        positive = _above(weight, 0.0)
        if selected[i] != int(positive):
            violations.append(
                f"solution completeness: selected[{i}] must equal 1 exactly when "
                f"weights[{i}] is positive"
            )

    # C1: Full-investment budget equality.
    total_weight = math.fsum(weights)
    if not _close(total_weight, 1.0):
        violations.append(f"C1: weights sum to {total_weight}, not 1")

    # C2: An activated security must meet its minimum investment threshold.
    for i in range(n):
        required_minimum = minimums[i] * selected[i]
        if _below(weights[i], required_minimum):
            violations.append(
                f"C2: weights[{i}]={weights[i]} is below "
                f"minimum_weights[{i}]*selected[{i}]={required_minimum}"
            )

    # C3: An inactive security has zero weight and an active one respects its maximum.
    for i in range(n):
        allowed_maximum = maximums[i] * selected[i]
        if _above(weights[i], allowed_maximum):
            violations.append(
                f"C3: weights[{i}]={weights[i]} exceeds "
                f"maximum_weights[{i}]*selected[{i}]={allowed_maximum}"
            )

    # C4: At most max_securities securities may be activated.
    selected_count = sum(selected)
    if selected_count > max_securities:
        violations.append(
            f"C4: {selected_count} securities are selected, exceeding {max_securities}"
        )

    # C5: Every supplied linear rule must satisfy its lower bound.
    for r, row in enumerate(coefficients):
        activity = math.fsum(row[i] * weights[i] for i in range(n))
        if _below(activity, rule_lowers[r]):
            violations.append(
                f"C5: linear rule {r} has value {activity}, below lower bound "
                f"{rule_lowers[r]}"
            )

    # C6: Every supplied linear rule must satisfy its upper bound.
    for r, row in enumerate(coefficients):
        activity = math.fsum(row[i] * weights[i] for i in range(n))
        if _above(activity, rule_uppers[r]):
            violations.append(
                f"C6: linear rule {r} has value {activity}, above upper bound "
                f"{rule_uppers[r]}"
            )

    return violations
