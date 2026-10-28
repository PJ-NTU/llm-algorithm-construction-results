"""Feasibility checker for the bin-packing solution format in MAT_s4.md."""


def _is_plain_int(value):
    """JSON integers are accepted; JSON booleans are not integer encodings here."""
    return isinstance(value, int) and not isinstance(value, bool)


def _check_c1(n, item_assignments):
    """C1: every item position contains exactly one valid bin index."""
    violations = []

    if len(item_assignments) != n:
        violations.append(
            "C1 | item_assignments must contain exactly one bin index for each "
            f"of the {n} items; received {len(item_assignments)} entries"
        )
        return violations

    for item_index, bin_index in enumerate(item_assignments):
        if not 0 <= bin_index < n:
            violations.append(
                "C1 | item "
                f"{item_index} is assigned to bin {bin_index}, outside the "
                f"described index range 0..{n - 1}"
            )

    return violations


def _check_c2(n, capacity, weights, bin_activation, item_assignments):
    """C2: assigned bins are active and every bin obeys its capacity bound."""
    loads = [0] * n
    item_counts = [0] * n

    for item_index, bin_index in enumerate(item_assignments):
        # C1 validates the index range before this function is called.
        loads[bin_index] += weights[item_index]
        item_counts[bin_index] += 1

    violations = []
    for bin_index in range(n):
        # The prose part of C2 explicitly requires any nonempty bin to be active.
        # This separate check remains correct even for degenerate signed weights.
        if item_counts[bin_index] and bin_activation[bin_index] != 1:
            violations.append(
                "C2 | bin "
                f"{bin_index} contains {item_counts[bin_index]} item(s) but its "
                "activation is not 1"
            )

        # capacity, weights, activations, and their sums/products are integers, so
        # this comparison is exact and intentionally has no numeric tolerance.
        allowed_load = capacity * bin_activation[bin_index]
        if loads[bin_index] > allowed_load:
            violations.append(
                "C2 | bin "
                f"{bin_index} has recomputed load {loads[bin_index]}, exceeding "
                f"capacity * activation = {allowed_load}"
            )

    return violations


def _check_c3(n, bin_activation):
    """C3: activation values are nonincreasing with the bin index."""
    violations = []
    for bin_index in range(n - 1):
        # Both operands are binary integers, so exact comparison is required.
        if bin_activation[bin_index] < bin_activation[bin_index + 1]:
            violations.append(
                "C3 | lower-index bin "
                f"{bin_index} has activation {bin_activation[bin_index]}, below "
                f"bin {bin_index + 1}'s activation "
                f"{bin_activation[bin_index + 1]}"
            )
    return violations


def check_solution(instance: dict, solution: dict) -> list:
    """Return concrete violations; return an empty list for a feasible solution."""
    violations = []

    if not isinstance(instance, dict):
        return ["FORMAT | instance must be a dictionary"]
    if not isinstance(solution, dict):
        return ["FORMAT | solution must be a dictionary"]

    for field in ("n", "capacity", "weights"):
        if field not in instance:
            violations.append(f"FORMAT | instance is missing required field '{field}'")

    for field in ("bin_activation", "item_assignments"):
        if field not in solution:
            violations.append(f"FORMAT | solution is missing required field '{field}'")

    if violations:
        return violations

    n = instance["n"]
    capacity = instance["capacity"]
    weights = instance["weights"]
    bin_activation = solution["bin_activation"]
    item_assignments = solution["item_assignments"]

    if not _is_plain_int(n):
        violations.append("FORMAT | instance.n must be an integer")
    if not _is_plain_int(capacity):
        violations.append("FORMAT | instance.capacity must be an integer")
    if not isinstance(weights, list):
        violations.append("FORMAT | instance.weights must be a list")
    if not isinstance(bin_activation, list):
        violations.append("FORMAT | solution.bin_activation must be a list")
    if not isinstance(item_assignments, list):
        violations.append("FORMAT | solution.item_assignments must be a list")

    if violations:
        return violations

    # A negative n cannot describe the required list lengths. This is a shape
    # failure, not an extra input-value assumption such as requiring n > 0.
    if n < 0:
        violations.append(
            "FORMAT | instance.n cannot be negative because it defines list lengths"
        )
        return violations

    if len(weights) != n:
        violations.append(
            "FORMAT | instance.weights length must equal instance.n; "
            f"received {len(weights)} and {n}"
        )
    else:
        for item_index, weight in enumerate(weights):
            if not _is_plain_int(weight):
                violations.append(
                    f"FORMAT | instance.weights[{item_index}] must be an integer"
                )

    if len(bin_activation) != n:
        violations.append(
            "FORMAT | solution.bin_activation must list all n potential bins; "
            f"received {len(bin_activation)} entries for n={n}"
        )
    else:
        for bin_index, activation in enumerate(bin_activation):
            if not _is_plain_int(activation) or activation not in (0, 1):
                violations.append(
                    "FORMAT | solution.bin_activation["
                    f"{bin_index}] must be the binary integer 0 or 1"
                )

    for item_index, bin_index in enumerate(item_assignments):
        if not _is_plain_int(bin_index):
            violations.append(
                "FORMAT | solution.item_assignments["
                f"{item_index}] must be an integer bin index"
            )

    if violations:
        return violations

    # Exactly three semantic checks correspond one-for-one to C1, C2, and C3.
    c1_violations = _check_c1(n, item_assignments)
    violations.extend(c1_violations)

    # C2 can safely reconstruct loads only after C1 has established one valid
    # bin index for every item. C3 is independent and is always checked here.
    if not c1_violations:
        violations.extend(
            _check_c2(n, capacity, weights, bin_activation, item_assignments)
        )
    violations.extend(_check_c3(n, bin_activation))

    # Optional derived fields (for example "objective") are deliberately not
    # trusted or required. The actual used-bin count is bin_activation's sum.
    _recomputed_objective = sum(bin_activation)
    del _recomputed_objective

    return violations
