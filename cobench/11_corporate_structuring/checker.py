"""Feasibility checker for the tax-efficient corporate hierarchy problem."""

from __future__ import annotations

import math
import re
from typing import Any


_REL = 1e-9
_SIGNED_INTEGER = re.compile(r"(?<!\d)[+-]?\d+(?!\d)")


def _leq(a: float, b: float) -> bool:
    return a - b <= _REL * max(1.0, abs(a), abs(b))


def _geq(a: float, b: float) -> bool:
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _close(a: float, b: float) -> bool:
    return _leq(a, b) and _geq(a, b)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _as_index(value: Any, n: int, *, allow_zero: bool) -> int | None:
    """Parse an index without imposing cosmetic rules on a string label."""
    parsed: int
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str):
        matches = _SIGNED_INTEGER.findall(value.strip())
        candidates = {int(match) for match in matches}
        if len(candidates) != 1:
            return None
        parsed = candidates.pop()
    else:
        return None

    lower = 0 if allow_zero else 1
    return parsed if lower <= parsed <= n else None


def _normalise_mapping(
    raw: Any,
    n: int,
    field: str,
    *,
    values_are_indices: bool,
) -> tuple[dict[int, Any], list[str]]:
    if not isinstance(raw, dict):
        return {}, [f"FORMAT | {field} must be a mapping"]

    normalised: dict[int, Any] = {}
    errors: list[str] = []
    for raw_key, raw_value in raw.items():
        key = _as_index(raw_key, n, allow_zero=False)
        if key is None:
            errors.append(
                f"FORMAT | {field} key {raw_key!r} does not uniquely identify "
                f"a country in 1..{n}"
            )
            continue
        if key in normalised:
            errors.append(
                f"FORMAT | {field} contains more than one entry for country {key}"
            )
            continue

        if values_are_indices:
            value = _as_index(raw_value, n, allow_zero=True)
            if value is None:
                errors.append(
                    f"FORMAT | {field}[{key}] does not identify a parent in 0..{n}"
                )
                continue
            normalised[key] = value
        else:
            if not _is_number(raw_value):
                errors.append(f"FORMAT | {field}[{key}] must be a finite number")
                continue
            normalised[key] = float(raw_value)

    return normalised, errors


def _prepare_instance(instance: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(instance, dict):
        return None, ["FORMAT | instance must be a mapping"]

    raw_n = instance.get("N")
    if isinstance(raw_n, bool) or not isinstance(raw_n, int):
        return None, ["FORMAT | instance.N must be an integer"]
    n = raw_n

    target = _as_index(instance.get("target"), n, allow_zero=False)
    if target is None:
        return None, [f"FORMAT | instance.target must identify a country in 1..{n}"]

    # Country records are vectors rather than scalar solution amounts, so parse
    # them separately while retaining the same permissive index-label policy.
    countries = {}
    raw_countries = instance.get("countries")
    errors: list[str] = []
    if not isinstance(raw_countries, dict):
        errors.append("FORMAT | instance.countries must be a mapping")
    else:
        for raw_key, record in raw_countries.items():
            key = _as_index(raw_key, n, allow_zero=False)
            if key is None:
                continue
            if key in countries:
                errors.append(
                    f"FORMAT | instance.countries has duplicate entries for country {key}"
                )
                continue
            if not isinstance(record, (list, tuple)) or len(record) < 4:
                errors.append(
                    f"FORMAT | instance.countries[{key}] must contain C, D, F_rate, P"
                )
                continue
            if not all(_is_number(record[position]) for position in range(4)):
                errors.append(
                    f"FORMAT | instance.countries[{key}] has a non-finite C/D/F_rate/P value"
                )
                continue
            countries[key] = tuple(float(record[position]) for position in range(4))

    # Do not impose an extra value-domain rule on the instance.  All N records
    # are needed only because the authoritative equations quantify over 1..N.
    for country in range(1, n + 1):
        if country not in countries:
            errors.append(
                f"FORMAT | instance.countries is missing country {country}, "
                "whose values are needed by C4-C8"
            )

    raw_withholding = instance.get("withholding")
    withholding: dict[int, dict[int, float]] = {}
    if not isinstance(raw_withholding, dict):
        errors.append("FORMAT | instance.withholding must be a mapping")
    else:
        for raw_from, raw_row in raw_withholding.items():
            source = _as_index(raw_from, n, allow_zero=False)
            if source is None or not isinstance(raw_row, dict):
                continue
            if source in withholding:
                errors.append(
                    f"FORMAT | instance.withholding has duplicate rows for country {source}"
                )
                continue
            row: dict[int, float] = {}
            for raw_to, raw_rate in raw_row.items():
                destination = _as_index(raw_to, n, allow_zero=False)
                if destination is None or destination in row:
                    continue
                if _is_number(raw_rate):
                    row[destination] = float(raw_rate)
            withholding[source] = row

    if errors:
        return None, errors
    return {
        "n": n,
        "target": target,
        "countries": countries,
        "withholding": withholding,
    }, []


def _prepare_solution(
    solution: Any, data: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(solution, dict):
        return None, ["FORMAT | solution must be a mapping"]

    n = data["n"]
    required_fields = ("structure", "net_dividends", "foreign_tax_paid")
    errors = [
        f"FORMAT | missing required output field {field!r}"
        for field in required_fields
        if field not in solution
    ]
    if errors:
        return None, errors

    structure, structure_errors = _normalise_mapping(
        solution["structure"], n, "structure", values_are_indices=True
    )
    dividends, dividend_errors = _normalise_mapping(
        solution["net_dividends"],
        n,
        "net_dividends",
        values_are_indices=False,
    )
    foreign_tax, tax_errors = _normalise_mapping(
        solution["foreign_tax_paid"],
        n,
        "foreign_tax_paid",
        values_are_indices=False,
    )
    errors.extend(structure_errors + dividend_errors + tax_errors)

    positive = {
        country
        for country, record in data["countries"].items()
        if record[3] > 0
    }
    for country in sorted(positive - dividends.keys()):
        errors.append(
            f"FORMAT | net_dividends is missing positive-profit country {country}"
        )
    for country in sorted(positive - foreign_tax.keys()):
        errors.append(
            f"FORMAT | foreign_tax_paid is missing positive-profit country {country}"
        )

    for country, value in sorted(dividends.items()):
        if not _geq(value, 0.0):
            errors.append(
                f"DOMAIN | net_dividends[{country}]={value!r} violates F[{country}] >= 0"
            )
    for country, value in sorted(foreign_tax.items()):
        if not _geq(value, 0.0):
            errors.append(
                f"DOMAIN | foreign_tax_paid[{country}]={value!r} violates "
                f"T_foreign[{country}] >= 0"
            )

    if errors:
        return None, errors
    return {
        "structure": structure,
        "net_dividends": dividends,
        "foreign_tax_paid": foreign_tax,
        "positive": positive,
    }, []


def _check_c1(data: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    """C1: every positive-profit country has exactly one non-self parent."""
    structure = answer["structure"]
    errors: list[str] = []
    for country in sorted(answer["positive"]):
        if country not in structure:
            errors.append(
                f"C1 | positive-profit country {country} has no parent in structure"
            )
        elif structure[country] == country:
            errors.append(f"C1 | country {country} cannot be its own parent")
    return errors


def _check_c2(data: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    """C2: the designated target has virtual parent zero."""
    target = data["target"]
    actual = answer["structure"].get(target)
    if actual != 0:
        return [f"C2 | target country {target} must have parent 0, got {actual!r}"]
    return []


def _check_c3(data: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    """C3: every included country belongs to the acyclic target-rooted tree."""
    target = data["target"]
    structure = answer["structure"]
    errors: list[str] = []

    for start in sorted(structure):
        seen: set[int] = set()
        current = start
        while current != 0:
            if current in seen:
                errors.append(
                    f"C3 | parent chain from country {start} contains a cycle at {current}"
                )
                break
            seen.add(current)
            if current not in structure:
                errors.append(
                    f"C3 | parent chain from country {start} stops at country {current} "
                    f"instead of reaching target {target} and then 0"
                )
                break
            parent = structure[current]
            if parent == 0 and current != target:
                errors.append(
                    f"C3 | included non-target country {current} points directly to 0; "
                    f"the tree must be rooted at target {target}"
                )
                break
            current = parent
    return errors


def _children(structure: dict[int, int], n: int) -> dict[int, list[int]]:
    result = {country: [] for country in range(1, n + 1)}
    for child, parent in structure.items():
        if 1 <= parent <= n:
            result[parent].append(child)
    for children in result.values():
        children.sort()
    return result


def _subtree_nodes(root: int, children: dict[int, list[int]]) -> set[int]:
    result: set[int] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.add(current)
        pending.extend(children[current])
    return result


def _withholding_rate(
    data: dict[str, Any], child: int, parent: int
) -> tuple[float | None, str | None]:
    value = data["withholding"].get(child, {}).get(parent)
    if value is None:
        return None, (
            f"FORMAT | instance.withholding[{child}][{parent}] is needed by C4-C8"
        )
    return value, None


def _build_completion(
    data: dict[str, Any], answer: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    """Complete omitted non-positive-country amounts when a completion exists."""
    n = data["n"]
    children = _children(answer["structure"], n)
    dividends_given = answer["net_dividends"]
    tax_given = answer["foreign_tax_paid"]
    f_value: dict[int, float] = {}
    tax_value: dict[int, float] = {}
    gross_value: dict[int, float] = {}
    prescribed_tax: dict[int, float] = {}
    errors: list[str] = []
    visiting: set[int] = set()

    def visit(country: int) -> None:
        if country in f_value or country in errors_by_country:
            return
        if country in visiting:
            errors_by_country.add(country)
            return
        visiting.add(country)
        for child in children[country]:
            visit(child)
        if any(child not in f_value for child in children[country]):
            visiting.remove(country)
            errors_by_country.add(country)
            return

        tax_code, domestic_rate, foreign_rate, profit = data["countries"][country]
        inflow = 0.0
        for child in children[country]:
            rate, error = _withholding_rate(data, child, country)
            if error is not None:
                errors.append(error)
                visiting.remove(country)
                errors_by_country.add(country)
                return
            inflow += f_value[child] * (1.0 - rate)

        gross = profit * (1.0 - domestic_rate) + inflow
        gross_value[country] = gross

        expected_tax: float | None = None
        if tax_code == 1:
            expected_tax = 0.0
        elif tax_code == 2:
            expected_tax = foreign_rate * inflow
        elif tax_code == 3:
            expected_tax = 0.0
            for child in children[country]:
                rate, _ = _withholding_rate(data, child, country)
                branch_profit = sum(
                    data["countries"][member][3]
                    for member in _subtree_nodes(child, children)
                )
                expected_tax += max(
                    0.0,
                    f_value[child] * (1.0 - rate)
                    - (1.0 - foreign_rate) * branch_profit,
                )
        elif tax_code == 4:
            descendant_profit = sum(
                data["countries"][member][3]
                for member in _subtree_nodes(country, children)
                if member != country
            )
            expected_tax = max(
                0.0,
                inflow - (1.0 - foreign_rate) * descendant_profit,
            )

        if expected_tax is not None:
            prescribed_tax[country] = expected_tax
            tax = expected_tax
            dividend = gross - tax
        else:
            has_f = country in dividends_given
            has_t = country in tax_given
            if has_f and has_t:
                dividend = dividends_given[country]
                tax = tax_given[country]
            elif has_f:
                dividend = dividends_given[country]
                tax = gross - dividend
            elif has_t:
                tax = tax_given[country]
                dividend = gross - tax
            else:
                tax = 0.0
                dividend = gross

        f_value[country] = dividend
        tax_value[country] = tax
        visiting.remove(country)

    errors_by_country: set[int] = set()
    for country in range(1, n + 1):
        visit(country)

    if errors_by_country and not errors:
        errors.append("C3 | cyclic structure prevents evaluation of C4-C8")
    if errors:
        return None, errors
    return {
        "children": children,
        "f": f_value,
        "tax": tax_value,
        "gross": gross_value,
        "prescribed_tax": prescribed_tax,
    }, []


def _check_c4(
    data: dict[str, Any], answer: dict[str, Any], completed: dict[str, Any]
) -> list[str]:
    """C4: recompute every country's dividend-balance equality."""
    errors: list[str] = []
    for country in range(1, data["n"] + 1):
        expected = completed["gross"][country] - completed["tax"][country]
        actual = completed["f"][country]
        if not _close(actual, expected):
            errors.append(
                f"C4 | country {country} has F={actual!r}, but its profit, child "
                f"inflows, and T_foreign imply {expected!r}"
            )
        if country in answer["net_dividends"] and not _close(
            answer["net_dividends"][country], expected
        ):
            errors.append(
                f"C4 | net_dividends[{country}]={answer['net_dividends'][country]!r}; "
                f"recalculation gives {expected!r}"
            )
        if not _geq(actual, 0.0):
            errors.append(
                f"C4 | country {country}'s balance forces F={actual!r}, "
                "so no nonnegative F satisfies the authoritative equality"
            )
        if not _geq(completed["tax"][country], 0.0):
            errors.append(
                f"C4 | country {country}'s completed T_foreign="
                f"{completed['tax'][country]!r} violates its nonnegative domain"
            )
    return errors


def _check_tax_code(
    constraint: str,
    code: int,
    data: dict[str, Any],
    answer: dict[str, Any],
    completed: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for country in range(1, data["n"] + 1):
        if data["countries"][country][0] != code:
            continue
        expected = completed["prescribed_tax"][country]
        if not _geq(expected, 0.0):
            errors.append(
                f"{constraint} | country {country}'s code-{code} formula forces "
                f"negative T_foreign={expected!r}"
            )
        if country in answer["foreign_tax_paid"]:
            actual = answer["foreign_tax_paid"][country]
            if not _close(actual, expected):
                errors.append(
                    f"{constraint} | foreign_tax_paid[{country}]={actual!r}; "
                    f"the code-{code} formula gives {expected!r}"
                )
    return errors


def _check_c5(
    data: dict[str, Any], answer: dict[str, Any], completed: dict[str, Any]
) -> list[str]:
    """C5: code 1 (exemption) prescribes zero foreign tax."""
    return _check_tax_code("C5", 1, data, answer, completed)


def _check_c6(
    data: dict[str, Any], answer: dict[str, Any], completed: dict[str, Any]
) -> list[str]:
    """C6: code 2 taxes the total direct-child inflow."""
    return _check_tax_code("C6", 2, data, answer, completed)


def _check_c7(
    data: dict[str, Any], answer: dict[str, Any], completed: dict[str, Any]
) -> list[str]:
    """C7: code 3 applies a separate credit calculation per child branch."""
    return _check_tax_code("C7", 3, data, answer, completed)


def _check_c8(
    data: dict[str, Any], answer: dict[str, Any], completed: dict[str, Any]
) -> list[str]:
    """C8: code 4 applies one pooled credit calculation to all branches."""
    return _check_tax_code("C8", 4, data, answer, completed)


def check_solution(instance: dict, solution: dict) -> list:
    """Return human-readable violations; an empty list means feasible."""
    try:
        data, errors = _prepare_instance(instance)
        if errors:
            return errors
        assert data is not None

        answer, errors = _prepare_solution(solution, data)
        if errors:
            return errors
        assert answer is not None

        violations: list[str] = []
        c1 = _check_c1(data, answer)
        c2 = _check_c2(data, answer)
        c3 = _check_c3(data, answer)
        violations.extend(c1)
        violations.extend(c2)
        violations.extend(c3)

        # C4-C8 depend on a well-founded child order.  They cannot be evaluated
        # meaningfully after a structural violation, which is already reported.
        if c1 or c2 or c3:
            return violations

        completed, completion_errors = _build_completion(data, answer)
        if completion_errors:
            return violations + completion_errors
        assert completed is not None

        violations.extend(_check_c4(data, answer, completed))
        violations.extend(_check_c5(data, answer, completed))
        violations.extend(_check_c6(data, answer, completed))
        violations.extend(_check_c7(data, answer, completed))
        violations.extend(_check_c8(data, answer, completed))
        return violations
    except Exception as exc:  # Defensive boundary: malformed inputs never escape.
        return [f"FORMAT | checker could not safely read the supplied data: {exc}"]
