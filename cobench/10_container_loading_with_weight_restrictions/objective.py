"""Smaller objective is better: return the negative raw volume utilisation, standard library only.

Callers should first confirm with checker.check that the input and solution are complete and feasible.
This function adds no penalty and does not re-check feasibility; downstream must not negate it again.
Values are computed with exact rational arithmetic on the decimal representations of int/float, converted to float at the end.
"""

from fractions import Fraction


def _exact(value):
    return Fraction(value) if type(value) is int else Fraction(str(value))


def objective(instance: dict, solution: dict) -> float:
    """Return -sum(s[i][k] * original length * original width * original height) / container volume."""
    loaded_volume = Fraction(0)
    for i in range(instance["num_box_types"]):
        original_volume = (
            _exact(instance["box_lengths"][i])
            * _exact(instance["box_widths"][i])
            * _exact(instance["box_heights"][i])
        )
        loaded_volume += sum(solution["s"][i]) * original_volume
    container_volume = (
        _exact(instance["container_length"])
        * _exact(instance["container_width"])
        * _exact(instance["container_height"])
    )
    return float(-loaded_volume / container_volume)
