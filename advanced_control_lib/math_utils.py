"""Public math utilities."""

from __future__ import annotations

import numpy as np


def clamp(value: float, lower: float | None, upper: float | None) -> float:
    """Limit a scalar value to [lower, upper].

    If a bound is None, that direction is unconstrained.

    Args:
        value: input value.
        lower: minimum allowable value, or None.
        upper: maximum allowable value, or None.

    Returns:
        Clamped value.
    """
    if lower is not None:
        value = max(value, lower)
    if upper is not None:
        value = min(value, upper)
    return value
