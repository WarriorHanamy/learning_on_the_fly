"""Public differentiator class."""

from __future__ import annotations

import numpy as np

from ._design import butterworth_denominator
from ._state_space import build_ss_from_tf
from .schema import DifferentiatorConfig


class Differentiator:
    """Band-limited differentiator.

    Continuous-time transfer function:
        H(s) = s / B_N(s / omega_c)

    where omega_c = 2 * pi * cutoff_frequency_hz and B_N is the normalized
    Butterworth polynomial.  The Butterworth denominator rolls off the
    differentiator gain at high frequencies.
    """

    def __init__(self, config: DifferentiatorConfig) -> None:
        order = config.order
        if not (1 <= order <= 6):
            raise ValueError(f"order must be 1–6, got {order}")

        omega_c = 2.0 * np.pi * config.cutoff_frequency_hz

        den = butterworth_denominator(order)
        den_scaled = den.copy()
        for i in range(order):
            den_scaled[i] = den_scaled[i] / (omega_c ** float(order - i))

        # Numerator: s term = 1.0, all other coefficients = 0
        # Coefficients: [s^N, s^{N-1}, ..., s, 1]
        num = np.zeros(order + 1, dtype=np.float64)
        num[order - 1] = 1.0  # s^1 coefficient

        self._ss = build_ss_from_tf(num, den_scaled, config.sample_period_s)

    def update(self, u: float) -> float:
        """Apply one differentiator step. Returns derivative estimate."""
        return self._ss.update(u)

    def reset(self) -> None:
        """Reset internal state to zero."""
        self._ss.reset()

    def switch(self, u: float, y: float) -> None:
        """Bumpless transfer."""
        self._ss.switch(u, y)

    @property
    def y(self) -> float:
        """Last output."""
        return self._ss.y
