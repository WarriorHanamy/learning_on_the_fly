"""Public filter classes: ButterworthLowpass, NotchFilter."""

from __future__ import annotations

import numpy as np

from ._design import butterworth_denominator
from ._state_space import build_ss_from_tf
from .schema import ButterworthLowpassConfig, NotchFilterConfig


class ButterworthLowpass:
    """N-th order Butterworth low-pass filter.

    Continuous-time transfer function:
        H(s) = 1 / B_N(s / omega_c)

    where omega_c = 2 * pi * cutoff_frequency_hz and B_N is the normalized
    Butterworth polynomial.
    """

    def __init__(self, config: ButterworthLowpassConfig) -> None:
        order = config.order
        if not (1 <= order <= 6):
            raise ValueError(f"order must be 1–6, got {order}")

        omega_c = 2.0 * np.pi * config.cutoff_frequency_hz
        den = butterworth_denominator(order)

        # Frequency-scale the denominator:  B_N(s) -> B_N(s / omega_c)
        den_scaled = den.copy()
        for i in range(order):
            den_scaled[i] = den_scaled[i] / (omega_c ** float(order - i))

        # Numerator is just 1
        num = np.zeros(order + 1, dtype=np.float64)
        num[order] = 1.0

        self._ss = build_ss_from_tf(num, den_scaled, config.sample_period_s)

    def update(self, u: float) -> float:
        """Apply one filter step. Returns filtered output."""
        return self._ss.update(u)

    def reset(self) -> None:
        """Reset internal state to zero."""
        self._ss.reset()

    def switch(self, u: float, y: float) -> None:
        """Bumpless transfer: set state so output matches y."""
        self._ss.switch(u, y)

    @property
    def y(self) -> float:
        """Last filter output."""
        return self._ss.y


class NotchFilter:
    """Second-order notch filter.

    Continuous-time transfer function:
        H(s) = depth_gain * (s^2 + omega_n^2) / (s^2 + bw_omega * s + omega_n^2)
    """

    def __init__(self, config: NotchFilterConfig) -> None:
        omega_n = 2.0 * np.pi * config.center_frequency_hz
        bw_omega = 2.0 * np.pi * config.bandwidth_frequency_hz

        ksi1 = config.bandwidth_frequency_hz
        ksi2 = config.bandwidth_frequency_hz / config.depth_gain

        num = np.array([1.0, 2.0 * ksi1 * omega_n, omega_n * omega_n], dtype=np.float64)
        den = np.array([1.0, 2.0 * ksi2 * omega_n, omega_n * omega_n], dtype=np.float64)

        self._ss = build_ss_from_tf(num, den, config.sample_period_s)

    def update(self, u: float) -> float:
        return self._ss.update(u)

    def reset(self) -> None:
        self._ss.reset()

    def switch(self, u: float, y: float) -> None:
        self._ss.switch(u, y)

    @property
    def y(self) -> float:
        return self._ss.y
