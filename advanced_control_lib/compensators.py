"""Public compensator classes: LeadCompensator, PIController, PILeadController.

Continuous-time transfer functions are documented in schema.py.
"""

from __future__ import annotations

import numpy as np

from ._state_space import _StateSpaceSISO1st
from .math_utils import clamp
from .schema import (
    IntegralSaturationConfig,
    LeadCompensatorConfig,
    PIControllerConfig,
    PILeadControllerConfig,
    SaturationConfig,
)


class LeadCompensator:
    """First-order lead compensator.

    C(s) = K * (s / omega_z + 1) / (s / omega_p + 1)
    """

    def __init__(self, config: LeadCompensatorConfig) -> None:
        omega_z = 2.0 * np.pi * config.zero_frequency_hz
        omega_p = 2.0 * np.pi * config.pole_frequency_hz
        K = config.proportional_gain

        num = np.array([K / omega_z, K], dtype=np.float64)
        den = np.array([1.0 / omega_p, 1.0], dtype=np.float64)

        self._ss = _StateSpaceSISO1st.from_continuous_tf(num, den, config.sample_period_s)
        self._sat = config.saturation

    def update(self, u: float) -> float:
        y = self._ss.update(u)
        if self._sat is not None and self._sat.output_limits is not None:
            y = clamp(y, *self._sat.output_limits)
        return y

    def reset(self) -> None:
        self._ss.reset()

    def switch(self, u: float, y: float) -> None:
        self._ss.switch(u, y)

    @property
    def y(self) -> float:
        return self._ss.y


class PIController:
    """Proportional-integral controller.

    C(s) = Kp + Ki / s
    Ki = 2 * pi * integral_frequency_hz
    """

    def __init__(self, config: PIControllerConfig) -> None:
        self._kp = config.proportional_gain
        self._sat = config.saturation
        self._sample_period_s = config.sample_period_s
        self._ki_ref = 2.0 * np.pi * config.integral_frequency_hz

        self._int_ss = _build_integrator_ss(self._ki_ref, self._sample_period_s)

        self._p_term: float = 0.0
        self._i_term: float = 0.0
        self._y: float = 0.0

    def update(self, error: float) -> float:
        self._p_term = self._kp * error

        # --- anti-windup ---
        sat = self._sat
        if (
            sat is not None
            and sat.anti_windup_reset_gain != 1.0
            and sat.anti_windup_deadband >= 0.0
        ):
            prev_y_i = self._int_ss.y
            outside = abs(prev_y_i) > sat.anti_windup_deadband
            opposing = error * prev_y_i < 0.0
            if outside and opposing:
                self._rebuild_integrator(sat.anti_windup_reset_gain)

        self._int_ss.update(error)
        self._i_term = self._int_ss.y

        # --- clamp integrator ---
        if sat is not None and sat.integral_output_limits is not None:
            i_lo, i_hi = sat.integral_output_limits
            if self._i_term > i_hi:
                self._i_term = i_hi
                self._int_ss.switch(error, i_hi)
            elif self._i_term < i_lo:
                self._i_term = i_lo
                self._int_ss.switch(error, i_lo)

        self._y = self._p_term + self._i_term

        if sat is not None and sat.output_limits is not None:
            self._y = clamp(self._y, *sat.output_limits)

        return self._y

    def _rebuild_integrator(self, dynamic_factor: float) -> None:
        """Rebuild integrator SS with scaled gain, preserving state."""
        u_saved = self._int_ss._u
        y_saved = self._int_ss.y
        ki_new = self._ki_ref * dynamic_factor
        self._int_ss = _build_integrator_ss(ki_new, self._sample_period_s)
        self._int_ss.switch(u_saved, y_saved)

    def reset(self) -> None:
        self._int_ss.reset()
        self._p_term = 0.0
        self._i_term = 0.0
        self._y = 0.0

    def switch(self, error: float, output: float) -> None:
        self._p_term = self._kp * error
        ki_term = output - self._p_term

        sat = self._sat
        if sat is not None and sat.integral_output_limits is not None:
            ki_term = clamp(ki_term, *sat.integral_output_limits)

        self._int_ss.switch(error, ki_term)
        self._i_term = self._int_ss.y
        self._y = self._p_term + self._i_term

        if sat is not None and sat.output_limits is not None:
            self._y = clamp(self._y, *sat.output_limits)

    @property
    def y(self) -> float:
        return self._y

    @property
    def p_term(self) -> float:
        return self._p_term

    @property
    def i_term(self) -> float:
        return self._i_term


class PILeadController:
    """PI controller cascaded with a first-order lead compensator.

    C(s) = (Kp + Ki / s) * (s / omega_z + 1) / (s / omega_p + 1)
    """

    def __init__(self, config: PILeadControllerConfig) -> None:
        self._kp = config.proportional_gain
        self._sat = config.saturation
        self._sample_period_s = config.sample_period_s
        self._ki_ref = 2.0 * np.pi * config.integral_frequency_hz

        self._int_ss = _build_integrator_ss(self._ki_ref, self._sample_period_s)

        omega_z = 2.0 * np.pi * config.lead_zero_frequency_hz
        omega_p = 2.0 * np.pi * config.lead_pole_frequency_hz
        lead_num = np.array([1.0 / omega_z, 1.0], dtype=np.float64)
        lead_den = np.array([1.0 / omega_p, 1.0], dtype=np.float64)
        self._lead_ss = _StateSpaceSISO1st.from_continuous_tf(
            lead_num, lead_den, self._sample_period_s
        )

        self._p_term: float = 0.0
        self._i_term: float = 0.0
        self._lead_term: float = 0.0
        self._y: float = 0.0

    def update(self, error: float) -> float:
        # --- lead stage (no saturation) ---
        self._lead_ss.update(error)
        self._lead_term = self._lead_ss.y

        # --- PI stage on lead output ---
        self._p_term = self._kp * self._lead_term

        sat = self._sat
        if (
            sat is not None
            and sat.anti_windup_reset_gain != 1.0
            and sat.anti_windup_deadband >= 0.0
        ):
            prev_y_i = self._int_ss.y
            outside = abs(prev_y_i) > sat.anti_windup_deadband
            opposing = self._lead_term * prev_y_i < 0.0
            if outside and opposing:
                self._rebuild_integrator(sat.anti_windup_reset_gain)

        self._int_ss.update(self._lead_term)
        self._i_term = self._int_ss.y

        if sat is not None and sat.integral_output_limits is not None:
            i_lo, i_hi = sat.integral_output_limits
            if self._i_term > i_hi:
                self._i_term = i_hi
                self._int_ss.switch(self._lead_term, i_hi)
            elif self._i_term < i_lo:
                self._i_term = i_lo
                self._int_ss.switch(self._lead_term, i_lo)

        self._y = self._p_term + self._i_term

        if sat is not None and sat.output_limits is not None:
            self._y = clamp(self._y, *sat.output_limits)

        return self._y

    def _rebuild_integrator(self, dynamic_factor: float) -> None:
        u_saved = self._int_ss._u
        y_saved = self._int_ss.y
        ki_new = self._ki_ref * dynamic_factor
        self._int_ss = _build_integrator_ss(ki_new, self._sample_period_s)
        self._int_ss.switch(u_saved, y_saved)

    def reset(self) -> None:
        self._int_ss.reset()
        self._lead_ss.reset()
        self._p_term = 0.0
        self._i_term = 0.0
        self._lead_term = 0.0
        self._y = 0.0

    def switch(self, error: float, output: float) -> None:
        # lead pass-through (gain = 1.0)
        self._lead_ss.switch(error, error)
        self._lead_term = error

        self._p_term = self._kp * self._lead_term
        ki_term = output - self._p_term

        sat = self._sat
        if sat is not None and sat.integral_output_limits is not None:
            ki_term = clamp(ki_term, *sat.integral_output_limits)

        self._int_ss.switch(self._lead_term, ki_term)
        self._i_term = self._int_ss.y
        self._y = self._p_term + self._i_term

        if sat is not None and sat.output_limits is not None:
            self._y = clamp(self._y, *sat.output_limits)

    @property
    def y(self) -> float:
        return self._y

    @property
    def p_term(self) -> float:
        return self._p_term

    @property
    def i_term(self) -> float:
        return self._i_term

    @property
    def lead_term(self) -> float:
        return self._lead_term


def _build_integrator_ss(ki: float, sample_period_s: float) -> _StateSpaceSISO1st:
    """Build a 1st-order integrator state-space:  Ki / s."""
    num = np.array([0.0, ki], dtype=np.float64)
    den = np.array([1.0, 0.0], dtype=np.float64)
    return _StateSpaceSISO1st.from_continuous_tf(num, den, sample_period_s)
