"""Public configuration schema for advanced_control_lib.

Each config dataclass documents the continuous-time transfer function,
parameter definitions, and units.  All frequency parameters use Hz;
implementations convert internally with omega = 2 * pi * frequency_hz.

every config is frozen and provides sensible defaults so that
DEFAULT_*_CONFIG can be constructed with zero arguments.  Users override
fields via dataclasses.replace.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ==============================================================================
# Saturation
# ==============================================================================


@dataclass(frozen=True)
class SaturationConfig:
    """Static output saturation.

    Mathematical operation:
        y_sat = clamp(y, output_limits[0], output_limits[1])

    Attributes:
        output_limits: lower and upper saturation bounds [native unit].
            None means no saturation.
    """

    output_limits: tuple[float, float] | None = None


@dataclass(frozen=True)
class IntegralSaturationConfig:
    """Output saturation with bounded integral contribution and anti-windup.

    Mathematical operation:
        y = clamp(y_p + y_i, output_limits[0], output_limits[1])
        y_i = clamp(y_i, integral_output_limits[0], integral_output_limits[1])

    Anti-windup reset: when the error drives the integral output toward zero
    and outside the deadband, the integrator gain is multiplied by
    anti_windup_reset_gain.

    Attributes:
        output_limits: controller output bounds [native unit].  None = no limit.
        integral_output_limits: integral term bounds [native unit].  None = no limit.
        anti_windup_reset_gain: integrator gain multiplier during reset [-].
        anti_windup_deadband: deadband for reset logic [native unit].
    """

    output_limits: tuple[float, float] | None = None
    integral_output_limits: tuple[float, float] | None = None
    anti_windup_reset_gain: float = 1.0
    anti_windup_deadband: float = 0.0


# ==============================================================================
# Filters
# ==============================================================================


@dataclass(frozen=True)
class ButterworthLowpassConfig:
    """N-th order Butterworth low-pass filter.

    Continuous-time transfer function:
        H(s) = 1 / B_N(s / omega_c)

    where:
        omega_c = 2 * pi * cutoff_frequency_hz
        B_N is the normalized Butterworth polynomial of order N.

    Attributes:
        order: filter order, 1–6 [-].
        cutoff_frequency_hz: -3 dB cutoff frequency [Hz].
        sample_period_s: discrete-time sample period [s].
    """

    order: int = 2
    cutoff_frequency_hz: float = 8.0
    sample_period_s: float = 0.02


@dataclass(frozen=True)
class NotchFilterConfig:
    """Second-order notch filter.

    Continuous-time transfer function:
        H(s) = depth_gain * (s^2 + omega_n^2) / (s^2 + bandwidth * s + omega_n^2)

    where:
        omega_n = 2 * pi * center_frequency_hz
        bandwidth = 2 * pi * bandwidth_frequency_hz

    Attributes:
        center_frequency_hz: notch center frequency [Hz].
        bandwidth_frequency_hz: notch bandwidth (omega_bw) [Hz].
        depth_gain: attenuation at the notch center [-].  0 < depth_gain <= 1.
        sample_period_s: discrete-time sample period [s].
    """

    center_frequency_hz: float = 50.0
    bandwidth_frequency_hz: float = 10.0
    depth_gain: float = 0.1
    sample_period_s: float = 0.02


# ==============================================================================
# Differentiator
# ==============================================================================


@dataclass(frozen=True)
class DifferentiatorConfig:
    """Band-limited differentiator.

    Continuous-time transfer function:
        H(s) = s / B_N(s / omega_c)

    where:
        omega_c = 2 * pi * cutoff_frequency_hz
        B_N is the normalized Butterworth polynomial of order N.

    Attributes:
        order: Butterworth denominator order, 1–6 [-].
        cutoff_frequency_hz: differentiator cutoff frequency [Hz].
        sample_period_s: discrete-time sample period [s].
    """

    order: int = 2
    cutoff_frequency_hz: float = 10.0
    sample_period_s: float = 0.02


# ==============================================================================
# Compensators
# ==============================================================================


@dataclass(frozen=True)
class LeadCompensatorConfig:
    """First-order lead compensator.

    Continuous-time transfer function:
        C(s) = K * (s / omega_z + 1) / (s / omega_p + 1)

    where:
        omega_z = 2 * pi * zero_frequency_hz
        omega_p = 2 * pi * pole_frequency_hz

    A lead compensator usually satisfies:
        pole_frequency_hz > zero_frequency_hz

    Attributes:
        proportional_gain: static gain K [-].
        zero_frequency_hz: compensator zero frequency [Hz].
        pole_frequency_hz: compensator pole frequency [Hz].
        sample_period_s: discrete-time sample period [s].
        saturation: optional output saturation.
    """

    proportional_gain: float = 1.0
    zero_frequency_hz: float = 3.0
    pole_frequency_hz: float = 12.0
    sample_period_s: float = 0.02
    saturation: SaturationConfig = field(default_factory=SaturationConfig)


@dataclass(frozen=True)
class PIControllerConfig:
    """Proportional-integral controller.

    Continuous-time transfer function:
        C(s) = Kp + Ki / s

    where:
        Ki = 2 * pi * integral_frequency_hz

    Attributes:
        proportional_gain: proportional gain Kp [-].
        integral_frequency_hz: integral corner frequency [Hz].
        sample_period_s: discrete-time sample period [s].
        saturation: output and integral saturation settings.
    """

    proportional_gain: float = 1.0
    integral_frequency_hz: float = 0.5
    sample_period_s: float = 0.02
    saturation: IntegralSaturationConfig = field(default_factory=IntegralSaturationConfig)


@dataclass(frozen=True)
class PILeadControllerConfig:
    """PI controller cascaded with a first-order lead compensator.

    Continuous-time transfer function:
        C(s) = (Kp + Ki / s)
             * (s / omega_z + 1) / (s / omega_p + 1)

    where:
        Ki = 2 * pi * integral_frequency_hz
        omega_z = 2 * pi * lead_zero_frequency_hz
        omega_p = 2 * pi * lead_pole_frequency_hz

    Attributes:
        proportional_gain: proportional gain Kp [-].
        integral_frequency_hz: integral corner frequency [Hz].
        lead_zero_frequency_hz: lead zero frequency [Hz].
        lead_pole_frequency_hz: lead pole frequency [Hz].
        sample_period_s: discrete-time sample period [s].
        saturation: output and integral saturation settings.
    """

    proportional_gain: float = 1.0
    integral_frequency_hz: float = 0.5
    lead_zero_frequency_hz: float = 3.0
    lead_pole_frequency_hz: float = 12.0
    sample_period_s: float = 0.02
    saturation: IntegralSaturationConfig = field(default_factory=IntegralSaturationConfig)


# ==============================================================================
# Default config singletons  (zero-argument constructible)
# ==============================================================================

DEFAULT_SATURATION_CONFIG = SaturationConfig()
DEFAULT_INTEGRAL_SATURATION_CONFIG = IntegralSaturationConfig()

DEFAULT_BUTTERWORTH_LOWPASS_CONFIG = ButterworthLowpassConfig()
DEFAULT_NOTCH_FILTER_CONFIG = NotchFilterConfig()
DEFAULT_DIFFERENTIATOR_CONFIG = DifferentiatorConfig()
DEFAULT_LEAD_COMPENSATOR_CONFIG = LeadCompensatorConfig()
DEFAULT_PI_CONTROLLER_CONFIG = PIControllerConfig()
DEFAULT_PI_LEAD_CONFIG = PILeadControllerConfig()
