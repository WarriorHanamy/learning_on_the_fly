"""Advanced Control Library — Python-only control primitives.

Public API provides filters, differentiators, compensators, and their
configuration schemas.  Implementation details are private.
"""

from advanced_control_lib.schema import (
    ButterworthLowpassConfig,
    DEFAULT_BUTTERWORTH_LOWPASS_CONFIG,
    DEFAULT_DIFFERENTIATOR_CONFIG,
    DEFAULT_INTEGRAL_SATURATION_CONFIG,
    DEFAULT_LEAD_COMPENSATOR_CONFIG,
    DEFAULT_NOTCH_FILTER_CONFIG,
    DEFAULT_PI_CONTROLLER_CONFIG,
    DEFAULT_PI_LEAD_CONFIG,
    DEFAULT_SATURATION_CONFIG,
    DifferentiatorConfig,
    IntegralSaturationConfig,
    LeadCompensatorConfig,
    NotchFilterConfig,
    PIControllerConfig,
    PILeadControllerConfig,
    SaturationConfig,
)
from advanced_control_lib.filters import ButterworthLowpass, NotchFilter
from advanced_control_lib.differentiators import Differentiator
from advanced_control_lib.compensators import (
    LeadCompensator,
    PIController,
    PILeadController,
)
from advanced_control_lib.math_utils import clamp

__all__ = [
    # configs
    "ButterworthLowpassConfig",
    "DifferentiatorConfig",
    "IntegralSaturationConfig",
    "LeadCompensatorConfig",
    "NotchFilterConfig",
    "PIControllerConfig",
    "PILeadControllerConfig",
    "SaturationConfig",
    # defaults
    "DEFAULT_BUTTERWORTH_LOWPASS_CONFIG",
    "DEFAULT_DIFFERENTIATOR_CONFIG",
    "DEFAULT_INTEGRAL_SATURATION_CONFIG",
    "DEFAULT_LEAD_COMPENSATOR_CONFIG",
    "DEFAULT_NOTCH_FILTER_CONFIG",
    "DEFAULT_PI_CONTROLLER_CONFIG",
    "DEFAULT_PI_LEAD_CONFIG",
    "DEFAULT_SATURATION_CONFIG",
    # tools
    "ButterworthLowpass",
    "NotchFilter",
    "Differentiator",
    "LeadCompensator",
    "PIController",
    "PILeadController",
    "clamp",
]
