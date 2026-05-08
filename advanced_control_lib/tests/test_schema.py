"""Tests for schema: all config dataclasses, defaults, and replace ergonomics."""

from dataclasses import replace

import pytest

from advanced_control_lib.schema import (
    DEFAULT_BUTTERWORTH_LOWPASS_CONFIG,
    DEFAULT_DIFFERENTIATOR_CONFIG,
    DEFAULT_INTEGRAL_SATURATION_CONFIG,
    DEFAULT_LEAD_COMPENSATOR_CONFIG,
    DEFAULT_NOTCH_FILTER_CONFIG,
    DEFAULT_PI_CONTROLLER_CONFIG,
    DEFAULT_PI_LEAD_CONFIG,
    DEFAULT_SATURATION_CONFIG,
    ButterworthLowpassConfig,
    DifferentiatorConfig,
    IntegralSaturationConfig,
    LeadCompensatorConfig,
    NotchFilterConfig,
    PIControllerConfig,
    PILeadControllerConfig,
    SaturationConfig,
)


class TestSaturationConfig:
    def test_default_is_none(self) -> None:
        s = SaturationConfig()
        assert s.output_limits is None

    def test_custom_limits(self) -> None:
        s = SaturationConfig(output_limits=(-2.0, 3.0))
        assert s.output_limits == (-2.0, 3.0)

    def test_frozen(self) -> None:
        s = SaturationConfig()
        with pytest.raises(Exception):
            s.output_limits = (0.0, 1.0)  # type: ignore[misc]


class TestIntegralSaturationConfig:
    def test_default(self) -> None:
        s = IntegralSaturationConfig()
        assert s.output_limits is None
        assert s.integral_output_limits is None
        assert s.anti_windup_reset_gain == 1.0
        assert s.anti_windup_deadband == 0.0

    def test_frozen(self) -> None:
        s = IntegralSaturationConfig()
        with pytest.raises(Exception):
            s.anti_windup_reset_gain = 5.0  # type: ignore[misc]


class TestButterworthLowpassConfig:
    def test_default(self) -> None:
        c = ButterworthLowpassConfig()
        assert c.order == 2
        assert c.cutoff_frequency_hz == 8.0
        assert c.sample_period_s == 0.02

    def test_default_singleton_equals_default(self) -> None:
        assert DEFAULT_BUTTERWORTH_LOWPASS_CONFIG == ButterworthLowpassConfig()

    def test_replace(self) -> None:
        c = replace(DEFAULT_BUTTERWORTH_LOWPASS_CONFIG, order=4, cutoff_frequency_hz=20.0)
        assert c.order == 4
        assert c.cutoff_frequency_hz == 20.0
        assert c.sample_period_s == 0.02  # unchanged

    def test_frozen(self) -> None:
        c = ButterworthLowpassConfig()
        with pytest.raises(Exception):
            c.order = 3  # type: ignore[misc]


class TestDifferentiatorConfig:
    def test_default(self) -> None:
        c = DifferentiatorConfig()
        assert c.order == 2
        assert c.cutoff_frequency_hz == 10.0

    def test_default_singleton(self) -> None:
        assert DEFAULT_DIFFERENTIATOR_CONFIG == DifferentiatorConfig()


class TestLeadCompensatorConfig:
    def test_default(self) -> None:
        c = LeadCompensatorConfig()
        assert c.proportional_gain == 1.0
        assert c.zero_frequency_hz == 3.0
        assert c.pole_frequency_hz == 12.0

    def test_default_saturation_is_empty(self) -> None:
        c = LeadCompensatorConfig()
        assert c.saturation.output_limits is None


class TestPIControllerConfig:
    def test_default(self) -> None:
        c = PIControllerConfig()
        assert c.proportional_gain == 1.0
        assert c.integral_frequency_hz == 0.5

    def test_default_saturation(self) -> None:
        c = PIControllerConfig()
        assert c.saturation.anti_windup_reset_gain == 1.0


class TestPILeadControllerConfig:
    def test_default(self) -> None:
        c = PILeadControllerConfig()
        assert c.proportional_gain == 1.0
        assert c.integral_frequency_hz == 0.5
        assert c.lead_zero_frequency_hz == 3.0
        assert c.lead_pole_frequency_hz == 12.0
        assert c.sample_period_s == 0.02

    def test_default_singleton(self) -> None:
        assert DEFAULT_PI_LEAD_CONFIG == PILeadControllerConfig()

    def test_replace_nested_saturation(self) -> None:
        sat = IntegralSaturationConfig(output_limits=(-2.0, 2.0))
        c = replace(DEFAULT_PI_LEAD_CONFIG, proportional_gain=2.0, saturation=sat)
        assert c.proportional_gain == 2.0
        assert c.saturation.output_limits == (-2.0, 2.0)


class TestNotchFilterConfig:
    def test_default(self) -> None:
        c = NotchFilterConfig()
        assert c.center_frequency_hz == 50.0
        assert c.bandwidth_frequency_hz == 10.0
        assert c.depth_gain == 0.1

    def test_default_singleton(self) -> None:
        assert DEFAULT_NOTCH_FILTER_CONFIG == NotchFilterConfig()
