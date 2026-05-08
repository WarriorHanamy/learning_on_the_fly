"""Tests for compensators: LeadCompensator, PIController, PILeadController."""

from dataclasses import replace

import pytest

from advanced_control_lib import (
    DEFAULT_LEAD_COMPENSATOR_CONFIG,
    DEFAULT_PI_CONTROLLER_CONFIG,
    DEFAULT_PI_LEAD_CONFIG,
    IntegralSaturationConfig,
    LeadCompensator,
    PIController,
    PILeadController,
    SaturationConfig,
)


# ============================================================================
# LeadCompensator
# ============================================================================


class TestLeadCompensator:
    def test_construct_from_default(self) -> None:
        lead = LeadCompensator(DEFAULT_LEAD_COMPENSATOR_CONFIG)
        assert lead.y == 0.0

    def test_step_response_converges_to_gain(self) -> None:
        lead = LeadCompensator(
            replace(DEFAULT_LEAD_COMPENSATOR_CONFIG, proportional_gain=2.0, sample_period_s=0.01)
        )
        for _ in range(200):
            lead.update(1.0)
        assert abs(lead.y - 2.0) < 0.02

    def test_negative_gain(self) -> None:
        lead = LeadCompensator(
            replace(DEFAULT_LEAD_COMPENSATOR_CONFIG, proportional_gain=-3.0, sample_period_s=0.01)
        )
        for _ in range(200):
            lead.update(1.0)
        assert abs(lead.y - (-3.0)) < 0.05

    def test_negative_input(self) -> None:
        lead = LeadCompensator(
            replace(DEFAULT_LEAD_COMPENSATOR_CONFIG, proportional_gain=1.5, sample_period_s=0.01)
        )
        for _ in range(200):
            lead.update(-2.0)
        assert abs(lead.y - (-3.0)) < 0.05

    def test_output_saturation_active(self) -> None:
        sat = SaturationConfig(output_limits=(-0.5, 0.5))
        lead = LeadCompensator(
            replace(
                DEFAULT_LEAD_COMPENSATOR_CONFIG,
                proportional_gain=2.0,
                sample_period_s=0.01,
                saturation=sat,
            )
        )
        for _ in range(200):
            y = lead.update(1.0)
        assert -0.5 <= y <= 0.5

    def test_reset_clears_state(self) -> None:
        lead = LeadCompensator(replace(DEFAULT_LEAD_COMPENSATOR_CONFIG, sample_period_s=0.01))
        for _ in range(50):
            lead.update(1.0)
        lead.reset()
        assert lead.y == 0.0

    def test_switch(self) -> None:
        lead = LeadCompensator(replace(DEFAULT_LEAD_COMPENSATOR_CONFIG, sample_period_s=0.01))
        lead.switch(1.0, 0.5)
        assert abs(lead.y - 0.5) < 1e-6

    def test_switch_multiple_times(self) -> None:
        lead = LeadCompensator(replace(DEFAULT_LEAD_COMPENSATOR_CONFIG, sample_period_s=0.01))
        for target in (0.0, 2.0, -1.5):
            lead.switch(0.0, target)
            assert abs(lead.y - target) < 1e-6

    def test_y_readonly(self) -> None:
        lead = LeadCompensator(replace(DEFAULT_LEAD_COMPENSATOR_CONFIG, sample_period_s=0.01))
        with pytest.raises(AttributeError):
            lead.y = 42.0  # type: ignore[misc]


# ============================================================================
# PIController
# ============================================================================


class TestPIController:
    def test_construct_from_default(self) -> None:
        pi = PIController(DEFAULT_PI_CONTROLLER_CONFIG)
        assert pi.y == 0.0
        assert pi.p_term == 0.0
        assert pi.i_term == 0.0

    def test_step_response_integrates(self) -> None:
        pi = PIController(
            replace(
                DEFAULT_PI_CONTROLLER_CONFIG,
                proportional_gain=1.0,
                integral_frequency_hz=0.5,
                sample_period_s=0.01,
            )
        )
        y0 = pi.update(1.0)
        values = [pi.update(1.0) for _ in range(5)]
        assert values[-1] > y0
        assert pi.p_term == pytest.approx(1.0)
        assert pi.i_term > 0.0

    def test_zero_integral_gain(self) -> None:
        """PI with Ki=0 behaves like pure P."""
        pi = PIController(
            replace(
                DEFAULT_PI_CONTROLLER_CONFIG,
                proportional_gain=2.0,
                integral_frequency_hz=0.0,
                sample_period_s=0.01,
            )
        )
        for _ in range(100):
            pi.update(1.0)
        assert pi.p_term == pytest.approx(2.0)
        assert pi.i_term == pytest.approx(0.0)

    def test_negative_error(self) -> None:
        pi = PIController(
            replace(
                DEFAULT_PI_CONTROLLER_CONFIG,
                proportional_gain=1.0,
                integral_frequency_hz=0.5,
                sample_period_s=0.01,
            )
        )
        for _ in range(50):
            pi.update(-2.0)
        assert pi.p_term == pytest.approx(-2.0)
        assert pi.i_term < 0.0

    def test_output_saturation(self) -> None:
        sat = IntegralSaturationConfig(output_limits=(-0.5, 0.5))
        pi = PIController(
            replace(
                DEFAULT_PI_CONTROLLER_CONFIG,
                proportional_gain=10.0,
                integral_frequency_hz=10.0,
                sample_period_s=0.01,
                saturation=sat,
            )
        )
        y = pi.update(1.0)
        assert -0.5 <= y <= 0.5

    def test_integral_saturation(self) -> None:
        sat = IntegralSaturationConfig(
            integral_output_limits=(-0.2, 0.2), output_limits=(-10.0, 10.0)
        )
        pi = PIController(
            replace(
                DEFAULT_PI_CONTROLLER_CONFIG,
                proportional_gain=0.0,
                integral_frequency_hz=10.0,
                sample_period_s=0.01,
                saturation=sat,
            )
        )
        for _ in range(100):
            pi.update(1.0)
        assert -0.2 <= pi.i_term <= 0.2

    def test_no_saturation(self) -> None:
        """No saturation config at all (None limits)."""
        sat = IntegralSaturationConfig()
        pi = PIController(
            replace(
                DEFAULT_PI_CONTROLLER_CONFIG,
                proportional_gain=1.0,
                integral_frequency_hz=10.0,
                sample_period_s=0.01,
                saturation=sat,
            )
        )
        for _ in range(100):
            pi.update(1.0)
        assert pi.i_term > 10.0  # unchecked growth

    def test_anti_windup_reset(self) -> None:
        """Anti-windup pulls integrator back faster when error changes sign."""
        sat = IntegralSaturationConfig(
            anti_windup_reset_gain=5.0,
            anti_windup_deadband=0.01,
            integral_output_limits=(-20.0, 20.0),
            output_limits=(-20.0, 20.0),
        )
        pi = PIController(
            replace(
                DEFAULT_PI_CONTROLLER_CONFIG,
                proportional_gain=0.0,
                integral_frequency_hz=5.0,
                sample_period_s=0.01,
                saturation=sat,
            )
        )
        for _ in range(50):
            pi.update(1.0)
        i_before = abs(pi.i_term)
        for _ in range(5):
            pi.update(-1.0)
        # With reset gain, integrator pulled back significantly
        # (without reset gain the decrease would be much slower)
        assert abs(pi.i_term) < i_before * 0.9

    def test_anti_windup_deadband(self) -> None:
        """Within deadband, anti-windup should not trigger."""
        sat = IntegralSaturationConfig(
            anti_windup_reset_gain=5.0,
            anti_windup_deadband=1000.0,  # very large deadband
            integral_output_limits=(-5.0, 5.0),
            output_limits=(-5.0, 5.0),
        )
        pi = PIController(
            replace(
                DEFAULT_PI_CONTROLLER_CONFIG,
                proportional_gain=0.0,
                integral_frequency_hz=5.0,
                sample_period_s=0.01,
                saturation=sat,
            )
        )
        for _ in range(50):
            pi.update(1.0)
        i_ref = pi.i_term
        # Opposite error, but within deadband — should still integrate
        pi.update(-1.0)
        assert abs(pi.i_term - i_ref) < 2.0  # small change expected

    def test_reset_clears_state(self) -> None:
        pi = PIController(replace(DEFAULT_PI_CONTROLLER_CONFIG, sample_period_s=0.01))
        for _ in range(50):
            pi.update(1.0)
        assert abs(pi.i_term) > 0.0
        pi.reset()
        assert pi.y == 0.0
        assert pi.i_term == 0.0
        assert pi.p_term == 0.0

    def test_switch_preserves_output(self) -> None:
        pi = PIController(
            replace(
                DEFAULT_PI_CONTROLLER_CONFIG,
                proportional_gain=1.0,
                integral_frequency_hz=0.5,
                sample_period_s=0.01,
            )
        )
        pi.switch(0.5, 1.5)
        assert abs(pi.y - 1.5) < 1e-6
        assert pi.p_term == pytest.approx(0.5)
        assert pi.i_term == pytest.approx(1.0)

    def test_switch_with_saturation(self) -> None:
        sat = IntegralSaturationConfig(integral_output_limits=(-0.5, 0.5))
        pi = PIController(
            replace(
                DEFAULT_PI_CONTROLLER_CONFIG,
                proportional_gain=1.0,
                integral_frequency_hz=0.5,
                sample_period_s=0.01,
                saturation=sat,
            )
        )
        # i_term = 3.0 - 1.0*2.0 = 1.0, clamped to 0.5
        pi.switch(2.0, 3.0)
        assert pi.i_term == pytest.approx(0.5)

    def test_y_readonly(self) -> None:
        pi = PIController(replace(DEFAULT_PI_CONTROLLER_CONFIG, sample_period_s=0.01))
        with pytest.raises(AttributeError):
            pi.y = 42.0  # type: ignore[misc]

    def test_p_term_readonly(self) -> None:
        pi = PIController(replace(DEFAULT_PI_CONTROLLER_CONFIG, sample_period_s=0.01))
        with pytest.raises(AttributeError):
            pi.p_term = 42.0  # type: ignore[misc]


# ============================================================================
# PILeadController
# ============================================================================


class TestPILeadController:
    def test_construct_from_default(self) -> None:
        pil = PILeadController(DEFAULT_PI_LEAD_CONFIG)
        assert pil.y == 0.0

    def test_step_response(self) -> None:
        pil = PILeadController(
            replace(
                DEFAULT_PI_LEAD_CONFIG,
                proportional_gain=1.0,
                integral_frequency_hz=0.5,
                sample_period_s=0.01,
            )
        )
        _y0 = pil.update(1.0)
        for _ in range(200):
            pil.update(1.0)
        assert pil.y > 2.0
        assert pil.lead_term != 0.0

    def test_negative_error(self) -> None:
        pil = PILeadController(
            replace(
                DEFAULT_PI_LEAD_CONFIG,
                proportional_gain=1.0,
                integral_frequency_hz=0.5,
                sample_period_s=0.01,
            )
        )
        for _ in range(200):
            pil.update(-0.5)
        assert pil.y < 0.0

    def test_output_saturation(self) -> None:
        sat = IntegralSaturationConfig(output_limits=(-0.3, 0.3))
        pil = PILeadController(
            replace(
                DEFAULT_PI_LEAD_CONFIG,
                proportional_gain=10.0,
                integral_frequency_hz=10.0,
                sample_period_s=0.01,
                saturation=sat,
            )
        )
        for _ in range(100):
            y = pil.update(1.0)
        assert -0.3 <= y <= 0.3

    def test_integral_saturation(self) -> None:
        sat = IntegralSaturationConfig(
            integral_output_limits=(-0.1, 0.1), output_limits=(-10.0, 10.0)
        )
        pil = PILeadController(
            replace(
                DEFAULT_PI_LEAD_CONFIG,
                proportional_gain=0.0,
                integral_frequency_hz=10.0,
                sample_period_s=0.01,
                saturation=sat,
            )
        )
        for _ in range(100):
            pil.update(1.0)
        assert -0.1 <= pil.i_term <= 0.1

    def test_anti_windup(self) -> None:
        sat = IntegralSaturationConfig(
            anti_windup_reset_gain=5.0,
            anti_windup_deadband=0.01,
            integral_output_limits=(-20.0, 20.0),
            output_limits=(-20.0, 20.0),
        )
        pil = PILeadController(
            replace(
                DEFAULT_PI_LEAD_CONFIG,
                proportional_gain=0.0,
                integral_frequency_hz=5.0,
                sample_period_s=0.01,
                saturation=sat,
            )
        )
        for _ in range(50):
            pil.update(1.0)
        i_before = abs(pil.i_term)
        for _ in range(5):
            pil.update(-1.0)
        assert abs(pil.i_term) < i_before * 0.9

    def test_reset_clears_state(self) -> None:
        pil = PILeadController(replace(DEFAULT_PI_LEAD_CONFIG, sample_period_s=0.01))
        for _ in range(50):
            pil.update(1.0)
        pil.reset()
        assert pil.y == 0.0
        assert pil.p_term == 0.0
        assert pil.i_term == 0.0
        assert pil.lead_term == 0.0

    def test_switch_preserves_output(self) -> None:
        pil = PILeadController(
            replace(
                DEFAULT_PI_LEAD_CONFIG,
                proportional_gain=1.0,
                integral_frequency_hz=0.5,
                sample_period_s=0.01,
            )
        )
        pil.switch(0.5, 1.5)
        assert abs(pil.y - 1.5) < 1e-6

    def test_switch_with_integral_saturation(self) -> None:
        sat = IntegralSaturationConfig(integral_output_limits=(-0.3, 0.3))
        pil = PILeadController(
            replace(
                DEFAULT_PI_LEAD_CONFIG,
                proportional_gain=1.0,
                integral_frequency_hz=0.5,
                sample_period_s=0.01,
                saturation=sat,
            )
        )
        pil.switch(2.0, 3.0)
        assert pil.i_term == pytest.approx(0.3)  # clamped

    def test_y_readonly(self) -> None:
        pil = PILeadController(replace(DEFAULT_PI_LEAD_CONFIG, sample_period_s=0.01))
        with pytest.raises(AttributeError):
            pil.y = 42.0  # type: ignore[misc]

    def test_p_term_readonly(self) -> None:
        pil = PILeadController(replace(DEFAULT_PI_LEAD_CONFIG, sample_period_s=0.01))
        with pytest.raises(AttributeError):
            pil.p_term = 42.0  # type: ignore[misc]

    def test_lead_term_readonly(self) -> None:
        pil = PILeadController(replace(DEFAULT_PI_LEAD_CONFIG, sample_period_s=0.01))
        with pytest.raises(AttributeError):
            pil.lead_term = 42.0  # type: ignore[misc]
