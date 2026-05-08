"""Tests for Differentiator."""

from dataclasses import replace

import numpy as np
import pytest

from advanced_control_lib import DEFAULT_DIFFERENTIATOR_CONFIG, Differentiator


class TestDifferentiator:
    # --- construction ---

    def test_construct_from_default(self) -> None:
        d = Differentiator(DEFAULT_DIFFERENTIATOR_CONFIG)
        assert d.y == 0.0

    def test_rejects_invalid_order(self) -> None:
        for bad in (0, 7, 10):
            with pytest.raises(ValueError):
                Differentiator(replace(DEFAULT_DIFFERENTIATOR_CONFIG, order=bad))

    def test_all_orders_construct(self) -> None:
        for order in range(1, 7):
            d = Differentiator(
                replace(
                    DEFAULT_DIFFERENTIATOR_CONFIG,
                    order=order,
                    cutoff_frequency_hz=5.0,
                    sample_period_s=0.01,
                )
            )
            assert d.y == 0.0

    # --- step response ---

    def test_step_response_converges_to_zero(self) -> None:
        d = Differentiator(
            replace(
                DEFAULT_DIFFERENTIATOR_CONFIG,
                order=2,
                cutoff_frequency_hz=5.0,
                sample_period_s=0.01,
            )
        )
        for _ in range(200):
            d.update(1.0)
        assert abs(d.y) < 0.01

    def test_step_response_all_orders_converge(self) -> None:
        for order in (1, 2, 3, 4, 5, 6):
            d = Differentiator(
                replace(
                    DEFAULT_DIFFERENTIATOR_CONFIG,
                    order=order,
                    cutoff_frequency_hz=5.0,
                    sample_period_s=0.01,
                )
            )
            for _ in range(300):
                d.update(3.0)
            assert abs(d.y) < 0.05, f"order={order}, y={d.y}"

    def test_negative_step_converges_to_zero(self) -> None:
        d = Differentiator(
            replace(
                DEFAULT_DIFFERENTIATOR_CONFIG,
                order=2,
                cutoff_frequency_hz=5.0,
                sample_period_s=0.01,
            )
        )
        for _ in range(200):
            d.update(-3.0)
        assert abs(d.y) < 0.01

    # --- ramp response ---

    def test_ramp_response_converges_to_one(self) -> None:
        d = Differentiator(
            replace(
                DEFAULT_DIFFERENTIATOR_CONFIG,
                order=2,
                cutoff_frequency_hz=5.0,
                sample_period_s=0.01,
            )
        )
        t = 0.0
        for _ in range(200):
            t += 0.01
            d.update(t)
        assert 0.9 <= d.y <= 1.1

    def test_negative_ramp_response(self) -> None:
        d = Differentiator(
            replace(
                DEFAULT_DIFFERENTIATOR_CONFIG,
                order=2,
                cutoff_frequency_hz=5.0,
                sample_period_s=0.01,
            )
        )
        t = 0.0
        for _ in range(200):
            t += 0.01
            d.update(-t)
        assert -1.1 <= d.y <= -0.9

    def test_scaled_ramp(self) -> None:
        """d/dt (k * t) = k."""
        d = Differentiator(
            replace(
                DEFAULT_DIFFERENTIATOR_CONFIG,
                order=2,
                cutoff_frequency_hz=10.0,
                sample_period_s=0.01,
            )
        )
        t = 0.0
        k = 5.0
        for _ in range(300):
            t += 0.01
            d.update(k * t)
        assert 4.8 <= d.y <= 5.2

    # --- sine response ---

    def test_sine_response_non_divergent(self) -> None:
        d = Differentiator(
            replace(
                DEFAULT_DIFFERENTIATOR_CONFIG,
                order=2,
                cutoff_frequency_hz=10.0,
                sample_period_s=0.01,
            )
        )
        for i in range(500):
            t = i * 0.01
            y = d.update(np.sin(2.0 * np.pi * 3.0 * t))
            assert abs(y) < 100.0

    # --- reset ---

    def test_reset_clears_state(self) -> None:
        d = Differentiator(
            replace(DEFAULT_DIFFERENTIATOR_CONFIG, cutoff_frequency_hz=5.0, sample_period_s=0.01)
        )
        for _ in range(50):
            d.update(1.0)
        d.reset()
        assert d.y == 0.0

    def test_reset_then_update_is_bounded(self) -> None:
        d = Differentiator(
            replace(
                DEFAULT_DIFFERENTIATOR_CONFIG,
                order=2,
                cutoff_frequency_hz=5.0,
                sample_period_s=0.01,
            )
        )
        for _ in range(100):
            d.update(1.0)
        d.reset()
        y1 = d.update(1.0)
        assert abs(y1) < 100.0  # not NaN or unbounded

    # --- switch ---

    def test_switch_sets_output(self) -> None:
        d = Differentiator(
            replace(DEFAULT_DIFFERENTIATOR_CONFIG, cutoff_frequency_hz=5.0, sample_period_s=0.01)
        )
        d.switch(1.0, 0.5)
        assert abs(d.y - 0.5) < 1e-6

    def test_switch_preserves_output_value(self) -> None:
        d = Differentiator(
            replace(DEFAULT_DIFFERENTIATOR_CONFIG, cutoff_frequency_hz=5.0, sample_period_s=0.01)
        )
        d.switch(1.0, 0.5)
        assert abs(d.y - 0.5) < 1e-6

    def test_switch_multiple_times(self) -> None:
        d = Differentiator(
            replace(DEFAULT_DIFFERENTIATOR_CONFIG, cutoff_frequency_hz=5.0, sample_period_s=0.01)
        )
        for target in (0.0, 1.2, -0.7):
            d.switch(0.0, target)
            assert abs(d.y - target) < 1e-6

    # --- property ---

    def test_y_is_readonly(self) -> None:
        d = Differentiator(
            replace(DEFAULT_DIFFERENTIATOR_CONFIG, cutoff_frequency_hz=5.0, sample_period_s=0.01)
        )
        d.update(1.0)
        with pytest.raises(AttributeError):
            d.y = 42.0  # type: ignore[misc]

    # --- high sample rate ---

    def test_high_sample_rate_step_response(self) -> None:
        d = Differentiator(
            replace(
                DEFAULT_DIFFERENTIATOR_CONFIG,
                order=2,
                cutoff_frequency_hz=20.0,
                sample_period_s=0.001,
            )
        )
        for _ in range(500):
            d.update(1.0)
        assert abs(d.y) < 0.01
