"""Tests for filters: ButterworthLowpass, NotchFilter."""

from dataclasses import replace

import numpy as np
import pytest

from advanced_control_lib import (
    DEFAULT_BUTTERWORTH_LOWPASS_CONFIG,
    DEFAULT_NOTCH_FILTER_CONFIG,
    ButterworthLowpass,
    NotchFilter,
)


class TestButterworthLowpass:
    # --- construction ---

    def test_construct_from_default(self) -> None:
        lp = ButterworthLowpass(DEFAULT_BUTTERWORTH_LOWPASS_CONFIG)
        assert lp.y == 0.0

    def test_construct_custom_order(self) -> None:
        for order in range(1, 7):
            cfg = replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG,
                order=order,
                cutoff_frequency_hz=5.0,
                sample_period_s=0.01,
            )
            lp = ButterworthLowpass(cfg)
            assert lp.y == 0.0

    def test_rejects_invalid_order(self) -> None:
        for bad in (0, 7, 10):
            with pytest.raises(ValueError):
                ButterworthLowpass(replace(DEFAULT_BUTTERWORTH_LOWPASS_CONFIG, order=bad))

    # --- step response ---

    def test_step_response_converges_to_one(self) -> None:
        lp = ButterworthLowpass(
            replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG,
                order=2,
                cutoff_frequency_hz=5.0,
                sample_period_s=0.01,
            )
        )
        for _ in range(200):
            lp.update(1.0)
        assert 0.98 <= lp.y <= 1.02

    def test_step_response_all_orders(self) -> None:
        for order in range(1, 7):
            lp = ButterworthLowpass(
                replace(
                    DEFAULT_BUTTERWORTH_LOWPASS_CONFIG,
                    order=order,
                    cutoff_frequency_hz=5.0,
                    sample_period_s=0.01,
                )
            )
            for _ in range(300):
                lp.update(1.0)
            assert 0.9 <= lp.y <= 1.1, f"order={order}, y={lp.y}"

    def test_negative_step_response(self) -> None:
        lp = ButterworthLowpass(
            replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG,
                order=2,
                cutoff_frequency_hz=5.0,
                sample_period_s=0.01,
            )
        )
        for _ in range(200):
            lp.update(-2.0)
        assert -2.02 <= lp.y <= -1.98

    def test_zero_input_converges_to_zero(self) -> None:
        lp = ButterworthLowpass(
            replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG,
                order=2,
                cutoff_frequency_hz=5.0,
                sample_period_s=0.01,
            )
        )
        for _ in range(100):
            lp.update(1.0)
        assert abs(lp.y - 1.0) < 0.1
        for _ in range(200):
            lp.update(0.0)
        assert abs(lp.y) < 0.01

    def test_sine_response_bounded(self) -> None:
        """Ensure filter does not blow up under sinusoidal input."""
        lp = ButterworthLowpass(
            replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG,
                order=4,
                cutoff_frequency_hz=10.0,
                sample_period_s=0.01,
            )
        )
        for i in range(500):
            t = i * 0.01
            y = lp.update(np.sin(2.0 * np.pi * 3.0 * t))
            assert abs(y) < 10.0

    # --- reset ---

    def test_reset_clears_state(self) -> None:
        lp = ButterworthLowpass(
            replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG, cutoff_frequency_hz=5.0, sample_period_s=0.01
            )
        )
        for _ in range(50):
            lp.update(1.0)
        assert abs(lp.y) > 0.1
        lp.reset()
        assert lp.y == 0.0

    def test_reset_then_update_starts_from_zero(self) -> None:
        lp = ButterworthLowpass(
            replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG,
                order=2,
                cutoff_frequency_hz=5.0,
                sample_period_s=0.01,
            )
        )
        for _ in range(200):
            lp.update(1.0)
        lp.reset()
        y1 = lp.update(1.0)
        assert abs(y1) < 0.5  # first step after reset should be small

    # --- switch ---

    def test_switch_sets_output(self) -> None:
        lp = ButterworthLowpass(
            replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG, cutoff_frequency_hz=5.0, sample_period_s=0.01
            )
        )
        lp.switch(1.0, 0.5)
        assert abs(lp.y - 0.5) < 1e-6

    def test_switch_no_transient(self) -> None:
        lp = ButterworthLowpass(
            replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG, cutoff_frequency_hz=5.0, sample_period_s=0.01
            )
        )
        lp.switch(1.0, 0.5)
        y0 = lp.y
        y1 = lp.update(1.0)
        assert abs(y1 - y0) < 0.1

    def test_switch_multiple_times(self) -> None:
        lp = ButterworthLowpass(
            replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG, cutoff_frequency_hz=5.0, sample_period_s=0.01
            )
        )
        for target in (0.2, 0.8, -0.3, 1.5):
            lp.switch(0.0, target)
            assert abs(lp.y - target) < 1e-6

    def test_switch_then_drive_steady_state(self) -> None:
        lp = ButterworthLowpass(
            replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG,
                order=2,
                cutoff_frequency_hz=5.0,
                sample_period_s=0.01,
            )
        )
        lp.switch(2.0, 1.0)
        for _ in range(200):
            lp.update(2.0)
        assert 1.98 <= lp.y <= 2.02

    # --- high sample rate ---

    def test_high_sample_rate(self) -> None:
        lp = ButterworthLowpass(
            replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG,
                order=2,
                cutoff_frequency_hz=20.0,
                sample_period_s=0.001,
            )
        )
        for _ in range(500):
            lp.update(1.0)
        assert 0.99 <= lp.y <= 1.01

    # --- property ---

    def test_y_is_readonly(self) -> None:
        lp = ButterworthLowpass(
            replace(
                DEFAULT_BUTTERWORTH_LOWPASS_CONFIG, cutoff_frequency_hz=5.0, sample_period_s=0.01
            )
        )
        lp.update(1.0)
        val = lp.y
        with pytest.raises(AttributeError):
            lp.y = 42.0  # type: ignore[misc]


class TestNotchFilter:
    def test_construct_from_default(self) -> None:
        nf = NotchFilter(DEFAULT_NOTCH_FILTER_CONFIG)
        assert nf.y == 0.0

    def test_step_response_non_divergent(self) -> None:
        nf = NotchFilter(
            replace(
                DEFAULT_NOTCH_FILTER_CONFIG,
                center_frequency_hz=50.0,
                bandwidth_frequency_hz=10.0,
                depth_gain=0.1,
                sample_period_s=0.01,
            )
        )
        for _ in range(100):
            nf.update(1.0)
        assert abs(nf.y) < 10.0

    def test_dc_gain(self) -> None:
        """Notch should pass DC (H(0) != 0)."""
        nf = NotchFilter(
            replace(
                DEFAULT_NOTCH_FILTER_CONFIG,
                center_frequency_hz=50.0,
                bandwidth_frequency_hz=10.0,
                depth_gain=0.1,
                sample_period_s=0.01,
            )
        )
        for _ in range(200):
            nf.update(1.0)
        assert abs(nf.y) < 10.0  # bounded DC response

    def test_reset_clears_state(self) -> None:
        nf = NotchFilter(replace(DEFAULT_NOTCH_FILTER_CONFIG, sample_period_s=0.01))
        for _ in range(50):
            nf.update(1.0)
        nf.reset()
        assert nf.y == 0.0

    def test_switch(self) -> None:
        nf = NotchFilter(replace(DEFAULT_NOTCH_FILTER_CONFIG, sample_period_s=0.01))
        nf.switch(1.0, 0.3)
        assert abs(nf.y - 0.3) < 1e-6

    def test_negative_input(self) -> None:
        nf = NotchFilter(
            replace(
                DEFAULT_NOTCH_FILTER_CONFIG,
                center_frequency_hz=50.0,
                bandwidth_frequency_hz=10.0,
                depth_gain=0.5,
                sample_period_s=0.01,
            )
        )
        for _ in range(100):
            nf.update(-1.0)
        assert abs(nf.y) < 10.0
