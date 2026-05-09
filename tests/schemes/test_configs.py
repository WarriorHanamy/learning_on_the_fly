"""Tests for scheme pydantic configs."""

import pytest
from pydantic import ValidationError

from lotf.schemes.configs import (
    QuadrotorParams,
    SimplestConfig,
    ResAccConfig,
    ApproxConfig,
    InnerLoopConfig,
    FullConfig,
)


class TestSimplestConfig:
    def test_default_name(self):
        cfg = SimplestConfig()
        assert cfg.name == "simplest"


class TestResAccConfig:
    def test_default_name(self):
        cfg = ResAccConfig()
        assert cfg.name == "resacc"


class TestApproxConfig:
    def test_validates_chirp_path_exists(self):
        ApproxConfig(chirp_path="audit/default_inner_loop_approx.json")

    def test_rejects_missing_chirp_path(self):
        with pytest.raises(ValidationError):
            ApproxConfig()  # chirp_path is required

    def test_rejects_nonexistent_chirp_path(self):
        with pytest.raises(ValidationError):
            ApproxConfig(chirp_path="nonexistent/path.json")


class TestInnerLoopConfig:
    def test_defaults(self):
        cfg = InnerLoopConfig()
        assert cfg.name == "inner_loop"
        assert cfg.dt_low_level == 0.001
        assert cfg.kp == (40.0, 40.0, 30.0)

    def test_custom_params(self):
        cfg = InnerLoopConfig(dt_low_level=0.002, kp=(50.0, 50.0, 40.0))
        assert cfg.dt_low_level == 0.002
        assert cfg.kp == (50.0, 50.0, 40.0)


class TestFullConfig:
    def test_defaults(self):
        cfg = FullConfig()
        assert cfg.name == "full"
        assert cfg.dt_low_level == 0.001


class TestQuadrotorParams:
    def test_minimal_required(self):
        params = QuadrotorParams(mass=0.75)
        assert params.mass == 0.75
        assert params.tbm_fr == (0.075, -0.10, 0.0)  # default
        assert params.inertia == (0.002410, 0.001800, 0.003759)  # default

    def test_nominal_motor_speed(self):
        params = QuadrotorParams(mass=0.75)
        speed = params.nominal_motor_speed_given_hovering
        assert speed > 0
        # Expected: sqrt(0.75 * 9.81 / (4 * 1.562522e-6)) ≈ 1085 rad/s
        assert 1000 < speed < 1200
