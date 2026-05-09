"""Tests for scheme factory."""

import pytest
from flax.core import FrozenDict

from lotf.schemes import (
    build_scheme,
    SimplestConfig,
    ResAccConfig,
    ApproxConfig,
    ApproxResAccConfig,
    InnerLoopConfig,
    FullConfig,
    QuadrotorParams,
    Scheme,
)


@pytest.fixture
def params():
    return QuadrotorParams(mass=0.75)


@pytest.fixture
def dummy_res():
    return FrozenDict({})


class TestBuildScheme:
    def test_simplest(self, params):
        scheme = build_scheme(SimplestConfig(), params)
        assert isinstance(scheme, Scheme), f"Got {type(scheme)}"

    def test_resacc_requires_res_params(self, params):
        with pytest.raises(ValueError, match="res_model_params"):
            build_scheme(ResAccConfig(), params)

    def test_resacc(self, params, dummy_res):
        scheme = build_scheme(ResAccConfig(), params, dummy_res)
        assert isinstance(scheme, Scheme)

    def test_approx(self, params, dummy_res):
        cfg = ApproxConfig(chirp_path="audit/default_inner_loop_approx.json")
        scheme = build_scheme(cfg, params, dummy_res)
        assert isinstance(scheme, Scheme)

    def test_approx_resacc(self, params, dummy_res):
        cfg = ApproxResAccConfig(chirp_path="audit/default_inner_loop_approx.json")
        scheme = build_scheme(cfg, params, dummy_res)
        assert isinstance(scheme, Scheme)

    def test_inner_loop(self, params):
        scheme = build_scheme(InnerLoopConfig(), params)
        assert isinstance(scheme, Scheme)

    def test_full_requires_res_params(self, params):
        with pytest.raises(ValueError, match="res_model_params"):
            build_scheme(FullConfig(), params)

    def test_full(self, params, dummy_res):
        scheme = build_scheme(FullConfig(), params, dummy_res)
        assert isinstance(scheme, Scheme)

    def test_inner_loop_requires_params(self):
        with pytest.raises(ValueError, match="params"):
            build_scheme(InnerLoopConfig(), None)
