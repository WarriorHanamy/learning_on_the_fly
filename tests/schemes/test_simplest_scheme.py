"""Integration tests for SimplestScheme."""

import jax
import jax.numpy as jnp
import pytest

from lotf.objects import Quadrotor, QuadrotorState
from lotf.schemes import build_scheme, SimplestConfig, QuadrotorParams


@pytest.fixture
def quad():
    params = QuadrotorParams(mass=0.75)
    scheme = build_scheme(SimplestConfig(), params)
    return Quadrotor(scheme, params)


@pytest.fixture
def hover_state(quad):
    return quad.create_state(
        p=jnp.array([0.0, 0.0, 1.5]),
        R=jnp.eye(3),
        v=jnp.zeros(3),
    )


class TestSimplestScheme:
    def test_hover_maintains_position(self, quad, hover_state):
        """With f_d = mass * g, position and velocity should stay constant."""
        f_hover = 9.81 * 0.75
        next_state = quad.step(hover_state, jnp.array(f_hover), jnp.zeros(3), jnp.array(0.02))
        assert jnp.allclose(next_state.p, hover_state.p, atol=1e-6)
        assert jnp.allclose(next_state.v, hover_state.v, atol=1e-6)

    def test_upward_thrust_accelerates(self, quad, hover_state):
        """Extra thrust should produce upward velocity."""
        f_up = 9.81 * 0.75 + 1.0  # hover + 1N → a = 1.3 m/s²
        next_state = quad.step(hover_state, jnp.array(f_up), jnp.zeros(3), jnp.array(0.02))
        assert next_state.v[2] > 0
        assert next_state.p[2] > hover_state.p[2]

    def test_zero_dt_noop(self, quad, hover_state):
        """dt = 0 should return unchanged state."""
        next_state = quad.step(hover_state, jnp.array(10.0), jnp.zeros(3), jnp.array(0.0))
        assert jnp.allclose(next_state.p, hover_state.p, atol=1e-6)
        assert jnp.allclose(next_state.v, hover_state.v, atol=1e-6)

    def test_state_replaces_p_r_v(self, quad, hover_state):
        """Result state should have updated p, R, v but keep scheme_state."""
        next_state = quad.step(hover_state, jnp.array(9.81 * 0.75), jnp.ones(3), jnp.array(0.02))
        assert next_state.p is not None
        assert next_state.R is not None
        assert next_state.v is not None
        assert isinstance(next_state.scheme_state, dict)

    def test_gradient_flows(self, quad, hover_state):
        """Gradient should propagate through step()."""

        def loss_fn(state):
            return jnp.sum(
                quad.step(state, jnp.array(9.81 * 0.75), jnp.zeros(3), jnp.array(0.02)).p
            )

        grads = jax.grad(loss_fn)(hover_state)
        assert jnp.allclose(grads.p, jnp.ones(3))
        assert not jnp.any(jnp.isnan(grads.p))

    def test_custom_jvp_consistent(self, quad, hover_state):
        """JVP should match finite differences."""
        f_d = jnp.array(9.81 * 0.75)
        omega_d = jnp.zeros(3)
        dt = jnp.array(0.02)

        # Use a large enough epsilon to avoid float32 truncation
        eps = 1.0
        f_up = f_d + eps
        orig_p = quad.step(hover_state, f_d, omega_d, dt).p
        pert_p = quad.step(hover_state, f_up, omega_d, dt).p
        numerical_dp_df = (pert_p[2] - orig_p[2]) / eps

        # Analytical gradient via jax.grad
        def pos_z(f):
            return quad.step(hover_state, f, omega_d, dt).p[2]

        analytical = jax.grad(pos_z)(f_d)

        assert jnp.allclose(numerical_dp_df, analytical, rtol=1e-1)
