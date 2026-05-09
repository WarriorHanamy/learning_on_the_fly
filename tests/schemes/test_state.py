"""Tests for QuadrotorState."""

import jax
import jax.numpy as jnp

from lotf.objects.quadrotor_state import QuadrotorState


class TestQuadrotorState:
    def test_default_construction(self):
        state = QuadrotorState()
        assert jnp.allclose(state.p, jnp.zeros(3))
        assert jnp.allclose(state.R, jnp.eye(3))
        assert jnp.allclose(state.v, jnp.zeros(3))
        assert isinstance(state.scheme_state, dict)
        assert len(state.scheme_state) == 0

    def test_custom_values(self):
        state = QuadrotorState(
            p=jnp.array([1.0, 2.0, 3.0]),
            R=jnp.eye(3),
            v=jnp.array([0.1, 0.2, 0.3]),
            scheme_state={"inner_loop_motor_omega": jnp.ones(4) * 1000},
        )
        assert state.p[0] == 1.0
        assert state.scheme_state["inner_loop_motor_omega"][0] == 1000.0

    def test_detached_stops_gradient(self):
        state = QuadrotorState(p=jnp.array([1.0, 2.0, 3.0]), scheme_state={"a": jnp.array(5.0)})
        detached = state.detached()
        # detached values should be the same
        assert jnp.allclose(detached.p, state.p)
        assert jnp.allclose(detached.scheme_state["a"], state.scheme_state["a"])

    def test_as_vector_roundtrip(self):
        state = QuadrotorState(
            p=jnp.array([1.0, 2.0, 3.0]),
            R=jnp.eye(3),
            v=jnp.array([0.1, 0.2, 0.3]),
            scheme_state={
                "inner_loop_omega": jnp.array([0.1, 0.2, 0.3]),
                "inner_loop_domega": jnp.array([0.01, 0.02, 0.03]),
                "inner_loop_motor_omega": jnp.array([1000.0, 1000.0, 1000.0, 1000.0]),
            },
        )
        vec = state.as_vector()
        assert vec.shape == (25,)  # 3 + 9 + 3 + 3 + 3 + 4 = 25
        restored = QuadrotorState.from_vector(vec)
        assert jnp.allclose(restored.p, state.p)
        assert jnp.allclose(restored.R, state.R)
        assert jnp.allclose(restored.v, state.v)

    def test_scheme_state_mutable_dict(self):
        """scheme_state can carry scheme-specific data across steps."""
        state = QuadrotorState(scheme_state={"approx_delay_idx": jnp.array(3)})
        new_ss = dict(state.scheme_state)
        new_ss["approx_delay_idx"] = jnp.array(4)
        new_state = state.replace(scheme_state=new_ss)
        assert new_state.scheme_state["approx_delay_idx"] == 4
