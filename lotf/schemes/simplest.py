"""SimplestScheme: point-mass dynamics with ideal body-rate tracking."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lotf.objects.quadrotor_state import QuadrotorState
from lotf.schemes._kernel import simplest_rk4_integrate


class SimplestScheme:
    """Minimal quadrotor integration: point-mass RK4 + exact SO(3) rotation.

    Forward dynamics::
        dv/dt = gravity + R @ [0, 0, ap_z]
        dp/dt = v
        R_new = R @ exp(dt * omega_d)
    """

    def __init__(self, gravity: jax.Array = jnp.array([0.0, 0.0, -9.81])):
        self._gravity = gravity

    def integrate(
        self,
        ap_z: jax.Array,
        omega_d: jax.Array,
        state: QuadrotorState,
        dt: jax.Array,
    ) -> QuadrotorState:
        p_new, R_new, v_new = simplest_rk4_integrate(
            state.p, state.R, state.v, ap_z, omega_d, dt, self._gravity
        )
        return state.replace(p=p_new, R=R_new, v=v_new)
