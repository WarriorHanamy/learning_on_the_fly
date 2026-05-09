"""Core integration kernel shared by schemes and the Quadrotor JVP backward.

This module MUST NOT import from any other lotf subpackage to avoid
circular imports (quadrotor_obj ↔ schemes.simplest).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lotf.utils.math import rotation_matrix_from_vector


def simplest_rk4_integrate(
    p: jax.Array,
    R: jax.Array,
    v: jax.Array,
    ap_z: jax.Array,
    omega: jax.Array,
    dt: jax.Array,
    gravity: jax.Array = jnp.array([0.0, 0.0, -9.81]),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """RK4 integration of p and v with exact SO(3) rotation step.

    Used by:
    - SimplestScheme (forward)
    - Quadrotor._step_jvp (backward gradient)

    Parameters
    ----------
    p : position [m] (world frame).
    R : rotation matrix body→world.
    v : velocity [m/s] (world frame).
    ap_z : thrust-axis acceleration [m/s²] (scalar).
    omega : body rates [rad/s] (3,).
    dt : timestep [s].
    gravity : gravity vector [m/s²].

    Returns
    -------
    (p_new, R_new, v_new) tuple.
    """

    def _dynamics(rk4_p, rk4_v):
        dvdt = gravity + R @ jnp.array([0.0, 0.0, ap_z])
        dpdt = rk4_v
        return dpdt, dvdt

    k1_p, k1_v = _dynamics(p, v)
    k2_p, k2_v = _dynamics(p + 0.5 * dt * k1_p, v + 0.5 * dt * k1_v)
    k3_p, k3_v = _dynamics(p + 0.5 * dt * k2_p, v + 0.5 * dt * k2_v)
    k4_p, k4_v = _dynamics(p + dt * k3_p, v + dt * k3_v)

    p_new = p + (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
    v_new = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    R_delta = rotation_matrix_from_vector(dt * omega)
    R_new = R @ R_delta

    return p_new, R_new, v_new
