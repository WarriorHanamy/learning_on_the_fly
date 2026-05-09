"""ResAccScheme: simplest dynamics with learned NN residual acceleration."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from lotf.objects.quadrotor_state import QuadrotorState
from lotf.schemes._kernel import simplest_rk4_integrate
from lotf.utils.residual_dynamics import get_residual_dyn_model_apply_fn


class ResAccScheme:
    """Simplest dynamics plus NN residual acceleration.

    Forward::
        a = ap_z + res_acc_mean
        dv/dt = gravity + R @ [0, 0, a]
    """

    def __init__(self, res_model_params: FrozenDict):
        self._compute_residual = get_residual_dyn_model_apply_fn()
        self._res_model_params = res_model_params

    def integrate(
        self,
        ap_z: jax.Array,
        omega_d: jax.Array,
        state: QuadrotorState,
        dt: jax.Array,
    ) -> QuadrotorState:
        p, R, v = state.p, state.R, state.v
        state_for_res = jnp.array(
            [
                p[0],
                p[1],
                p[2],
                R[0, 0],
                R[0, 1],
                R[0, 2],
                R[1, 0],
                R[1, 1],
                R[1, 2],
                R[2, 0],
                R[2, 1],
                R[2, 2],
                v[0],
                v[1],
                v[2],
                ap_z,
                omega_d[0],
                omega_d[1],
                omega_d[2],
            ]
        )
        preds = self._compute_residual(self._res_model_params, state_for_res)
        res_acc_mean = jnp.mean(preds, axis=0)  # (3,) world-frame residual
        # Apply residual as world-frame acceleration alongside gravity
        gravity_with_res = jnp.array([0.0, 0.0, -9.81]) + res_acc_mean

        p_new, R_new, v_new = simplest_rk4_integrate(p, R, v, ap_z, omega_d, dt, gravity_with_res)

        scheme_state = dict(state.scheme_state)
        scheme_state["res_acc_mean"] = res_acc_mean
        return state.replace(p=p_new, R=R_new, v=v_new, scheme_state=scheme_state)
