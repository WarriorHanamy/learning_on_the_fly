"""ApproxScheme: chirp-fitted inner-loop approximation with delayed first-order filter."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from lotf.objects.quadrotor_state import QuadrotorState
from lotf.schemes._kernel import simplest_rk4_integrate
from lotf.utils.residual_dynamics import get_residual_dyn_model_apply_fn


class ApproxScheme:
    """Chirp-fitted inner-loop approximation.

    Body rates go through per-channel delayed first-order filter.
    Thrust passes through unfiltered.

    Forward::
        omega_filtered = approx_K[1:] * delayed(omega_d)
        a = ap_z + res_acc_mean
        dv/dt = gravity + R @ [0, 0, a]
    """

    def __init__(self, chirp_path: str, res_model_params: FrozenDict):
        chirp_path = Path(chirp_path)
        if not chirp_path.is_absolute():
            from lotf import LOTF_PATH

            chirp_path = Path(LOTF_PATH) / chirp_path
        with open(chirp_path) as f:
            data = json.load(f)
        channels = {ch["channel"]: ch for ch in data["channels"]}

        self._approx_K = jnp.array([channels[c]["K"] for c in ["thrust", "p", "q", "r"]])
        self._approx_tau = jnp.array(
            [max(channels[c]["tau"], 1e-6) for c in ["thrust", "p", "q", "r"]]
        )
        self._approx_delay = jnp.array([channels[c]["delay"] for c in ["thrust", "p", "q", "r"]])
        max_delay_s = float(jnp.max(self._approx_delay))
        self._approx_max_delay = max(int(jnp.ceil(max_delay_s / 0.02)), 1)

        self._compute_residual = get_residual_dyn_model_apply_fn()
        self._res_model_params = res_model_params

    def integrate(
        self,
        ap_z: jax.Array,
        omega_d: jax.Array,
        state: QuadrotorState,
        dt: jax.Array,
    ) -> QuadrotorState:
        ss = dict(state.scheme_state)

        delay_buffer = ss.get("approx_delay_buffer", jnp.zeros((self._approx_max_delay + 1, 4)))
        delay_idx = ss.get("approx_delay_idx", jnp.array(0.0))

        omega_d_arr = jnp.asarray(omega_d).reshape(3)
        u = omega_d_arr  # (3,) [p, q, r]

        idx = delay_idx.astype(jnp.int32)
        delay_buffer = delay_buffer.at[idx, :3].set(u)
        new_idx = (idx + 1) % delay_buffer.shape[0]

        n_delay = jnp.ceil(self._approx_delay[1:] / jnp.maximum(dt, 1e-9)).astype(jnp.int32)
        read_pos = (new_idx - 1 - n_delay) % delay_buffer.shape[0]
        u_delayed = delay_buffer[read_pos, jnp.arange(3)]  # (3, 3) per-channel

        omega_filtered = self._approx_K[1:] * u_delayed  # (3,) elementwise

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
        gravity_with_res = jnp.array([0.0, 0.0, -9.81]) + res_acc_mean

        p_new, R_new, v_new = simplest_rk4_integrate(
            p, R, v, ap_z, omega_filtered, dt, gravity_with_res
        )

        ss["approx_delay_buffer"] = delay_buffer
        ss["approx_delay_idx"] = new_idx.astype(jnp.float32)
        ss["res_acc_mean"] = res_acc_mean
        return state.replace(p=p_new, R=R_new, v=v_new, scheme_state=ss)
