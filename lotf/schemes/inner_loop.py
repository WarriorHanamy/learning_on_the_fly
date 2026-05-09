"""InnerLoopScheme: full Betaflight-style inner-loop dynamics with motor model."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lotf import LOTF_PATH
from lotf.objects.quadrotor_state import QuadrotorState
from lotf.schemes.configs import QuadrotorParams
from lotf.simulation.model_rotor import AugmentationParams, compute_residuals
from lotf.utils.math import rotation_matrix_from_vector

# betaflight constants
P_GAIN_SCALING = 1.818e-3
I_GAIN_SCALING = 16.67e-6
D_GAIN_SCALING = -31.25e-6

# sbus constants
SBUS_MIN_VAL = 192
SBUS_MAX_VAL = 1792
SBUS_VAL_RANGE = SBUS_MAX_VAL - SBUS_MIN_VAL


class InnerLoopScheme:
    """Full Betaflight-style inner-loop controller with motor dynamics and rotor model.

    Forward::
        1. Betaflight body-rate PD controller → motor throttle
        2. DShot signal conversion chain
        3. Motor lag dynamics (first-order)
        4. Full RK4 angular velocity integration
        5. Rotor polynomial residual
        6. RK4 translational integration
    """

    def __init__(
        self,
        params: QuadrotorParams,
        dt_low_level: float = 0.001,
        rotor_augmentation_path: str | None = None,
        omega_max: tuple[float, float, float] = (10.0, 10.0, 4.0),
        motor_tau: float = 0.033,
        motor_inertia: float = 5.64e-6,
        kp: tuple[float, float, float] = (40.0, 40.0, 30.0),
        kd: tuple[float, float, float] = (20.0, 20.0, 0.0),
    ):
        self._params = params
        self._dt_low_level = dt_low_level
        self._gravity = jnp.array([0.0, 0.0, -9.81])
        self._motor_tau = motor_tau
        self._motor_inertia = motor_inertia

        self._kp = jnp.array(kp)
        self._kd = jnp.array(kd) * 0.1

        if rotor_augmentation_path is None:
            rotor_augmentation_path = str(
                Path(LOTF_PATH) / "simulation" / "augmentation_files" / "example_model_rotor.yaml"
            )
        self.rotor_augmentation_model = AugmentationParams.from_yaml(rotor_augmentation_path)

        self._thrust_map = jnp.array(params.thrust_map)
        self._motor_omega_min = params.motor_omega_min
        self._motor_omega_max = params.motor_omega_max

    def _force_to_sbus(self, force: jax.Array) -> jax.Array:
        coeffs = jnp.array(
            [-770.1619262695312, 982.5460205078125, -149.59286499023438, 4.386282444000244]
        )
        sbus = (
            coeffs[0] + coeffs[1] * jnp.sqrt(force + 1) + coeffs[2] * force + coeffs[3] * force**2
        )
        return jnp.clip(sbus, 0, SBUS_MAX_VAL)

    def _throttle_to_dshot(self, throttle: jax.Array) -> jax.Array:
        bfl_min_throttle = 1025
        bfl_max_throttle = 2000
        bfl_dshot_offset = 0.055
        PWM_MIN_VAL = 1000
        PWM_MAX_VAL = 2000
        PWM_RANGE = PWM_MAX_VAL - PWM_MIN_VAL
        DSHOT_MIN_VAL = 48
        DSHOT_MAX_VAL = 2048
        DSHOT_RANGE = DSHOT_MAX_VAL - DSHOT_MIN_VAL

        min_throttle = (bfl_min_throttle - PWM_MIN_VAL) / PWM_RANGE
        max_throttle = (bfl_max_throttle - PWM_MIN_VAL) / PWM_RANGE
        dshot_offset = bfl_dshot_offset * DSHOT_RANGE
        dshot_slope = DSHOT_MAX_VAL - dshot_offset

        throttle = jnp.clip(throttle, min_throttle, max_throttle)
        dshot = throttle * dshot_slope + dshot_offset
        return jnp.maximum(dshot, 0)

    def _dshot_to_motor_speeds(self, dshot: jax.Array) -> jax.Array:
        omega_cmd_sqrt = 59.673
        omega_cmd_lin = 0.7595
        omega_volt = 78.325
        omega_offset = -1658.0
        voltage = 17.0

        motor_speeds = (
            omega_offset
            + omega_volt * voltage
            + omega_cmd_lin * dshot
            + omega_cmd_sqrt * jnp.sqrt(dshot)
        )
        return jnp.clip(motor_speeds, self._motor_omega_min, self._motor_omega_max)

    def _compute_motor_commands(
        self,
        omega: jax.Array,
        domega: jax.Array,
        f_T: jax.Array,
        omega_cmd: jax.Array,
        dt: jax.Array,
    ) -> jax.Array:
        sbus = self._force_to_sbus(f_T)
        throttle = (sbus - SBUS_MIN_VAL) / SBUS_VAL_RANGE

        Kp = P_GAIN_SCALING * self._kp
        Kd = D_GAIN_SCALING * self._kd
        Kd = Kd * 0.001 / dt
        torque = Kp * (omega_cmd - omega) + Kd * domega

        alpha = jnp.concatenate([throttle[None], torque])
        B_allocation = jnp.array([[1, -1, -1, -1], [1, 1, 1, -1], [1, -1, 1, 1], [1, 1, -1, 1]])
        motor_throttle = B_allocation @ alpha

        dshot = self._throttle_to_dshot(motor_throttle)
        motor_omega_d = self._dshot_to_motor_speeds(dshot)
        return motor_omega_d

    def _integrate_substep(self, state_dict: dict, motor_omega_d: jax.Array, dt: jax.Array) -> dict:
        p = state_dict["p"]
        R = state_dict["R"]
        v = state_dict["v"]
        omega = state_dict["omega"]
        motor_omega = state_dict["motor_omega"]
        dr_key = state_dict["dr_key"]

        key_thrust, key_drag = jax.random.split(dr_key)
        thrust_map = self._thrust_map[0]
        thrust_map = jax.random.uniform(
            key_thrust, thrust_map.shape, minval=0.85 * thrust_map, maxval=1.15 * thrust_map
        )

        f = thrust_map * motor_omega**2
        f_vec = jnp.array([0, 0, jnp.sum(f)])

        # build a minimal state for compute_residuals
        from lotf.objects.quadrotor_state import QuadrotorState as _QS

        tmp_state = _QS(p=p, R=R, v=v, scheme_state={"motor_omega": motor_omega})
        rotor_acc_residual = compute_residuals(tmp_state, self.rotor_augmentation_model)[0]

        acc = self._gravity + R @ f_vec / self._params.mass + rotor_acc_residual

        # rk4 for p and v
        def int_pv(_p, _v):
            return _v, acc

        k1_p, k1_v = int_pv(p, v)
        k2_p, k2_v = int_pv(p + 0.5 * dt * k1_p, v + 0.5 * dt * k1_v)
        k3_p, k3_v = int_pv(p + 0.5 * dt * k2_p, v + 0.5 * dt * k2_v)
        k4_p, k4_v = int_pv(p + dt * k3_p, v + dt * k3_v)
        p_new = p + (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        v_new = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        R_delta = rotation_matrix_from_vector(dt * omega)
        R_new = R @ R_delta

        dmotor_omega = 1 / self._motor_tau * (motor_omega_d - motor_omega)
        motor_directions = jnp.array([-1, -1, 1, 1])
        inertia_torque = jnp.array(
            [0, 0, (dmotor_omega * motor_directions).sum() * self._motor_inertia]
        )

        J = jnp.diag(jnp.array(self._params.inertia))
        J_inv = jnp.linalg.inv(J)
        f_T_and_tau = self._allocation_matrix() @ f
        tau = f_T_and_tau[1:]

        def int_omega(w):
            return J_inv @ (tau - jnp.cross(w, J @ w) + inertia_torque)

        k1_w = int_omega(omega)
        k2_w = int_omega(omega + 0.5 * dt * k1_w)
        k3_w = int_omega(omega + 0.5 * dt * k2_w)
        k4_w = int_omega(omega + dt * k3_w)
        omega_new = omega + (dt / 6.0) * (k1_w + 2 * k2_w + 2 * k3_w + k4_w)
        domega_new = int_omega(omega)

        motor_omega_new = (motor_omega - motor_omega_d) * jnp.exp(
            -dt / self._motor_tau
        ) + motor_omega_d
        motor_omega_new = jnp.clip(motor_omega_new, self._motor_omega_min, self._motor_omega_max)

        return {
            "p": p_new,
            "R": R_new,
            "v": v_new,
            "omega": omega_new,
            "domega": domega_new,
            "motor_omega": motor_omega_new,
            "acc": acc,
            "dr_key": key_drag,
        }

    def _allocation_matrix(self) -> jax.Array:
        tbm_fr = jnp.array(self._params.tbm_fr)
        tbm_bl = jnp.array(self._params.tbm_bl)
        tbm_br = jnp.array(self._params.tbm_br)
        tbm_fl = jnp.array(self._params.tbm_fl)
        rotor_coordinates = jnp.stack([tbm_fr, tbm_bl, tbm_br, tbm_fl])
        x = rotor_coordinates[:, 0]
        y = rotor_coordinates[:, 1]
        return jnp.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                y,
                -x,
                self._params.kappa * jnp.array([-1.0, -1.0, 1.0, 1.0]),
            ],
            dtype=jnp.float32,
        )

    def integrate(
        self,
        ap_z: jax.Array,
        omega_d: jax.Array,
        state: QuadrotorState,
        dt: jax.Array,
    ) -> QuadrotorState:
        dt = np.round(dt, 5)
        if dt <= 0.0:
            return state

        # extract inner-loop state from scheme_state dict
        ss = dict(state.scheme_state)
        omega = ss.get("inner_loop_omega", jnp.zeros(3))
        domega = ss.get("inner_loop_domega", jnp.zeros(3))
        motor_omega = ss.get("inner_loop_motor_omega", jnp.ones(4) * 150.0)
        dr_key = ss.get("inner_loop_dr_key", jax.random.key(0))

        f_d = ap_z * self._params.mass

        def _control_fn(carry, _unused):
            motor_omega_cmd = self._compute_motor_commands(
                carry["omega"], carry["domega"], f_d, omega_d, self._dt_low_level
            )
            new_carry = self._integrate_substep(carry, motor_omega_cmd, self._dt_low_level)
            return new_carry, None

        N = np.ceil(dt / self._dt_low_level).item()
        sub_state = {
            "p": state.p,
            "R": state.R,
            "v": state.v,
            "omega": omega,
            "domega": domega,
            "motor_omega": motor_omega,
            "dr_key": dr_key,
            "acc": ss.get("inner_loop_acc", jnp.zeros(3)),
        }
        final_sub, _ = jax.lax.scan(_control_fn, sub_state, length=N)

        ss["inner_loop_omega"] = final_sub["omega"]
        ss["inner_loop_domega"] = final_sub["domega"]
        ss["inner_loop_motor_omega"] = final_sub["motor_omega"]
        ss["inner_loop_acc"] = final_sub["acc"]
        ss["inner_loop_dr_key"] = final_sub["dr_key"]

        return state.replace(p=final_sub["p"], R=final_sub["R"], v=final_sub["v"], scheme_state=ss)
