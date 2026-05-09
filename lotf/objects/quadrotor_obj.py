from __future__ import annotations

import os
import numpy as np
import yaml
from functools import partial

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import chex
from flax.core import FrozenDict

from lotf import LOTF_PATH
from lotf.simulation.model_rotor import AugmentationParams, compute_residuals
from lotf.utils.pytrees import field_jnp, CustomPyTree
from lotf.utils.residual_dynamics import get_residual_dyn_model_apply_fn
from lotf.utils.math import rotation_matrix_from_vector


# betaflight constants
P_GAIN_SCALING = 1.818e-3
I_GAIN_SCALING = 16.67e-6
D_GAIN_SCALING = -31.25e-6

# sbus constants
SBUS_MIN_VAL = 192
SBUS_MAX_VAL = 1792
SBUS_VAL_RANGE = SBUS_MAX_VAL - SBUS_MIN_VAL


@jdc.pytree_dataclass
class QuadrotorState(CustomPyTree):
    """
    Data structure representing the full state of a quadrotor.

    Attributes:
        p: position in world frame
        R: rotation matrix (body to world)
        v: velocity in world frame
        omega: angular velocity in body frame
        domega: angular acceleration in body frame
        motor_omega: angular velocities of the 4 motors
        acc: linear acceleration in world frame
        res_acc_mean: predicted residual acceleration from learned model
        dr_key: random key for domain randomization
    """

    p: jax.Array = field_jnp([0.0, 0.0, 0.0])
    R: jax.Array = field_jnp(jnp.eye(3))
    v: jax.Array = field_jnp([0.0, 0.0, 0.0])
    omega: jax.Array = field_jnp([0.0, 0.0, 0.0])
    domega: jax.Array = field_jnp([0.0, 0.0, 0.0])
    motor_omega: jax.Array = field_jnp([0.0, 0.0, 0.0, 0.0])
    acc: jax.Array = field_jnp([0.0, 0.0, 0.0])
    res_acc_mean: jax.Array = field_jnp([0.0, 0.0, 0.0])
    dr_key: chex.PRNGKey = field_jnp(jax.random.key(0))
    # inner-loop approximation state
    approx_delay_buffer: jax.Array = field_jnp(jnp.zeros((10, 4)))
    approx_delay_idx: jax.Array = field_jnp(0)

    def detached(self):
        """Returns a copy of the state with gradients stopped."""
        return QuadrotorState(
            p=jax.lax.stop_gradient(self.p),
            R=jax.lax.stop_gradient(self.R),
            v=jax.lax.stop_gradient(self.v),
            omega=jax.lax.stop_gradient(self.omega),
            domega=jax.lax.stop_gradient(self.domega),
            motor_omega=jax.lax.stop_gradient(self.motor_omega),
            acc=jax.lax.stop_gradient(self.acc),
            res_acc_mean=jax.lax.stop_gradient(self.res_acc_mean),
            dr_key=jax.lax.stop_gradient(self.dr_key),
            approx_delay_buffer=jax.lax.stop_gradient(self.approx_delay_buffer),
            approx_delay_idx=jax.lax.stop_gradient(self.approx_delay_idx),
        )

    def as_vector(self):
        """Serializes the state into a flat jax array."""
        return jnp.concatenate(
            [self.p, self.R.flatten(), self.v, self.omega, self.domega, self.motor_omega]
        )

    @classmethod
    def from_vector(cls, vector):
        """Reconstructs state from a flattened array."""
        p = vector[:3]
        R = vector[3:12].reshape(3, 3)
        v = vector[12:15]
        omega = vector[15:18]
        domega = vector[18:21]
        motor_omega = vector[21:]
        return cls(p, R, v, omega, domega, motor_omega)


class Quadrotor:
    """
    Quadrotor model supporting nominal and inner-loop dynamics with optional residual acceleration.

    This class handles the simulation of quadrotor physics, including inner-loop
    control (Betaflight-style), motor dynamics, and optional learned residual acceleration.
    """

    def __init__(
        self,
        *,
        drone_name="example_quad",
        mass=0.75,
        tbm_fr=jnp.array([0.075, -0.10, 0.0]),
        tbm_bl=jnp.array([-0.075, 0.10, 0.0]),
        tbm_br=jnp.array([-0.075, -0.10, 0.0]),
        tbm_fl=jnp.array([0.075, 0.10, 0.0]),
        inertia=jnp.array([0.002410, 0.001800, 0.003759]),
        motor_omega_min=150.0,
        motor_omega_max=2800.0,
        motor_tau=0.033,
        motor_inertia=5.64e-6,
        omega_max=jnp.array([10.0, 10.0, 4.0]),
        thrust_map=jnp.array([1.562522e-6, 0.0, 0.0]),
        kappa=0.022,
        thrust_min=0.0,
        thrust_max=8.5,
        rotors_config="cross",
        dt_low_level=0.001,
        forward_model_config=None,
    ):
        """Initializes the quadrotor physical parameters and configuration."""
        assert rotors_config == "cross", "Only cross rotors configuration is supported"

        # physical constants
        self._drone_name = drone_name
        self._mass = mass
        self._tbm_fr = tbm_fr
        self._tbm_bl = tbm_bl
        self._tbm_br = tbm_br
        self._tbm_fl = tbm_fl
        self._inertia = inertia
        self._motor_omega_min = motor_omega_min
        self._motor_omega_max = motor_omega_max
        self._motor_tau = motor_tau
        self._motor_inertia = motor_inertia
        self._omega_max = omega_max
        self._thrust_map = thrust_map
        self._kappa = kappa
        self._thrust_min = thrust_min

        # initialize thrust floor
        if thrust_min <= 0.0:
            self._thrust_min += thrust_map[0] * motor_omega_min**2

        self._thrust_max = thrust_max
        self._rotors_config = rotors_config
        self._dt_low_level = dt_low_level
        self._gravity = jnp.array([0, 0, -9.81])

        # body rates pd parameters
        self._kp = jnp.array([40.0, 40.0, 30.0])
        self._kd = jnp.array([20.0, 20.0, 0.0]) * 0.1

        # load rotor dynamics augmentation
        self.rotor_augmentation_model = self.get_rotor_model()

        # simulation configuration
        if forward_model_config is None:
            forward_model_config = {
                "enable_inner_loop_dynamics": False,
                "enable_residual_acceleration": False,
            }
        self._enable_inner_loop_dynamics = forward_model_config["enable_inner_loop_dynamics"]
        self._enable_residual_acceleration = forward_model_config["enable_residual_acceleration"]
        self._enable_inner_loop_approx = forward_model_config.get("enable_inner_loop_approx", False)

        # load approximated inner-loop parameters from chirp analysis
        self._approx_K: jnp.ndarray | None = None
        self._approx_tau: jnp.ndarray | None = None
        self._approx_delay: jnp.ndarray | None = None
        self._approx_max_delay: int = 0
        if self._enable_inner_loop_approx:
            approx_path = forward_model_config.get("inner_loop_approx_path", None)
            if approx_path is None:
                raise ValueError(
                    "inner_loop_approx_path is required when enable_inner_loop_approx=True"
                )
            import json

            with open(approx_path) as f:
                data = json.load(f)
            channels = {ch["channel"]: ch for ch in data["channels"]}
            self._approx_K = jnp.array([channels[c]["K"] for c in ["thrust", "p", "q", "r"]])
            self._approx_tau = jnp.array(
                [max(channels[c]["tau"], 1e-6) for c in ["thrust", "p", "q", "r"]]
            )
            self._approx_delay = jnp.array(
                [channels[c]["delay"] for c in ["thrust", "p", "q", "r"]]
            )
            # buffer size: max delay + ceil(1 step) for safety
            max_delay_s = float(jnp.max(self._approx_delay))
            self._approx_max_delay = max(int(jnp.ceil(max_delay_s / 0.02)), 1)

        # load inference function for residual acceleration
        if self._enable_residual_acceleration:
            self._compute_residual_acceleration_fn = get_residual_dyn_model_apply_fn()

    @classmethod
    def from_name(cls, name: str, forward_model_config=None) -> Quadrotor:
        """Loads quadrotor parameters by name from the quadrotor_files directory."""
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "quadrotor_files/")
        filename += f"{name}.yaml"
        return cls.from_yaml(filename, forward_model_config)

    @classmethod
    def from_yaml(cls, path: str, forward_model_config=None) -> Quadrotor:
        """Loads quadrotor parameters from a specific YAML file path."""
        with open(path) as stream:
            try:
                config = yaml.safe_load(stream)
                return cls.from_dict(config, forward_model_config)
            except yaml.YAMLError as exc:
                raise exc

    @classmethod
    def example_quadrotor(cls) -> Quadrotor:
        """Convenience method to load the default example quadrotor."""
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "quadrotor_files/example_quad.yaml")
        return cls.from_yaml(filename)

    @classmethod
    def from_dict(cls, config: dict, forward_model_config=None) -> Quadrotor:
        """Constructs a Quadrotor object from a configuration dictionary."""
        return cls(
            drone_name=config["name"],
            mass=config["mass"],
            tbm_fr=jnp.array(config["tbm_fr"]),
            tbm_bl=jnp.array(config["tbm_bl"]),
            tbm_br=jnp.array(config["tbm_br"]),
            tbm_fl=jnp.array(config["tbm_fl"]),
            inertia=jnp.array(config["inertia"]),
            motor_omega_min=config["motor_omega_min"],
            motor_omega_max=config["motor_omega_max"],
            motor_tau=config["motor_tau"],
            motor_inertia=config["motor_inertia"],
            omega_max=jnp.array(config["omega_max"]),
            thrust_map=jnp.array(config["thrust_map"]),
            kappa=config["kappa"],
            thrust_min=config["thrust_min"],
            thrust_max=config["thrust_max"],
            rotors_config=config["rotors_config"],
            forward_model_config=forward_model_config,
        )

    @property
    def nominal_motor_speed_given_hovering(self) -> float:
        """Calculates the theoretical motor speed required to hover [rad/s]."""
        return jnp.sqrt(self._mass * 9.81 / (4 * self._thrust_map[0]))

    def default_state(self) -> QuadrotorState:
        """Returns a state initialized with nominal motor speeds."""
        nominal_motor_speeds = jnp.ones(4) * self.nominal_motor_speed_given_hovering
        return QuadrotorState(motor_omega=nominal_motor_speeds)

    def create_state(self, p, R, v, **kwargs) -> QuadrotorState:
        """Helper to create a state with specific position/rotation/velocity."""
        nominal_motor_speed = jnp.ones(4) * self.nominal_motor_speed_given_hovering
        if "motor_omega" not in kwargs.keys():
            kwargs["motor_omega"] = nominal_motor_speed

        return QuadrotorState(p, R, v, **kwargs)

    @property
    def allocation_matrix(self):
        """
        Matrix mapping individual thrusts [f1, f2, f3, f4] to total thrust and body torques.
        Returns: 4x4 allocation matrix.
        """
        rotor_coordinates = jnp.stack([self._tbm_fr, self._tbm_bl, self._tbm_br, self._tbm_fl])
        x = rotor_coordinates[:, 0]
        y = rotor_coordinates[:, 1]

        return jnp.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                y,
                -x,
                self._kappa * jnp.array([-1.0, -1.0, 1.0, 1.0]),
            ],
            dtype=jnp.float32,
        )

    def inertial_matrix(self):
        """Returns the 3x3 diagonal inertia matrix."""
        return jnp.diag(self._inertia)

    @property
    def allocation_matrix_inv(self):
        """Inverse of the thrust allocation matrix."""
        return jnp.linalg.inv(self.allocation_matrix)

    def get_rotor_model(self):
        """Loads augmentation parameters for specialized rotor dynamics."""
        augmentation_files_path = LOTF_PATH + "/simulation/augmentation_files/"
        path = augmentation_files_path + "example_model_rotor.yaml"
        return AugmentationParams.from_yaml(path)

    def step(
        self,
        state: QuadrotorState,
        f_d: jax.Array,
        omega_d: jax.Array,
        res_model_params: FrozenDict,
        dt: jax.Array,
    ) -> QuadrotorState:
        """
        Main simulation step for the quadrotor.

        Args:
            state: current state
            f_d: total desired thrust [N]
            omega_d: desired body rates [rad/s]
            res_model_params: parameters for the neural network residual model
            dt: integration time step [s]

        Returns:
            next_state: updated state after dt
        """

        # compute learned residual acceleration if enabled
        if self._enable_residual_acceleration:
            p = state.p
            R = state.R
            v = state.v
            # prepare input vector for the mlp
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
                    f_d,
                    omega_d[0],
                    omega_d[1],
                    omega_d[2],
                ]
            )

            # compute residual acceleration mean
            preds = self._compute_residual_acceleration_fn(res_model_params, state_for_res)
            res_acc_mean = jnp.mean(preds, axis=0)
            state = state.replace(res_acc_mean=res_acc_mean)

        # define the forward step with custom jvp for backprop control
        @partial(jax.custom_jvp, nondiff_argnums=(3,))
        def _step(state: QuadrotorState, f_d, omega_d, dt):
            """Internal forward pass of the quadrotor dynamics."""

            # prevent small negative or zero dt issues
            dt = np.round(dt, 5)
            if dt <= 0.0:
                return state

            # approximated inner-loop: delayed first-order filter → nominal dynamics
            if self._enable_inner_loop_approx:
                # only filter body rates p/q/r (thrust passes through unfiltered)
                omega_d_arr = jnp.asarray(omega_d)
                u = omega_d_arr.reshape(3)  # (3,) [p, q, r]

                # push current body-rate input into delay buffer
                buf = state.approx_delay_buffer.at[:, :3]  # only use first 3 cols
                idx = state.approx_delay_idx.astype(jnp.int32)
                full_buf = state.approx_delay_buffer
                full_buf = full_buf.at[idx, :3].set(u)
                new_idx = (idx + 1) % full_buf.shape[0]

                # compute delayed input for rate channels
                n_delay = jnp.ceil(self._approx_delay[1:] / jnp.maximum(dt, 1e-9)).astype(jnp.int32)
                read_pos = (new_idx - 1 - n_delay) % full_buf.shape[0]
                u_delayed = full_buf[read_pos, jnp.arange(3)]

                # gain-only filter for body rates (tau ≈ 0)
                omega_filtered = self._approx_K[1:] * u_delayed  # (3,)

                # thrust passes through unfiltered
                f_d_used = jnp.asarray(f_d)

                # nominal dynamics with residual if enabled
                res = state.res_acc_mean if self._enable_residual_acceleration else jnp.zeros(3)
                p_new, R_new, v_new = self._integrate_nominal_dynamics_with_residual(
                    state.p,
                    state.R,
                    state.v,
                    f_d_used / self._mass,
                    res,
                    omega_filtered,
                    dt,
                )

                return state.replace(
                    p=p_new,
                    R=R_new,
                    v=v_new,
                    omega=omega_filtered,
                    approx_delay_buffer=full_buf,
                    approx_delay_idx=new_idx.astype(jnp.int32),
                )

                return state.replace(
                    p=p_new,
                    R=R_new,
                    v=v_new,
                    omega=omega_filtered,
                    approx_delay_buffer=buf,
                    approx_delay_idx=new_idx.astype(jnp.int32),
                )

            # inner-loop forward simulation using low-level control and rk4
            if self._enable_inner_loop_dynamics:

                def control_fn(state, _unused):
                    # compute motor commands from betaflight-style inner-loop controller
                    motor_omega_d = self._compute_inner_loop_motor_commands(
                        state, f_d, omega_d, self._dt_low_level
                    )
                    # integrate dynamics at the controller frequency
                    state = self._integrate_inner_loop_dynamics(
                        state, motor_omega_d, self._dt_low_level
                    )
                    return state, None

                # calculate number of steps required to reach dt
                N = np.ceil(dt / self._dt_low_level).item()
                assert np.isclose(N * self._dt_low_level, dt), (
                    f"dt ({dt}) must be a multiple of dt_low_level ({self._dt_low_level})"
                )

                state_new, _ = jax.lax.scan(control_fn, state, length=N)
                return state_new

            # nominal forward simulation (point mass + exact rotation)
            else:
                if self._enable_residual_acceleration:
                    p_new, R_new, v_new = self._integrate_nominal_dynamics_with_residual(
                        state.p, state.R, state.v, f_d / self._mass, state.res_acc_mean, omega_d, dt
                    )
                else:
                    p_new, R_new, v_new = integrate_nominal_dynamics(
                        state.p, state.R, state.v, f_d / self._mass, omega_d, dt
                    )

                return state.replace(p=p_new, R=R_new, v=v_new)

        @_step.defjvp
        def _step_jvp(dt, primals, tangents):
            """Custom gradient definition for the step function."""

            # unpack primals for forward pass
            state, f_d, omega_d = primals
            p, R, v = state.p, state.R, state.v

            # unpack tangents for gradient propagation
            state_dot, f_d_dot, omega_d_dot = tangents
            p_dot, R_dot, v_dot = state_dot.p, state_dot.R, state_dot.v

            # calculate primal next state
            state_new = _step(state, f_d, omega_d, dt)

            # gradient pass through nominal analytical dynamics (no residual, no inner-loop)
            primals_simple = (p, R, v, f_d / self._mass, omega_d, dt)
            tangents_simple = (p_dot, R_dot, v_dot, f_d_dot / self._mass, omega_d_dot, 0.0)
            _, tan_out = jax.jvp(integrate_nominal_dynamics, primals_simple, tangents_simple)

            p_tan, R_tan, v_tan = tan_out

            # decay factor for long-horizon stability (set to 1.0 by default)
            decay_factor = 1.0

            # update the tangent state for jvp output
            # Note: dr_key is excluded from gradient propagation (PRNG keys are non-differentiable)
            state_dot_new = state_dot.replace(
                p=decay_factor * p_tan,
                R=decay_factor * R_tan,
                v=decay_factor * v_tan,
            )

            return state_new, state_dot_new

        # execute the step
        return _step(state, f_d, omega_d, dt)

    @partial(jax.jit, static_argnums=(0,))
    def _integrate_nominal_dynamics_with_residual(
        self,
        p: jax.Array,
        R: jax.Array,
        v: jax.Array,
        a: jax.Array,
        a_res: jax.Array,
        omega: jax.Array,
        dt: jax.Array,
        gravity: jax.Array = jnp.array([0, 0, -9.81]),
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Integrates position and velocity using RK4 with a pre-computed residual acceleration."""

        def dynamics(rk4_p, rk4_v):
            dvdt = gravity + R @ jnp.array([0.0, 0.0, a]) + a_res
            dpdt = v
            return dpdt, dvdt

        # perform rk4 integration
        k1_p, k1_v = dynamics(p, v)
        k2_p, k2_v = dynamics(p + 0.5 * dt * k1_p, v + 0.5 * dt * k1_v)
        k3_p, k3_v = dynamics(p + 0.5 * dt * k2_p, v + 0.5 * dt * k2_v)
        k4_p, k4_v = dynamics(p + dt * k3_p, v + dt * k3_v)

        p_new = p + (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        v_new = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        # analytical rotation update
        R_delta = rotation_matrix_from_vector(dt * omega)
        R_new = R @ R_delta

        return p_new, R_new, v_new

    def _integrate_inner_loop_dynamics(self, state: QuadrotorState, motor_omega_d, dt):
        """Inner-loop physics integration (motor lag + full body dynamics) using RK4."""
        p = state.p
        R = state.R
        v = state.v
        omega = state.omega
        motor_omega = state.motor_omega
        res_acc_mean = state.res_acc_mean

        # prepare domain randomization
        key_thrust, key_drag = jax.random.split(state.dr_key)
        thrust_map = self._thrust_map[0]
        thrust_map = jax.random.uniform(
            key_thrust,
            thrust_map.shape,
            minval=0.85 * thrust_map,
            maxval=1.15 * thrust_map,
        )

        # calculate individual motor thrusts
        f = thrust_map * motor_omega**2
        f_vec = jnp.array([0, 0, jnp.sum(f)])

        # rotor model augmentation
        rotor_acc_residual = compute_residuals(state, self.rotor_augmentation_model)[0]

        # total acceleration in world frame
        acc = self._gravity + R @ f_vec / self._mass + rotor_acc_residual + res_acc_mean

        # rk4 for p and v
        def int_pv(p, v):
            return v, acc

        k1_p, k1_v = int_pv(p, v)
        k2_p, k2_v = int_pv(p + 0.5 * dt * k1_p, v + 0.5 * dt * k1_v)
        k3_p, k3_v = int_pv(p + 0.5 * dt * k2_p, v + 0.5 * dt * k2_v)
        k4_p, k4_v = int_pv(p + dt * k3_p, v + dt * k3_v)

        p_new = p + (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        v_new = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        # orientation update
        R_delta = rotation_matrix_from_vector(dt * omega)
        R_new = R @ R_delta

        # motor inertia effects
        dmotor_omega = 1 / self._motor_tau * (motor_omega_d - motor_omega)
        motor_directions = jnp.array([-1, -1, 1, 1])
        inertia_torque = jnp.array(
            [0, 0, (dmotor_omega * motor_directions).sum() * self._motor_inertia]
        )

        # angular dynamics
        J = self.inertial_matrix()
        J_inv = jnp.linalg.inv(J)
        f_T_and_tau = self.allocation_matrix @ f
        f_T, tau = f_T_and_tau[0], f_T_and_tau[1:]

        def int_omega(omega):
            return J_inv @ (tau - jnp.cross(omega, J @ omega) + inertia_torque)

        k1_omega = int_omega(omega)
        k2_omega = int_omega(omega + 0.5 * dt * k1_omega)
        k3_omega = int_omega(omega + 0.5 * dt * k2_omega)
        k4_omega = int_omega(omega + dt * k3_omega)

        omega_new = omega + (dt / 6.0) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)
        domega_new = int_omega(omega)

        # motor dynamic lag
        motor_omega_new = (motor_omega - motor_omega_d) * jnp.exp(
            -dt / self._motor_tau
        ) + motor_omega_d
        motor_omega_new = jnp.clip(motor_omega_new, self._motor_omega_min, self._motor_omega_max)

        return state.replace(
            p=p_new,
            R=R_new,
            v=v_new,
            omega=omega_new,
            domega=domega_new,
            motor_omega=motor_omega_new,
            acc=acc,
        )

    def _compute_inner_loop_motor_commands(self, state: QuadrotorState, f_T, omega_cmd, dt):
        """Builds motor commands from a Betaflight-style body rate controller."""
        omega = state.omega
        domega = state.domega

        # map force to throttle sbus signal
        sbus = self._force_to_sbus(f_T)
        throttle = (sbus - SBUS_MIN_VAL) / SBUS_VAL_RANGE

        # body rate pd controller
        Kp = P_GAIN_SCALING * self._kp
        Kd = D_GAIN_SCALING * self._kd
        Kd = Kd * 0.001 / dt  # adjust d-gain for control frequency
        torque = Kp * (omega_cmd - omega) + Kd * domega

        # allocation and mixer
        alpha = jnp.concatenate([throttle[None], torque])
        B_allocation = jnp.array([[1, -1, -1, -1], [1, 1, 1, -1], [1, -1, 1, 1], [1, 1, -1, 1]])
        motor_throttle = B_allocation @ alpha

        # conversion chain: throttle -> dshot -> motor rad/s
        dshot = self._throttle_to_dshot(motor_throttle)
        motor_omega_d = self._dshot_to_motor_speeds(dshot)

        return motor_omega_d

    def _force_to_sbus(self, force):
        """Maps physical force to internal SBUS command units."""
        coeffs = jnp.array(
            [-770.1619262695312, 982.5460205078125, -149.59286499023438, 4.386282444000244]
        )
        sbus = (
            coeffs[0] + coeffs[1] * jnp.sqrt(force + 1) + coeffs[2] * force + coeffs[3] * force**2
        )
        sbus = jnp.clip(sbus, 0, SBUS_MAX_VAL)
        return sbus

    def _throttle_to_dshot(self, throttle):
        """Maps normalized throttle [0, 1] to DShot digital signal."""
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

    def _dshot_to_motor_speeds(self, dshot):
        """Converts DShot values to motor angular velocities."""
        omega_cmd_sqrt = 59.673
        omega_cmd_lin = 0.7595
        omega_volt = 78.325
        omega_offset = -1658.0

        voltage = 17.0  # nominal voltage

        motor_speeds = (
            omega_offset
            + omega_volt * voltage
            + omega_cmd_lin * dshot
            + omega_cmd_sqrt * jnp.sqrt(dshot)
        )
        return jnp.clip(motor_speeds, self._motor_omega_min, self._motor_omega_max)

    def motor_omega_to_thrust(self, motor_omega):
        """Converts motor rad/s to total thrust force."""
        return self._thrust_map[0] * motor_omega**2

    def print_config(self):
        """Prints current simulation configuration to console."""
        print(f"[QUAD OBJ] Drone name: {self._drone_name}")
        print(f"[QUAD OBJ] Enable inner-loop dynamics: {self._enable_inner_loop_dynamics}")
        print(f"[QUAD OBJ] Enable residual acceleration: {self._enable_residual_acceleration}")


def integrate_nominal_dynamics(
    p: jax.Array,
    R: jax.Array,
    v: jax.Array,
    a: jax.Array,
    omega: jax.Array,
    dt: jax.Array,
    gravity: jax.Array = jnp.array([0, 0, -9.81]),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Quadrotor dynamics with RK4 integration for position and velocity.
    :param p: position
    :param R: orientation matrix
    :param v: velocity
    :param a: acceleration in body z direction
    :param omega: body rates
    :param dt: time step
    :param gravity: gravity vector
    :return: new position, orientation matrix, and velocity
    """

    def dynamics(p, v):
        dvdt = gravity + R @ jnp.array([0.0, 0.0, a])
        dpdt = v
        return dpdt, dvdt

    # RK4 for position and velocity
    k1_p, k1_v = dynamics(p, v)
    k2_p, k2_v = dynamics(p + 0.5 * dt * k1_p, v + 0.5 * dt * k1_v)
    k3_p, k3_v = dynamics(p + 0.5 * dt * k2_p, v + 0.5 * dt * k2_v)
    k4_p, k4_v = dynamics(p + dt * k3_p, v + dt * k3_v)

    p_new = p + (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
    v_new = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    # Exact step for orientation
    R_delta = rotation_matrix_from_vector(dt * omega)
    R_new = R @ R_delta

    return p_new, R_new, v_new
