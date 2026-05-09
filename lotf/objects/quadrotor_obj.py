"""Quadrotor thin bridge: policy output → scheme dispatch + JVP backward.

The Quadrotor is a minimal bridge that:
1. Converts policy output (f_d, omega_d) → scheme input (ap_z, omega_d)
2. Dispatches to the active integration scheme
3. Handles JVP backward through simplest model for gradient stability
"""

from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp
import yaml

from lotf.objects.quadrotor_state import QuadrotorState
from lotf.schemes.base import Scheme
from lotf.schemes.configs import QuadrotorParams
from lotf.schemes._kernel import simplest_rk4_integrate


class Quadrotor:
    """Thin bridge between policy output and integration scheme.

    Responsibilities:
    - Convert f_d → ap_z = f_d / mass (scheme convention)
    - Dispatch to active scheme's integrate()
    - JVP backward through simplest model

    Usage::

        scheme = build_scheme(config, params, res_params)
        quad = Quadrotor(scheme, params)
        next_state = quad.step(state, f_d, omega_d, dt)
    """

    def __init__(self, scheme: Scheme, params: QuadrotorParams):
        self._scheme = scheme
        self._mass = params.mass
        self._params = params

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def omega_max(self) -> jax.Array:
        return jnp.array(self._params.omega_max)

    @property
    def thrust_min(self) -> float:
        return self._params.thrust_min

    @property
    def thrust_max(self) -> float:
        return self._params.thrust_max

    @property
    def params(self) -> QuadrotorParams:
        return self._params

    @property
    def nominal_motor_speed_given_hovering(self) -> float:
        """Theoretical motor speed required to hover [rad/s]."""
        return float(jnp.sqrt(self._mass * 9.81 / (4 * self._params.thrust_map[0])))

    def step(
        self,
        state: QuadrotorState,
        f_d: jax.Array,
        omega_d: jax.Array,
        dt: jax.Array,
    ) -> QuadrotorState:
        """Advance the plant by dt.

        Parameters
        ----------
        state : QuadrotorState
            Current kinematic state + scheme_state dict.
        f_d : jax.Array
            Total desired thrust [N] (scalar).
        omega_d : jax.Array
            Desired body rates [rad/s] (3,).
        dt : jax.Array
            Timestep [s].

        Returns
        -------
        QuadrotorState
            Next state after integration.
        """
        ap_z = f_d / self._mass

        @partial(jax.custom_jvp, nondiff_argnums=(3,))
        def _step(state: QuadrotorState, ap_z: jax.Array, omega_d: jax.Array, dt: jax.Array):
            return self._scheme.integrate(ap_z, omega_d, state, dt)

        @_step.defjvp
        def _step_jvp(dt, primals, tangents):
            """Backward pass always uses simplest model for gradient stability."""
            state, ap_z, omega_d = primals
            state_dot, a_p_dot, omega_d_dot = tangents

            p, R, v = state.p, state.R, state.v
            p_dot, R_dot, v_dot = state_dot.p, state_dot.R, state_dot.v

            # forward primal through actual scheme
            state_new = _step(state, ap_z, omega_d, dt)

            # backward tangent through simplest model (no residual, no inner-loop)
            primals_simple = (p, R, v, ap_z, omega_d, dt)
            tangents_simple = (p_dot, R_dot, v_dot, a_p_dot, omega_d_dot, 0.0)
            _, tan_out = jax.jvp(simplest_rk4_integrate, primals_simple, tangents_simple)

            p_tan, R_tan, v_tan = tan_out
            decay_factor = 1.0

            state_dot_new = state_dot.replace(
                p=decay_factor * p_tan,
                R=decay_factor * R_tan,
                v=decay_factor * v_tan,
            )

            return state_new, state_dot_new

        return _step(state, ap_z, omega_d, dt)

    @classmethod
    def from_yaml(cls, path: str, scheme: Scheme) -> Quadrotor:
        """Load QuadrotorParams from YAML and construct bridge."""
        with open(path) as stream:
            config = yaml.safe_load(stream)
        params = QuadrotorParams(
            mass=config["mass"],
            tbm_fr=tuple(config["tbm_fr"]),
            tbm_bl=tuple(config["tbm_bl"]),
            tbm_br=tuple(config["tbm_br"]),
            tbm_fl=tuple(config["tbm_fl"]),
            inertia=tuple(config["inertia"]),
            motor_omega_min=config.get("motor_omega_min", 150.0),
            motor_omega_max=config.get("motor_omega_max", 2800.0),
            motor_tau=config.get("motor_tau", 0.033),
            motor_inertia=config.get("motor_inertia", 5.64e-6),
            omega_max=tuple(config.get("omega_max", [10.0, 10.0, 4.0])),
            thrust_map=tuple(config.get("thrust_map", [1.562522e-6, 0.0, 0.0])),
            kappa=config.get("kappa", 0.022),
            thrust_min=config.get("thrust_min", 0.0),
            thrust_max=config.get("thrust_max", 8.5),
            rotors_config=config.get("rotors_config", "cross"),
        )
        return cls(scheme, params)

    @classmethod
    def from_name(cls, name: str, scheme: Scheme) -> Quadrotor:
        """Load quadrotor params by name from quadrotor_files directory."""
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "quadrotor_files/") + f"{name}.yaml"
        return cls.from_yaml(filename, scheme)

    @classmethod
    def from_dict(cls, config: dict, scheme: Scheme) -> Quadrotor:
        """Construct from a configuration dictionary."""
        params = QuadrotorParams(
            mass=config["mass"],
            tbm_fr=tuple(config["tbm_fr"]),
            tbm_bl=tuple(config["tbm_bl"]),
            tbm_br=tuple(config["tbm_br"]),
            tbm_fl=tuple(config["tbm_fl"]),
            inertia=tuple(config["inertia"]),
            motor_omega_min=config.get("motor_omega_min", 150.0),
            motor_omega_max=config.get("motor_omega_max", 2800.0),
            motor_tau=config.get("motor_tau", 0.033),
            motor_inertia=config.get("motor_inertia", 5.64e-6),
            omega_max=tuple(config.get("omega_max", [10.0, 10.0, 4.0])),
            thrust_map=tuple(config.get("thrust_map", [1.562522e-6, 0.0, 0.0])),
            kappa=config.get("kappa", 0.022),
            thrust_min=config.get("thrust_min", 0.0),
            thrust_max=config.get("thrust_max", 8.5),
            rotors_config=config.get("rotors_config", "cross"),
        )
        return cls(scheme, params)

    def create_state(self, p, R, v, **kwargs) -> QuadrotorState:
        """Helper to create a state with kinematic + scheme_state fields.

        Keyword args are stored in scheme_state dict.
        Common inner_loop keys (omega, domega, motor_omega, dr_key) are
        also stored with ``inner_loop_`` prefix for scheme compatibility.
        """
        scheme_state = dict(kwargs)
        # Also add scheme-prefixed keys so tree_select works across reset/step
        _inner_prefix_map = {
            "omega": "inner_loop_omega",
            "domega": "inner_loop_domega",
            "motor_omega": "inner_loop_motor_omega",
            "acc": "inner_loop_acc",
            "dr_key": "inner_loop_dr_key",
        }
        for key, prefixed in _inner_prefix_map.items():
            if key in scheme_state:
                scheme_state[prefixed] = scheme_state[key]
        # Ensure scheme-state keys always exist so tree_select works across reset/step
        _inner_defaults = {}
        for key, default in [
            ("inner_loop_omega", jnp.zeros(3)),
            ("inner_loop_domega", jnp.zeros(3)),
            ("inner_loop_motor_omega", jnp.ones(4) * 150.0),
            ("inner_loop_acc", jnp.zeros(3)),
            ("res_acc_mean", jnp.zeros(3)),
            ("approx_delay_buffer", jnp.zeros((10, 4))),
            ("approx_delay_idx", jnp.array(0.0)),
        ]:
            if key not in scheme_state:
                _inner_defaults[key] = default
        scheme_state.update(_inner_defaults)
        # dr_key is int-typed (PRNG key) — only set if explicitly provided
        if "dr_key" in scheme_state and "inner_loop_dr_key" not in scheme_state:
            _inner_defaults["inner_loop_dr_key"] = scheme_state["dr_key"]
        if "inner_loop_dr_key" in scheme_state:
            _inner_defaults["inner_loop_dr_key"] = scheme_state["inner_loop_dr_key"]
        scheme_state.update(_inner_defaults)
        return QuadrotorState(p=p, R=R, v=v, scheme_state=scheme_state)

    def default_state(self) -> QuadrotorState:
        """Returns a state initialized with nominal motor speeds for inner_loop."""
        nominal_motor_speeds = jnp.ones(4) * self.nominal_motor_speed_given_hovering
        return QuadrotorState(scheme_state={"inner_loop_motor_omega": nominal_motor_speeds})

    def print_config(self):
        """Prints current simulation configuration."""
        print(f"[QUAD OBJ] Scheme: {type(self._scheme).__name__}")
