"""Bridge between the local experiment schema and the LOTF simulator backend.

This is the ONLY file in ``examples/classic_control_hover`` allowed to
import from ``lotf.*``.  It translates between LOTF-native data and the
canonical ``StateSample`` / ``ControlModel`` contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial as _partial
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from orbax.checkpoint import PyTreeCheckpointer

from lotf import LOTF_ROOT
from lotf.objects.quadrotor_obj import Quadrotor
from lotf.objects.quadrotor_state import QuadrotorState
from lotf.schemes import build_scheme
from lotf.schemes.configs import (
    QuadrotorParams,
    SimplestConfig,
    ResAccConfig,
    ApproxConfig,
    ApproxResAccConfig,
    InnerLoopConfig,
    FullConfig,
)

from .schema import ControlModel, HoverTarget, StateSample


# ---------------------------------------------------------------------------
# adaptor-side configuration
# ---------------------------------------------------------------------------


@dataclass
class LotfAdapterConfig:
    """Backend-specific configuration for the LOTF adaptor.

    ``setting`` is the canonical LOTF forward-model name (e.g. ``"full"``).
    Experiment-layer code should never import or interpret this type directly.
    """

    dt: float = 0.02
    duration: float = 140.0  # [s]
    setting: str = "full"
    residual_checkpoint: Optional[str] = None
    approx_path: Optional[str] = None  # path to inner_loop_approx.json

    def num_steps(self) -> int:
        return int(self.duration / self.dt)


# ---------------------------------------------------------------------------
# JIT-compiled plant step
# ---------------------------------------------------------------------------


@_partial(jax.jit, static_argnums=(0, 4))
def _jit_step(quad, state, f_d, omega_d, dt_val):
    """JIT-compiled plant step — ``quad`` and ``dt_val`` are compile-time constants."""
    return quad.step(state, f_d, omega_d, dt_val)


# ---------------------------------------------------------------------------
# residual loading
# ---------------------------------------------------------------------------

_SETTING_NEEDS_REAL_RESIDUAL = {"resacc", "full"}
_DUMMY_RESIDUAL_PATH = LOTF_ROOT / "checkpoints" / "residual_dynamics" / "dummy_params"
_REAL_RESIDUAL_PATH = LOTF_ROOT / "checkpoints" / "residual_dynamics" / "residual_params"


def _load_residual_params(setting: str, checkpoint_override: str | None) -> FrozenDict:
    ckptr = PyTreeCheckpointer()
    if checkpoint_override is not None:
        path = Path(checkpoint_override)
        if not path.is_absolute():
            path = LOTF_ROOT / path
        return ckptr.restore(str(path))
    if setting in _SETTING_NEEDS_REAL_RESIDUAL:
        return ckptr.restore(str(_REAL_RESIDUAL_PATH))
    return ckptr.restore(str(_DUMMY_RESIDUAL_PATH))


# ---------------------------------------------------------------------------
# backend-native ↔ canonical conversion
# ---------------------------------------------------------------------------


def _R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → Hamilton quaternion [qw, qx, qy, qz]."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s])
    if m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        return np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s])
    if m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        return np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s])
    s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
    return np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s])


def _quad_state_to_sample(qs: QuadrotorState, thrust_coeff: float) -> StateSample:
    """Convert a LOTF ``QuadrotorState`` to a canonical ``StateSample``.

    Backend-specific fields (raw rotation matrix, angular acceleration,
    motor speeds, estimated thrust) are placed in ``extras``.
    """
    R_np = np.array(qs.R)
    ss = qs.scheme_state
    motor_omega_np = np.array(ss.get("inner_loop_motor_omega", np.zeros(4)))
    thrust_est = float(np.sum(thrust_coeff * motor_omega_np**2))

    return StateSample(
        p_world_m=np.array(qs.p),
        v_world_mps=np.array(qs.v),
        q_world_from_body_wxyz=_R_to_quat_wxyz(R_np),
        omega_body_radps=np.array(ss.get("inner_loop_omega", np.zeros(3))),
        acc_world_mps2=np.array(ss.get("inner_loop_acc", np.zeros(3))),
        extras={
            "R_world_from_body": R_np,
            "domega_body_radps2": np.array(ss.get("inner_loop_domega", np.zeros(3))),
            "motor_omega_radps": motor_omega_np,
            "thrust_est_N": thrust_est,
        },
    )


# ---------------------------------------------------------------------------
# SimAdapter
# ---------------------------------------------------------------------------


class SimAdapter:
    """Owns a LOTF ``Quadrotor`` and bridges local experiment ↔ LOTF backend.

    Usage::

        adapter = SimAdapter(LotfAdapterConfig(setting="full"))
        model = adapter.control_model
        sample = adapter.initialize(HoverTarget((0, 0, 1.5), 0.0))
        ...
        sample = adapter.step(action_array)
    """

    def __init__(self, config: LotfAdapterConfig):
        self._dt = float(config.dt)

        # determine scheme name from config
        setting = config.setting
        if setting == "innerloop":
            setting = "inner_loop"

        # load physical params
        import os
        import yaml

        lotf_obj_dir = os.path.join(LOTF_ROOT, "lotf", "objects")
        param_path = os.path.join(lotf_obj_dir, "quadrotor_files", "example_quad.yaml")
        with open(param_path) as f:
            param_dict = yaml.safe_load(f)
        params = QuadrotorParams(
            mass=param_dict["mass"],
            tbm_fr=tuple(param_dict["tbm_fr"]),
            tbm_bl=tuple(param_dict["tbm_bl"]),
            tbm_br=tuple(param_dict["tbm_br"]),
            tbm_fl=tuple(param_dict["tbm_fl"]),
            inertia=tuple(param_dict["inertia"]),
            motor_omega_min=param_dict.get("motor_omega_min", 150.0),
            motor_omega_max=param_dict.get("motor_omega_max", 2800.0),
            motor_tau=param_dict.get("motor_tau", 0.033),
            motor_inertia=param_dict.get("motor_inertia", 5.64e-6),
            omega_max=tuple(param_dict.get("omega_max", [10.0, 10.0, 4.0])),
            thrust_map=tuple(param_dict.get("thrust_map", [1.562522e-6, 0.0, 0.0])),
            kappa=param_dict.get("kappa", 0.022),
            thrust_min=param_dict.get("thrust_min", 0.0),
            thrust_max=param_dict.get("thrust_max", 8.5),
            rotors_config=param_dict.get("rotors_config", "cross"),
        )

        # build scheme config from setting name
        if setting == "simplest":
            scheme_cfg = SimplestConfig()
        elif setting == "resacc":
            scheme_cfg = ResAccConfig()
        elif setting == "approx":
            approx_path = config.approx_path or "simulation/inner_loop_approx.json"
            scheme_cfg = ApproxConfig(chirp_path=approx_path)
        elif setting == "approx_resacc":
            approx_path = config.approx_path or "simulation/inner_loop_approx.json"
            scheme_cfg = ApproxResAccConfig(chirp_path=approx_path)
        elif setting == "inner_loop":
            scheme_cfg = InnerLoopConfig()
        elif setting == "full":
            scheme_cfg = FullConfig()
        else:
            raise ValueError(f"Unknown setting: {setting}")

        res_params = _load_residual_params(config.setting, config.residual_checkpoint)
        scheme = build_scheme(scheme_cfg, params, res_params)
        self._quad = Quadrotor(scheme, params)

        # --- extract universal control model ---
        self._control_model = ControlModel(
            mass_kg=self._quad.mass,
            thrust_limits_N=(
                self._quad.thrust_min * 4,
                self._quad.thrust_max * 4,
            ),
            rate_limits_body_radps=np.array(self._quad.omega_max),
        )

        # --- cache thrust coefficient for extras population ---
        self._thrust_coeff = float(self._quad.params.thrust_map[0])

        self._current_quad_state: QuadrotorState | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def control_model(self) -> ControlModel:
        """Read-only universal plant limits."""
        return self._control_model

    def initialize(self, target: HoverTarget, seed: int = 0) -> StateSample:
        """Create the initial hover state and return it as a canonical sample."""
        yaw = float(target.yaw_rad)
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        self._current_quad_state = self._quad.create_state(
            p=jnp.array(target.p_world),
            R=jnp.array(R_yaw),
            v=jnp.zeros(3),
            omega=jnp.zeros(3),
            dr_key=jax.random.key(seed),
        )
        return _quad_state_to_sample(self._current_quad_state, self._thrust_coeff)

    def step(self, action: np.ndarray) -> StateSample:
        """Advance the plant by one timestep.

        Parameters
        ----------
        action : np.ndarray
            Shape ``(4,)``, canonical order ``[thrust, p, q, r]``.

        Returns
        -------
        StateSample
            Canonical kinematic snapshot after integration.
        """
        if self._current_quad_state is None:
            raise RuntimeError("Call initialize() before step()")

        f_d = float(action[0])
        omega_d = jnp.array(action[1:4].astype(np.float64))

        new_state = _jit_step(
            self._quad,
            self._current_quad_state,
            jnp.array(f_d),
            omega_d,
            self._dt,
        )
        self._current_quad_state = new_state
        return _quad_state_to_sample(new_state, self._thrust_coeff)
