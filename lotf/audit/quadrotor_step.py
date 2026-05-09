"""Independent audit wrapper around ``Quadrotor.step``."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

from lotf import LOTF_ROOT, resolve_path
from lotf.audit.schema import QuadrotorStepAuditConfig, QuadrotorStepInput
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

_DUMMY_RESIDUAL_PATH = LOTF_ROOT / "checkpoints" / "residual_dynamics" / "dummy_params"


@partial(jax.jit, static_argnums=(0, 4))
def _jit_step(quad: Quadrotor, state, thrust_N, omega_body_radps, dt):
    return quad.step(state, thrust_N, omega_body_radps, dt)


def _convert_fwd_to_scheme_config(fwd_config, approx_path: str | None = None):
    """Convert Legacy ForwardModelConfig to scheme config."""
    enable_res = fwd_config.enable_residual_acceleration
    enable_llc = fwd_config.enable_inner_loop_dynamics
    enable_approx = fwd_config.enable_inner_loop_approx

    if enable_llc:
        if enable_res:
            return FullConfig()
        return InnerLoopConfig()
    elif enable_approx:
        path = (
            approx_path or fwd_config.inner_loop_approx_path or "simulation/inner_loop_approx.json"
        )
        if enable_res:
            return ApproxResAccConfig(chirp_path=path)
        return ApproxConfig(chirp_path=path)
    else:
        if enable_res:
            return ResAccConfig()
        return SimplestConfig()


class QuadrotorStepAuditor:
    """Small public facade that exposes only ``Quadrotor.step`` for audit use."""

    def __init__(self, config: QuadrotorStepAuditConfig):
        self.config = config

        # load physical params
        import os

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

        scheme_cfg = _convert_fwd_to_scheme_config(config.forward_model_config)
        res_params = None
        if isinstance(scheme_cfg, (ResAccConfig, ApproxConfig, ApproxResAccConfig, FullConfig)):
            path = config.residual_checkpoint or _DUMMY_RESIDUAL_PATH
            from orbax.checkpoint import PyTreeCheckpointer

            res_params = PyTreeCheckpointer().restore(str(Path(resolve_path(path))))

        scheme = build_scheme(scheme_cfg, params, res_params)
        self.quadrotor = Quadrotor(scheme, params)

    def default_state(self) -> QuadrotorState:
        """Return the backend default quadrotor state."""
        return self.quadrotor.default_state()

    def step(
        self,
        state: QuadrotorState,
        command: QuadrotorStepInput,
    ) -> QuadrotorState:
        """Run exactly one ``Quadrotor.step`` call with schema-validated inputs."""
        thrust_N = jnp.asarray(command.thrust_N)
        omega_body_radps = jnp.asarray(command.omega_body_radps)
        if self.config.jit:
            return _jit_step(
                self.quadrotor,
                state,
                thrust_N,
                omega_body_radps,
                self.config.dt,
            )
        return self.quadrotor.step(
            state,
            thrust_N,
            omega_body_radps,
            self.config.dt,
        )


def load_quadrotor_step_auditor(
    config: QuadrotorStepAuditConfig | None = None,
) -> QuadrotorStepAuditor:
    """Load the core quadrotor compute object and expose its step function."""
    return QuadrotorStepAuditor(config or QuadrotorStepAuditConfig())
