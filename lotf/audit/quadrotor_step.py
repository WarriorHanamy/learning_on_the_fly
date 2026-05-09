"""Independent audit wrapper around ``Quadrotor.step``."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from orbax.checkpoint import PyTreeCheckpointer

from lotf import LOTF_ROOT, resolve_path
from lotf.audit.schema import QuadrotorStepAuditConfig, QuadrotorStepInput
from lotf.objects.quadrotor_obj import Quadrotor, QuadrotorState

_DUMMY_RESIDUAL_PATH = LOTF_ROOT / "checkpoints" / "residual_dynamics" / "dummy_params"


@partial(jax.jit, static_argnums=(0, 5))
def _jit_step(quad: Quadrotor, state, thrust_N, omega_body_radps, residual_params, dt):
    return quad.step(state, thrust_N, omega_body_radps, residual_params, dt)


def _load_residual_params(config: QuadrotorStepAuditConfig) -> FrozenDict:
    if config.residual_checkpoint is None:
        if not config.forward_model_config.enable_residual_acceleration:
            return FrozenDict({})
        path = _DUMMY_RESIDUAL_PATH
    else:
        path = resolve_path(config.residual_checkpoint)

    return PyTreeCheckpointer().restore(str(Path(path)))


class QuadrotorStepAuditor:
    """Small public facade that exposes only ``Quadrotor.step`` for audit use."""

    def __init__(self, config: QuadrotorStepAuditConfig):
        self.config = config
        self.quadrotor = Quadrotor.from_name(
            config.drone_name,
            config.forward_model_config,
        )
        self.residual_params = _load_residual_params(config)

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
                self.residual_params,
                self.config.dt,
            )
        return self.quadrotor.step(
            state,
            thrust_N,
            omega_body_radps,
            self.residual_params,
            self.config.dt,
        )


def load_quadrotor_step_auditor(
    config: QuadrotorStepAuditConfig | None = None,
) -> QuadrotorStepAuditor:
    """Load the core quadrotor compute object and expose its step function."""
    return QuadrotorStepAuditor(config or QuadrotorStepAuditConfig())
