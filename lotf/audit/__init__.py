"""Public audit API for the LOTF quadrotor forward model."""

from lotf.audit.approx_channel import run_approx_channel_audit
from lotf.audit.schema import (
    AUDIT_CHANNELS,
    AUDIT_OUTPUT_DIR,
    DEFAULT_APPROX_PATH,
    ApproxChannelArtifactPaths,
    ApproxChannelAuditConfig,
    ApproxChannelEnvironment,
    ApproxChannelExcitation,
    ApproxChannelOutputForm,
    QuadrotorStepAuditConfig,
    QuadrotorStepInput,
)

__all__ = [
    "AUDIT_CHANNELS",
    "AUDIT_OUTPUT_DIR",
    "DEFAULT_APPROX_PATH",
    "ApproxChannelArtifactPaths",
    "ApproxChannelAuditConfig",
    "ApproxChannelEnvironment",
    "ApproxChannelExcitation",
    "ApproxChannelOutputForm",
    "QuadrotorStepAuditor",
    "QuadrotorStepAuditConfig",
    "QuadrotorStepInput",
    "load_quadrotor_step_auditor",
    "run_approx_channel_audit",
]


def __getattr__(name: str):
    if name in {"QuadrotorStepAuditor", "load_quadrotor_step_auditor"}:
        from lotf.audit.quadrotor_step import QuadrotorStepAuditor, load_quadrotor_step_auditor

        values = {
            "QuadrotorStepAuditor": QuadrotorStepAuditor,
            "load_quadrotor_step_auditor": load_quadrotor_step_auditor,
        }
        return values[name]
    raise AttributeError(f"module 'lotf.audit' has no attribute {name!r}")
