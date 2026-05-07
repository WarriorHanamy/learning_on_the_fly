"""Shared forward model configuration for quadrotor simulation backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForwardModelConfig:
    """Two-axis configuration for quadrotor forward simulation fidelity.

    Attributes:
        enable_residual_acceleration: Add learned residual acceleration to the
            translational dynamics.  Corresponds to the NN ensemble prediction
            stored in ``QuadrotorState.res_acc_mean``.
        enable_inner_loop_dynamics: Replace ideal body-rate commands with a
            full Betaflight-style low-level controller, motor dynamics, and
            RK4 integration of angular velocity.
    """

    enable_residual_acceleration: bool = False
    enable_inner_loop_dynamics: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "enable_residual_acceleration": self.enable_residual_acceleration,
            "enable_inner_loop_dynamics": self.enable_inner_loop_dynamics,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ForwardModelConfig:
        return cls(
            enable_residual_acceleration=d.get("enable_residual_acceleration", False),
            enable_inner_loop_dynamics=d.get("enable_inner_loop_dynamics", False),
        )


SETTING_ORDER = ["nominal", "resacc", "innerloop", "full"]

SETTING_SPECS: dict[str, ForwardModelConfig] = {
    "nominal": ForwardModelConfig(
        enable_residual_acceleration=False,
        enable_inner_loop_dynamics=False,
    ),
    "resacc": ForwardModelConfig(
        enable_residual_acceleration=True,
        enable_inner_loop_dynamics=False,
    ),
    "innerloop": ForwardModelConfig(
        enable_residual_acceleration=False,
        enable_inner_loop_dynamics=True,
    ),
    "full": ForwardModelConfig(
        enable_residual_acceleration=True,
        enable_inner_loop_dynamics=True,
    ),
}


def get_forward_model_config(setting_name: str) -> ForwardModelConfig:
    """Return the canonical forward model config for a standard setting."""
    try:
        return SETTING_SPECS[setting_name]
    except KeyError as e:
        allowed = ", ".join(SETTING_ORDER)
        raise ValueError(f"Unknown setting '{setting_name}'. Allowed settings: {allowed}") from e


def infer_setting_name(forward_model_config: ForwardModelConfig) -> str:
    """Infer the standard setting name for a forward model config."""
    for name in SETTING_ORDER:
        if SETTING_SPECS[name] == forward_model_config:
            return name
    raise ValueError(f"Forward model config is not a standard setting: {forward_model_config}")


def checkpoint_name_for_setting(base_name: str | Path, setting_name: str) -> Path:
    """Append the standard setting suffix to a checkpoint stem when needed."""
    get_forward_model_config(setting_name)
    path = Path(base_name)
    suffix = f"__{setting_name}"
    if path.name.endswith(suffix):
        return path
    return path.with_name(f"{path.name}{suffix}")
