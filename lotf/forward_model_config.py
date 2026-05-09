"""Shared forward model configuration for quadrotor simulation backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        enable_inner_loop_approx: Replace the full inner-loop dynamics with a
            per-channel delayed first-order filter fitted from chirp data.
            Parameters are loaded from ``inner_loop_approx_path``.
        inner_loop_approx_path: Path to inner_loop_approx.json produced by
            ``chirp_analysis_adapter.run_analysis()``.  Required when
            ``enable_inner_loop_approx`` is True.
    """

    enable_residual_acceleration: bool = False
    enable_inner_loop_dynamics: bool = False
    enable_inner_loop_approx: bool = False
    inner_loop_approx_path: str | None = None

    def to_dict(self) -> dict[str, bool | str | None]:
        return {
            "enable_residual_acceleration": self.enable_residual_acceleration,
            "enable_inner_loop_dynamics": self.enable_inner_loop_dynamics,
            "enable_inner_loop_approx": self.enable_inner_loop_approx,
            "inner_loop_approx_path": self.inner_loop_approx_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ForwardModelConfig:
        return cls(
            enable_residual_acceleration=d.get("enable_residual_acceleration", False),
            enable_inner_loop_dynamics=d.get("enable_inner_loop_dynamics", False),
            enable_inner_loop_approx=d.get("enable_inner_loop_approx", False),
            inner_loop_approx_path=d.get("inner_loop_approx_path", None),
        )


def coerce_forward_model_config(
    config: ForwardModelConfig | Mapping[str, Any] | None,
) -> ForwardModelConfig:
    """Normalize public forward-model configuration inputs to the schema type."""
    if config is None:
        return ForwardModelConfig()
    if isinstance(config, ForwardModelConfig):
        return config
    if isinstance(config, Mapping):
        return ForwardModelConfig.from_dict(dict(config))
    raise TypeError(
        "forward_model_config must be ForwardModelConfig, mapping, or None; "
        f"got {type(config).__name__}"
    )


SETTING_ORDER = ["nominal", "resacc", "innerloop", "full", "approx", "approx_resacc"]

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
    "approx": ForwardModelConfig(
        enable_residual_acceleration=False,
        enable_inner_loop_dynamics=False,
        enable_inner_loop_approx=True,
    ),
    "approx_resacc": ForwardModelConfig(
        enable_residual_acceleration=True,
        enable_inner_loop_dynamics=False,
        enable_inner_loop_approx=True,
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
