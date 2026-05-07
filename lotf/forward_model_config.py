"""Shared forward model configuration for quadrotor simulation backend."""

from __future__ import annotations

from dataclasses import dataclass


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
