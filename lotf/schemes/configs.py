"""Pydantic configuration models for quadrotor simulation schemes."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import jax.numpy as jnp
from pydantic import BaseModel, Field, model_validator

from lotf import LOTF_PATH


class QuadrotorParams(BaseModel):
    """Physical constants shared across schemes.

    Only ``mass`` is required (needed by the Quadrotor bridge for f_d→ap_z conversion).
    Other fields are consumed by InnerLoopScheme and FullScheme only.
    """

    mass: float
    tbm_fr: tuple[float, float, float] = (0.075, -0.10, 0.0)
    tbm_bl: tuple[float, float, float] = (-0.075, 0.10, 0.0)
    tbm_br: tuple[float, float, float] = (-0.075, -0.10, 0.0)
    tbm_fl: tuple[float, float, float] = (0.075, 0.10, 0.0)
    inertia: tuple[float, float, float] = (0.002410, 0.001800, 0.003759)
    motor_omega_min: float = 150.0
    motor_omega_max: float = 2800.0
    motor_tau: float = 0.033
    motor_inertia: float = 5.64e-6
    omega_max: tuple[float, float, float] = (10.0, 10.0, 4.0)
    thrust_map: tuple[float, float, float] = (1.562522e-6, 0.0, 0.0)
    kappa: float = 0.022
    thrust_min: float = 0.0
    thrust_max: float = 8.5
    rotors_config: Literal["cross"] = "cross"

    @property
    def nominal_motor_speed_given_hovering(self) -> float:
        """Theoretical motor speed required to hover [rad/s]."""
        return float(jnp.sqrt(self.mass * 9.81 / (4 * self.thrust_map[0])))


class SimplestConfig(BaseModel):
    """Simplest point-mass dynamics with ideal body-rate tracking."""

    name: Literal["simplest"] = "simplest"


class ResAccConfig(BaseModel):
    """Simplest dynamics with learned NN residual acceleration."""

    name: Literal["resacc"] = "resacc"


class ApproxConfig(BaseModel):
    """Chirp-fitted inner-loop approximation with delayed first-order filter."""

    name: Literal["approx"] = "approx"
    chirp_path: str

    @model_validator(mode="after")
    def _resolve_chirp_path(self) -> ApproxConfig:
        path = Path(self.chirp_path)
        if not path.is_absolute():
            path = Path(LOTF_PATH) / path
        if not path.exists():
            raise ValueError(f"Chirp approximation file not found: {path}")
        return self


class ApproxResAccConfig(BaseModel):
    """Chirp approximation with learned NN residual acceleration."""

    name: Literal["approx_resacc"] = "approx_resacc"
    chirp_path: str

    @model_validator(mode="after")
    def _resolve_chirp_path(self) -> ApproxResAccConfig:
        path = Path(self.chirp_path)
        if not path.is_absolute():
            path = Path(LOTF_PATH) / path
        if not path.exists():
            raise ValueError(f"Chirp approximation file not found: {path}")
        return self


class InnerLoopConfig(BaseModel):
    """Full Betaflight-style inner-loop dynamics with motor model."""

    name: Literal["inner_loop"] = "inner_loop"
    dt_low_level: float = 0.001
    rotor_augmentation_path: str | None = None
    omega_max: tuple[float, float, float] = (10.0, 10.0, 4.0)
    motor_tau: float = 0.033
    motor_inertia: float = 5.64e-6
    kp: tuple[float, float, float] = (40.0, 40.0, 30.0)
    kd: tuple[float, float, float] = (20.0, 20.0, 0.0)


class FullConfig(BaseModel):
    """Inner-loop dynamics with learned NN residual acceleration."""

    name: Literal["full"] = "full"
    dt_low_level: float = 0.001
    rotor_augmentation_path: str | None = None
    omega_max: tuple[float, float, float] = (10.0, 10.0, 4.0)
    motor_tau: float = 0.033
    motor_inertia: float = 5.64e-6
    kp: tuple[float, float, float] = (40.0, 40.0, 30.0)
    kd: tuple[float, float, float] = (20.0, 20.0, 0.0)


SchemeConfig = Annotated[
    SimplestConfig
    | ResAccConfig
    | ApproxConfig
    | ApproxResAccConfig
    | InnerLoopConfig
    | FullConfig,
    Field(discriminator="name"),
]
