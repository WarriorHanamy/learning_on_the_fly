"""Factory for constructing Scheme instances from pydantic configs."""

from __future__ import annotations

from flax.core import FrozenDict

from lotf.schemes.base import Scheme
from lotf.schemes.configs import (
    QuadrotorParams,
    SimplestConfig,
    ResAccConfig,
    ApproxConfig,
    ApproxResAccConfig,
    InnerLoopConfig,
    FullConfig,
)
from lotf.schemes.simplest import SimplestScheme
from lotf.schemes.resacc import ResAccScheme
from lotf.schemes.approx import ApproxScheme
from lotf.schemes.approx_resacc import ApproxResAccScheme
from lotf.schemes.inner_loop import InnerLoopScheme
from lotf.schemes.full import FullScheme


def build_scheme(
    config: SimplestConfig
    | ResAccConfig
    | ApproxConfig
    | ApproxResAccConfig
    | InnerLoopConfig
    | FullConfig,
    params: QuadrotorParams | None = None,
    res_model_params: FrozenDict | None = None,
) -> Scheme:
    """Build a Scheme instance from its pydantic config.

    Parameters
    ----------
    config : Scheme config dataclass
        Discriminated union of scheme-specific configs.
    params : QuadrotorParams or None
        Physical constants. Required only for InnerLoop/Full schemes.
    res_model_params : FrozenDict or None
        NN residual model parameters. Required for ResAcc/Approx/Full schemes.

    Returns
    -------
    Scheme
        Ready-to-use integration scheme.

    Raises
    ------
    ValueError
        If required params are missing for the selected scheme.
    """
    if isinstance(config, SimplestConfig):
        return SimplestScheme()

    if isinstance(config, ResAccConfig):
        if res_model_params is None:
            raise ValueError("res_model_params is required for ResAccConfig")
        return ResAccScheme(res_model_params)

    if isinstance(config, ApproxConfig):
        if res_model_params is None:
            raise ValueError("res_model_params is required for ApproxConfig")
        return ApproxScheme(config.chirp_path, res_model_params)

    if isinstance(config, ApproxResAccConfig):
        if res_model_params is None:
            raise ValueError("res_model_params is required for ApproxResAccConfig")
        return ApproxResAccScheme(config.chirp_path, res_model_params)

    if isinstance(config, InnerLoopConfig):
        if params is None:
            raise ValueError("params is required for InnerLoopConfig")
        return InnerLoopScheme(
            params=params,
            dt_low_level=config.dt_low_level,
            rotor_augmentation_path=config.rotor_augmentation_path,
            omega_max=config.omega_max,
            motor_tau=config.motor_tau,
            motor_inertia=config.motor_inertia,
            kp=config.kp,
            kd=config.kd,
        )

    if isinstance(config, FullConfig):
        if params is None:
            raise ValueError("params is required for FullConfig")
        if res_model_params is None:
            raise ValueError("res_model_params is required for FullConfig")
        return FullScheme(
            params=params,
            res_model_params=res_model_params,
            dt_low_level=config.dt_low_level,
            rotor_augmentation_path=config.rotor_augmentation_path,
            omega_max=config.omega_max,
            motor_tau=config.motor_tau,
            motor_inertia=config.motor_inertia,
            kp=config.kp,
            kd=config.kd,
        )

    raise ValueError(f"Unknown scheme config type: {type(config)}")
