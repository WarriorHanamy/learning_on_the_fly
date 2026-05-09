from __future__ import annotations

from typing import NamedTuple
import yaml

import jax.numpy as jnp


class AugmentationParams(NamedTuple):
    """
    Parameters for calculating aerodynamic and motor-related residuals.

    Attributes:
        fx, fy, fz: polynomial coefficients for force residuals
        tx, ty, tz: polynomial coefficients for torque residuals
        scale_force: global scaling factor for force residuals
        scale_torque: global scaling factor for torque residuals
    """

    fx: jnp.ndarray
    fy: jnp.ndarray
    fz: jnp.ndarray
    tx: jnp.ndarray
    ty: jnp.ndarray
    tz: jnp.ndarray
    scale_force: float
    scale_torque: float

    @classmethod
    def from_yaml(cls, path: str) -> AugmentationParams:
        """Loads augmentation parameters from a yaml file"""
        with open(path) as stream:
            try:
                config = yaml.safe_load(stream)
                return cls.from_dict(config)
            except yaml.YAMLError as exc:
                raise exc

    @classmethod
    def from_dict(cls, config: dict) -> AugmentationParams:
        """Parses configuration dictionary into specific coefficient arrays"""
        # coefficient extraction for x-axis force
        fx = jnp.array(
            [
                config["fx"].get("vx", 0.0),
                config["fx"].get("m_mean", 0.0),
                config["fx"].get("ang_y", 0.0),
                config["fx"].get("vx*m_mean", 0.0),
                config["fx"].get("vx_sqr", 0.0),
                config["fx"].get("vx_cub", 0.0),
            ]
        )
        # coefficient extraction for y-axis force
        fy = jnp.array(
            [
                config["fy"].get("vy", 0.0),
                config["fy"].get("m_mean", 0.0),
                config["fy"].get("ang_x", 0.0),
                config["fy"].get("vy*m_mean", 0.0),
                config["fy"].get("vy_sqr", 0.0),
                config["fy"].get("vy_cub", 0.0),
            ]
        )
        # coefficient extraction for z-axis force
        fz = jnp.array(
            [
                config["fz"].get("offset", 0.0),
                config["fz"].get("vhor", 0.0),
                config["fz"].get("vz", 0.0),
                config["fz"].get("m_mean", 0.0),
                config["fz"].get("vhor*vz", 0.0),
                config["fz"].get("vhor*m_mean", 0.0),
                config["fz"].get("vz*m_mean", 0.0),
                config["fz"].get("vhor_sqr", 0.0),
                config["fz"].get("vhor*vz*m_mean", 0.0),
                config["fz"].get("vz_cub", 0.0),
            ]
        )
        # coefficient extraction for x-axis torque
        tx = jnp.array(
            [
                config["tx"].get("vy", 0.0),
                config["tx"].get("m_mean", 0.0),
                config["tx"].get("vy*m_mean", 0.0),
            ]
        )
        # coefficient extraction for y-axis torque
        ty = jnp.array(
            [
                config["ty"].get("vx", 0.0),
                config["ty"].get("m_mean", 0.0),
                config["ty"].get("vx*m_mean", 0.0),
            ]
        )
        # coefficient extraction for z-axis torque
        tz = jnp.array(
            [
                config["tz"].get("vx", 0.0),
                config["tz"].get("vy", 0.0),
            ]
        )

        scale_force = config["scale_force"]
        scale_torque = config["scale_torque"]
        return cls(fx, fy, fz, tx, ty, tz, scale_force, scale_torque)


def compute_residuals(state, params: AugmentationParams):
    """
    Computes force and torque residuals using a polynomial model.

    Args:
        state: current system state containing velocity, orientation, and motors.
            Supports both old (direct fields) and new (scheme_state dict) formats.
        params: AugmentationParams containing model coefficients
    Returns:
        tuple of (residual_acceleration_world, residual_torque_body)
    """
    v_world = state.v
    v_body = state.R.T @ v_world
    vx, vy, vz = v_body
    vhor = jnp.sqrt(vx**2 + vy**2)

    # access motor_omega from either old (direct) or new (scheme_state dict) format
    motor_omega = _get_motor_omega(state)
    ang_x = 0.0
    ang_y = 0.0

    m_mean = (motor_omega**2).mean()

    # build polynomial features for force
    poly_fx = jnp.array([vx, m_mean, ang_y, vx * m_mean, vx * jnp.abs(vx), vx**3])
    poly_fy = jnp.array([vy, m_mean, ang_x, vy * m_mean, vy * jnp.abs(vy), vy**3])
    poly_fz = jnp.array(
        [
            1,
            vhor,
            vz,
            m_mean,
            vhor * vz,
            vhor * m_mean,
            vz * m_mean,
            vhor**2,
            vhor * vz * m_mean,
            vz**3,
        ]
    )

    # build polynomial features for torque
    poly_tx = jnp.array([vy, m_mean, vy * m_mean])
    poly_ty = jnp.array([vx, m_mean, vx * m_mean])
    poly_tz = jnp.array([vx, vy])

    # compute dot products to get force increments
    delta_fx = jnp.dot(poly_fx, params.fx)
    delta_fy = jnp.dot(poly_fy, params.fy)
    delta_fz = jnp.dot(poly_fz, params.fz)

    residual_acceleration = params.scale_force * jnp.array([delta_fx, delta_fy, delta_fz])
    # transform residual acceleration from body to world frame
    residual_acceleration = state.R @ residual_acceleration

    # compute dot products to get torque increments
    delta_tx = jnp.dot(poly_tx, params.tx)
    delta_ty = jnp.dot(poly_ty, params.ty)
    delta_tz = jnp.dot(poly_tz, params.tz)

    residual_torque = params.scale_torque * jnp.array([delta_tx, delta_ty, delta_tz])

    return residual_acceleration, residual_torque


def _get_motor_omega(state) -> jnp.ndarray:
    """Extract motor_omega from old (direct field) or new (scheme_state dict) state.

    Supports backward compatibility during the scheme-based refactoring transition.
    """
    if hasattr(state, "motor_omega"):
        return state.motor_omega
    ss = getattr(state, "scheme_state", {})
    return ss.get("inner_loop_motor_omega", ss.get("motor_omega", jnp.zeros(4)))
