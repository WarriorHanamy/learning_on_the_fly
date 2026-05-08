"""Canonical conventions for the classic_control_hover experiment package.

This is the single source of truth for signs, axes, units, command ordering,
and data contracts within the experiment layer.  Backend adaptors translate
between backend-native representations and these canonical conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ============================================================================
# 1. World frame
# ============================================================================
# +z up, gravity points downward.
# x and y axes form a right-handed system with z.
# At yaw = 0, body +x (forward) aligns with world +x.

GRAVITY_WORLD = (0.0, 0.0, -9.81)  # m/s^2 in world frame

# ============================================================================
# 2. Body frame  —  FLU (Forward-Left-Up)
# ============================================================================
#   +x body  =  Forward
#   +y body  =  Left
#   +z body  =  Up  (= thrust direction)
#
# Body rates (p, q, r) follow the right-hand rule about each body axis:
#   p > 0  ->  rotation about +x  ->  right roll
#   q > 0  ->  rotation about +y  ->  pitch up
#   r > 0  ->  rotation about +z  ->  yaw (clockwise viewed from above)

# ============================================================================
# 3. Rotation representation
# ============================================================================
# Hamilton quaternion, scalar-first:  [qw, qx, qy, qz]
#   This is the only native rotation representation in the canonical layer.
#   Controller / plotting / recorder must convert to R internally if needed.
#   Backend adaptors are responsible for converting backend-native R or
#   Euler angles to/from this quaternion convention.

# ============================================================================
# 4. Error convention
# ============================================================================
# All errors follow  error = desired - current:
#   e_pos = p_des - p             (world frame)
#   e_vel = v_des - v             (world frame)
#   e_R   = 0.5 vee(R_des^T R - R^T R_des)   (body frame, SO(3) log approx)

# ============================================================================
# 5. Command / action order
# ============================================================================
ACTION_ORDER = ("thrust", "p", "q", "r")
THRUST_IDX = 0  # total thrust [N]
P_IDX = 1  # roll  body-rate [rad/s]
Q_IDX = 2  # pitch body-rate [rad/s]
R_IDX = 3  # yaw   body-rate [rad/s]
N_ACTIONS = 4

BODY_RATE_ORDER = ("p", "q", "r")  # [rad/s] in body frame

# ============================================================================
# 6. Units
# ============================================================================
# position   : m     (world frame)
# velocity   : m/s   (world frame)
# accel      : m/s^2 (world frame)
# thrust     : N     (total, body-z direction)
# body rates : rad/s (body frame)
# time       : s

# ============================================================================
# 7. I/O
# ============================================================================
OUTPUT_DIR = "outputs/classic_control_hover"

# ============================================================================
# 8. Data contracts  —  backend-agnostic dataclasses
# ============================================================================


@dataclass
class HoverTarget:
    """Desired hover operating point."""

    p_world: tuple[float, float, float] = (0.0, 0.0, 1.5)  # [m]
    yaw_rad: float = 0.0


@dataclass
class ControllerGains:
    """SE(3) hover controller gains."""

    kp_pos: tuple[float, float, float] = (4.0, 4.0, 6.0)  # position gain  [1/s^2]
    kv_pos: tuple[float, float, float] = (2.0, 2.0, 4.0)  # velocity gain  [1/s]
    kR: tuple[float, float, float] = (8.0, 8.0, 3.0)  # attitude gain  [rad/s]
    komega: tuple[float, float, float] = (0.5, 0.5, 0.2)  # rate damping   [-]


@dataclass
class ChirpSegment:
    """One artificial-injection segment for closed-loop excitation."""

    channel: str  # "thrust" | "p" | "q" | "r"
    amplitude: float  # excitation amplitude in native unit
    f0_hz: float  # start frequency [Hz]
    f1_hz: float  # end frequency [Hz]
    t_start: float  # segment start time [s]
    duration: float  # segment length [s]
    kind: str = "log"  # "log" (log-sweep) or "linear"
    window_s: float = 2.0  # Tukey window half-width at each end [s]


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration (does NOT contain backend details)."""

    target: HoverTarget = field(default_factory=HoverTarget)
    gains: ControllerGains = field(default_factory=ControllerGains)
    chirp_segments: list[ChirpSegment] = field(default_factory=list)
    output_dir: str = OUTPUT_DIR


@dataclass
class StateSample:
    """Canonical kinematic snapshot delivered by a backend adaptor.

    Every backend adaptor promises to provide these fields in the canonical
    convention (FLU body frame, +z-up world frame, Hamilton quaternion).

    ``extras`` carries optional backend-specific telemetry (raw rotation
    matrix, motor speeds, angular acceleration, estimated thrust, …).
    Code that reads ``extras`` keys should degrade gracefully when a key
    is absent.
    """

    p_world_m: np.ndarray  # (3,)  position
    v_world_mps: np.ndarray  # (3,)  linear velocity
    q_world_from_body_wxyz: np.ndarray  # (4,)  Hamilton quaternion [qw,qx,qy,qz]
    omega_body_radps: np.ndarray  # (3,)  angular velocity [p, q, r]
    acc_world_mps2: np.ndarray  # (3,)  linear acceleration
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerDiagnostics:
    """Diagnostic snapshot of one SE(3) hover controller evaluation.

    Contains the command sent to the plant plus internal error and reference
    signals used for logging and visualisation.
    """

    f_cmd_N: float  # total thrust command [N]
    omega_cmd_body_radps: np.ndarray  # (3,) [p, q, r] body-rate command
    e_pos_world_m: np.ndarray  # (3,) position error
    e_R_body: np.ndarray  # (3,) attitude error in body frame
    F_des_world_N: np.ndarray  # (3,) desired force vector (world)
    R_des_world_from_body: np.ndarray  # (3,3) desired attitude


@dataclass
class ControlModel:
    """Universal plant limits required by the controller layer.

    Provided by the adaptor at initialisation time.  Actuator-specific
    parameters (thrust coefficient, motor count, …) are kept inside the
    adaptor and exposed only as optional extras.
    """

    mass_kg: float
    thrust_limits_N: tuple[float, float]  # (min, max) total thrust
    rate_limits_body_radps: np.ndarray  # (3,) [p_max, q_max, r_max]
