"""Self-contained SE(3) hover controller.

Produces a total-thrust + body-rate command suitable for a plant interface
that accepts ``(thrust_N, p, q, r)``.  Operates exclusively on the canonical
``StateSample`` and ``ControlModel`` — no backend-specific knowledge.
"""

from __future__ import annotations

import numpy as np

from .schema import (
    ControlModel,
    ControllerDiagnostics,
    ControllerGains,
    GRAVITY_WORLD,
    HoverTarget,
    StateSample,
)

_GRAVITY = np.array(GRAVITY_WORLD, dtype=np.float64)  # (3,) world frame


# ---------------------------------------------------------------------------
# Lie-algebra helpers
# ---------------------------------------------------------------------------


def hat(v: np.ndarray) -> np.ndarray:
    """Maps a 3-vector to its skew-symmetric matrix (so(3) element).

    ``hat([x, y, z]) = [[0, -z, y], [z, 0, -x], [-y, x, 0]]``
    """
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def vee(S: np.ndarray) -> np.ndarray:
    """Maps a skew-symmetric 3x3 matrix back to a 3-vector.

    Inverse of ``hat``.
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float64)


def rot_error(R_des: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Attitude error vector (SO(3) logarithmic map approximation).

    ``e_R = 0.5 * vee(R_des^T @ R - R^T @ R_des)``

    The result lives in the body frame.
    """
    S = R_des.T @ R - R.T @ R_des
    return 0.5 * vee(S)


# ---------------------------------------------------------------------------
# quaternion <-> rotation matrix (canonical wxyz / Hamilton)
# ---------------------------------------------------------------------------


def quat_to_R_wxyz(q: np.ndarray) -> np.ndarray:
    """Hamilton quaternion [qw, qx, qy, qz] → rotation matrix body→world."""
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# SE(3) hover controller
# ---------------------------------------------------------------------------


def se3_hover_controller(
    sample: StateSample,
    target: HoverTarget,
    gains: ControllerGains,
    control_model: ControlModel,
) -> ControllerDiagnostics:
    """Compute thrust and body-rate commands to hover at a fixed point.

    Parameters
    ----------
    sample : StateSample
        Current canonical kinematic snapshot.
    target : HoverTarget
        Desired hover location and heading.
    gains : ControllerGains
        SE(3) controller gains.
    control_model : ControlModel
        Universal plant limits (mass, thrust bounds, rate bounds).

    Returns
    -------
    ControllerDiagnostics
        Thrust command, body-rate command, error signals, desired attitude.
    """
    # --- unpack canonical state ---
    p = sample.p_world_m.astype(np.float64)
    v = sample.v_world_mps.astype(np.float64)
    q_wxyz = sample.q_world_from_body_wxyz.astype(np.float64)
    omega = sample.omega_body_radps.astype(np.float64)

    R = quat_to_R_wxyz(q_wxyz)  # (3,3) body→world

    p_des = np.array(target.p_world, dtype=np.float64)
    yaw_des = float(target.yaw_rad)
    m = float(control_model.mass_kg)

    kp = np.array(gains.kp_pos, dtype=np.float64)
    kv = np.array(gains.kv_pos, dtype=np.float64)
    kR = np.array(gains.kR, dtype=np.float64)
    komega = np.array(gains.komega, dtype=np.float64)

    # --- position loop (world frame) ---
    e_p = p_des - p  # error = desired - current
    e_v = -v  # desired velocity = 0

    a_des = kp * e_p + kv * e_v - _GRAVITY
    F_des = m * a_des

    # --- attitude shaping ---
    f_norm = float(np.linalg.norm(F_des))
    if f_norm < 1e-6:
        cy, sy = np.cos(yaw_des), np.sin(yaw_des)
        R_des = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    else:
        b3_des = F_des / f_norm
        cy, sy = np.cos(yaw_des), np.sin(yaw_des)
        c1 = np.array([cy, sy, 0.0], dtype=np.float64)

        b2_des = np.cross(b3_des, c1)
        b2_norm = float(np.linalg.norm(b2_des))
        if b2_norm < 1e-6:
            b2_des = np.cross(b3_des, np.array([1.0, 0.0, 0.0]))
            b2_norm = float(np.linalg.norm(b2_des))
        b2_des = b2_des / b2_norm if b2_norm > 1e-12 else np.array([0.0, 1.0, 0.0])
        b1_des = np.cross(b2_des, b3_des)

        R_des = np.column_stack([b1_des, b2_des, b3_des])

    # --- attitude error ---
    e_R = rot_error(R_des, R)

    # --- body-rate command (body frame) ---
    omega_cmd = -kR * e_R - komega * omega

    # --- thrust command ---
    f_cmd = float(np.dot(F_des, R[:, 2]))

    # --- saturation ---
    f_min, f_max = control_model.thrust_limits_N
    f_cmd = float(np.clip(f_cmd, f_min, f_max))

    rate_max = control_model.rate_limits_body_radps.astype(np.float64)
    omega_cmd = np.clip(omega_cmd, -rate_max, rate_max)

    return ControllerDiagnostics(
        f_cmd_N=f_cmd,
        omega_cmd_body_radps=omega_cmd,
        e_pos_world_m=e_p,
        e_R_body=e_R,
        F_des_world_N=F_des,
        R_des_world_from_body=R_des,
    )
