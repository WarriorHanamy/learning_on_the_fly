"""Experiment-level plotting — reads only the local log schema.

Generates per-segment figures (command vs response) and an overview figure.
Uses the canonical ``q_world_from_body_wxyz`` for attitude reporting;
optional backend extras are used when available.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .schema import P_IDX, Q_IDX, R_IDX, THRUST_IDX
from .schema import ChirpSegment


def _quat_to_R_wxyz(q: np.ndarray) -> np.ndarray:
    """Hamilton quaternion [qw,qx,qy,qz] array (N,4) → rotation matrices (N,3,3)."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[np.newaxis, :]
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[:, 0, 1] = 2 * (qx * qy - qw * qz)
    R[:, 0, 2] = 2 * (qx * qz + qw * qy)
    R[:, 1, 0] = 2 * (qx * qy + qw * qz)
    R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[:, 1, 2] = 2 * (qy * qz - qw * qx)
    R[:, 2, 0] = 2 * (qx * qz - qw * qy)
    R[:, 2, 1] = 2 * (qy * qz + qw * qx)
    R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
    return R.squeeze()


def _euler_xyz_from_R(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrices (N,3,3) to XYZ Euler angles [deg]."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 2:
        R = R[np.newaxis, ...]
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    pitch = -np.arcsin(np.clip(R[:, 2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.rad2deg(np.column_stack([roll, pitch, yaw]))


def _get_R(log: dict) -> np.ndarray:
    """Retrieve rotation matrices, preferring canonical quaternion conversion."""
    if "q_world_from_body_wxyz" in log:
        return _quat_to_R_wxyz(log["q_world_from_body_wxyz"])
    if "ext_R_world_from_body" in log:
        return log["ext_R_world_from_body"]
    raise KeyError("No rotation data found in log (missing q_world_from_body_wxyz)")


def _highlight_segment(ax, seg: ChirpSegment):
    """Draw a shaded region for the active chirp segment."""
    ax.axvspan(seg.t_start, seg.t_start + seg.duration, alpha=0.08, color="orange")


def _save_close(fig: plt.Figure, path: Path):
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# segment-specific figures
# ---------------------------------------------------------------------------


def plot_thrust_segment(
    log: dict[str, np.ndarray],
    seg: ChirpSegment,
    output_dir: str | Path,
) -> None:
    """Thrust-loop figure: f_cmd vs thrust_est, z / vz / az."""
    t = log["t_s"]
    f_cmd = log["f_cmd_N"]
    p = log["p_world_m"]
    v = log["v_world_mps"]
    acc = log["acc_world_mps2"]
    chirp = log["chirp_offset"]

    thrust_est = log.get("ext_thrust_est_N", None)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), squeeze=False)
    ax0, ax1, ax2 = axes[0, 0], axes[1, 0], axes[2, 0]

    ax0.plot(t, f_cmd, linewidth=1.0, label="f_cmd [N]", color="#e45756")
    if thrust_est is not None:
        ax0.plot(t, thrust_est, linewidth=1.0, label="thrust_est [N]", color="#4c78a8")
    ax0.set_ylabel("Thrust [N]")
    ax0.legend(loc="upper right", fontsize="small")
    ax0.grid(True, alpha=0.25)
    _highlight_segment(ax0, seg)

    ax1.plot(t, p[:, 2], linewidth=1.0, label="z [m]", color="#54a24b")
    ax1.set_ylabel("z [m]")
    ax1.legend(loc="upper right", fontsize="small")
    ax1.grid(True, alpha=0.25)
    ax1b = ax1.twinx()
    ax1b.plot(
        t, chirp[:, THRUST_IDX], linewidth=0.6, label="chirp thrust", color="orange", alpha=0.7
    )
    ax1b.set_ylabel("chirp offset [N]", color="orange")
    ax1b.tick_params(axis="y", labelcolor="orange")
    _highlight_segment(ax1, seg)

    ax2.plot(t, v[:, 2], linewidth=1.0, label="vz [m/s]", color="#f58518")
    ax2.plot(t, acc[:, 2], linewidth=1.0, label="az [m/s^2]", color="#b279a2")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("[m/s] / [m/s^2]")
    ax2.legend(loc="upper right", fontsize="small")
    ax2.grid(True, alpha=0.25)
    _highlight_segment(ax2, seg)

    fig.suptitle(
        f"Thrust loop  |  chirp {seg.f0_hz:.1f}→{seg.f1_hz:.1f} Hz  |  A={seg.amplitude:.3f} N"
    )
    _save_close(fig, Path(output_dir) / "segment_thrust.png")


def plot_rate_segment(
    log: dict[str, np.ndarray],
    seg: ChirpSegment,
    output_dir: str | Path,
) -> None:
    """Angular-loop figure for a single body-rate channel (p / q / r)."""
    t = log["t_s"]
    omega_cmd = log["omega_cmd_body_radps"]
    omega = log["omega_body_radps"]
    chirp = log["chirp_offset"]
    R = _get_R(log)
    euler = _euler_xyz_from_R(R)

    axis_map = {"p": (0, 0, "roll"), "q": (1, 1, "pitch"), "r": (2, 2, "yaw")}
    ch_idx, action_idx, euler_label = axis_map[seg.channel]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), squeeze=False)
    ax0, ax1 = axes[0, 0], axes[1, 0]

    ax0.plot(
        t, omega_cmd[:, ch_idx], linewidth=1.0, label=f"{seg.channel}_cmd [rad/s]", color="#e45756"
    )
    ax0.plot(
        t, omega[:, ch_idx], linewidth=1.0, label=f"{seg.channel} actual [rad/s]", color="#4c78a8"
    )
    ax0.set_ylabel("Body rate [rad/s]")
    ax0.legend(loc="upper right", fontsize="small")
    ax0.grid(True, alpha=0.25)
    ax0b = ax0.twinx()
    ax0b.plot(t, chirp[:, action_idx], linewidth=0.6, label="chirp", color="orange", alpha=0.7)
    ax0b.set_ylabel("chirp offset", color="orange")
    ax0b.tick_params(axis="y", labelcolor="orange")
    _highlight_segment(ax0, seg)

    ax1.plot(t, euler[:, ch_idx], linewidth=1.0, label=euler_label, color="#54a24b")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(f"{euler_label} [deg]")
    ax1.legend(loc="upper right", fontsize="small")
    ax1.grid(True, alpha=0.25)
    _highlight_segment(ax1, seg)

    fig.suptitle(
        f"{seg.channel}-rate loop  |  chirp {seg.f0_hz:.1f}→{seg.f1_hz:.1f} Hz  |  "
        f"A={seg.amplitude:.3f} rad/s"
    )
    _save_close(fig, Path(output_dir) / f"segment_{seg.channel}.png")


# ---------------------------------------------------------------------------
# overview figure
# ---------------------------------------------------------------------------


def plot_overview(log: dict[str, np.ndarray], output_dir: str | Path) -> None:
    """Single overview figure: chirp offsets + position/attitude error norms."""
    t = log["t_s"]
    chirp = log["chirp_offset"]
    e_pos = log["e_pos_world_m"]
    e_R = log["e_R_body"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), squeeze=False)
    ax0, ax1 = axes[0, 0], axes[1, 0]

    labels = ["thrust", "p", "q", "r"]
    colors = ["#e45756", "#4c78a8", "#54a24b", "#f58518"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax0.plot(t, chirp[:, i], linewidth=0.8, label=label, color=color)
    ax0.set_ylabel("Chirp offset")
    ax0.legend(loc="upper right", fontsize="small", ncol=4)
    ax0.grid(True, alpha=0.25)

    ax1.plot(
        t, np.linalg.norm(e_pos, axis=1), linewidth=1.0, label="||e_pos|| [m]", color="#4c78a8"
    )
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Position error norm [m]", color="#4c78a8")
    ax1.tick_params(axis="y", labelcolor="#4c78a8")
    ax1b = ax1.twinx()
    ax1b.plot(t, np.linalg.norm(e_R, axis=1), linewidth=1.0, label="||e_R|| [-]", color="#f58518")
    ax1b.set_ylabel("Attitude error norm [-]", color="#f58518")
    ax1b.tick_params(axis="y", labelcolor="#f58518")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize="small")
    ax1.grid(True, alpha=0.25)

    fig.suptitle("Experiment overview")
    _save_close(fig, Path(output_dir) / "overview.png")


def plot_all(
    log: dict[str, np.ndarray],
    segments: list[ChirpSegment],
    output_dir: str | Path,
) -> None:
    """Generate all per-segment figures plus the overview."""
    for seg in segments:
        if seg.channel == "thrust":
            plot_thrust_segment(log, seg, output_dir)
        else:
            plot_rate_segment(log, seg, output_dir)
    plot_overview(log, output_dir)
