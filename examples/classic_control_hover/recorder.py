"""Experiment recorder — owns the log schema, append logic, and persistence.

Reads and writes only local canonical types.  Backend-specific telemetry is
read from ``StateSample.extras`` and stored with an ``ext_`` prefix so
consumers can distinguish guaranteed fields from optional ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import N_ACTIONS, ChirpSegment, ControllerDiagnostics, StateSample


def init_log(num_steps: int) -> dict[str, np.ndarray]:
    """Allocate fixed-shape arrays for the experiment log.

    Parameters
    ----------
    num_steps : int
        Total number of simulation steps.

    Returns
    -------
    dict[str, np.ndarray]
        Pre-allocated log dictionary.  Keys without ``ext_`` prefix are
        guaranteed by the canonical contract.
    """
    n = num_steps
    return {
        # --- canonical fields ---
        "t_s": np.zeros(n, dtype=np.float64),
        "p_world_m": np.zeros((n, 3), dtype=np.float64),
        "v_world_mps": np.zeros((n, 3), dtype=np.float64),
        "q_world_from_body_wxyz": np.zeros((n, 4), dtype=np.float64),
        "omega_body_radps": np.zeros((n, 3), dtype=np.float64),
        "acc_world_mps2": np.zeros((n, 3), dtype=np.float64),
        "f_cmd_N": np.zeros(n, dtype=np.float64),
        "omega_cmd_body_radps": np.zeros((n, 3), dtype=np.float64),
        "chirp_offset": np.zeros((n, N_ACTIONS), dtype=np.float64),
        "action_total": np.zeros((n, N_ACTIONS), dtype=np.float64),
        "e_pos_world_m": np.zeros((n, 3), dtype=np.float64),
        "e_R_body": np.zeros((n, 3), dtype=np.float64),
        "segment_id": np.zeros(n, dtype=np.int32),
        # --- optional backend-specific extras ---
        "ext_R_world_from_body": np.zeros((n, 3, 3), dtype=np.float64),
        "ext_domega_body_radps2": np.zeros((n, 3), dtype=np.float64),
        "ext_motor_omega_radps": np.zeros((n, 4), dtype=np.float64),
        "ext_thrust_est_N": np.zeros(n, dtype=np.float64),
    }


def append_log(
    log: dict[str, np.ndarray],
    step_idx: int,
    *,
    t: float,
    sample: StateSample,
    ctrl: ControllerDiagnostics,
    chirp_off: np.ndarray,
    action_total: np.ndarray,
    seg_id: int,
) -> None:
    """Write one timestep into the pre-allocated log arrays.

    Parameters
    ----------
    log : dict
        Log dict created by ``init_log``.
    step_idx : int
        Current step index (0-based).
    t : float
        Simulation time [s].
    sample : StateSample
        Current canonical kinematic sample.
    ctrl : ControllerDiagnostics
        Controller output at this step.
    chirp_off : np.ndarray
        4-channel chirp offset vector.
    action_total : np.ndarray
        4-channel saturated total action vector.
    seg_id : int
        1-based active segment index (0 = no chirp active).
    """
    i = step_idx
    extras = sample.extras

    # canonical
    log["t_s"][i] = t
    log["p_world_m"][i] = sample.p_world_m
    log["v_world_mps"][i] = sample.v_world_mps
    log["q_world_from_body_wxyz"][i] = sample.q_world_from_body_wxyz
    log["omega_body_radps"][i] = sample.omega_body_radps
    log["acc_world_mps2"][i] = sample.acc_world_mps2
    log["f_cmd_N"][i] = ctrl.f_cmd_N
    log["omega_cmd_body_radps"][i] = ctrl.omega_cmd_body_radps
    log["chirp_offset"][i] = chirp_off
    log["action_total"][i] = action_total
    log["e_pos_world_m"][i] = ctrl.e_pos_world_m
    log["e_R_body"][i] = ctrl.e_R_body
    log["segment_id"][i] = seg_id

    # optional backend extras
    for key, log_key in [
        ("R_world_from_body", "ext_R_world_from_body"),
        ("domega_body_radps2", "ext_domega_body_radps2"),
        ("motor_omega_radps", "ext_motor_omega_radps"),
        ("thrust_est_N", "ext_thrust_est_N"),
    ]:
        if key in extras:
            log[log_key][i] = np.asarray(extras[key], dtype=log[log_key].dtype)


def save_log(
    log: dict[str, np.ndarray],
    output_dir: str | Path,
    *,
    segments: list[ChirpSegment] | None = None,
) -> Path:
    """Persist the log dictionary as ``.npz`` and companion ``.json``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    npz_path = out / "log.npz"
    np.savez_compressed(str(npz_path), **log)

    meta = {
        "num_steps": int(len(log["t_s"])),
        "duration_s": float(log["t_s"][-1]) if len(log["t_s"]) > 0 else 0.0,
        "fields": sorted(log.keys()),
    }
    if segments:
        meta["chirp_segments"] = [
            {
                "channel": s.channel,
                "amplitude": s.amplitude,
                "f0_hz": s.f0_hz,
                "f1_hz": s.f1_hz,
                "t_start": s.t_start,
                "duration": s.duration,
                "kind": s.kind,
            }
            for s in segments
        ]

    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return npz_path
