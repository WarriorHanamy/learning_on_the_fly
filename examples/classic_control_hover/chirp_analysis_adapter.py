"""Chirp analysis adapter — bridges classic_control_hover log.npz → plant_analysis.

Extracts per-channel chirp sweeps from experiment log.npz, adapts them into
plant_analysis SweepResult objects, runs delayed-first-order system identification
for each channel, and saves the fitted parameters as inner_loop_approx.json.

Usage:
    from examples.classic_control_hover.chirp_analysis_adapter import run_analysis
    result = run_analysis("outputs/classic_control_hover/log.npz", "outputs/analysis")
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plant_analysis.core.bode import freqresp_hz
from plant_analysis.core.schemas import SweepResult, TimeSeries
from plant_analysis.core.sysid import fit_delayed_first_order, tfestimate
from plant_analysis.utils.plot_utils import bode_subplot_grid

from .schema import (
    InnerLoopApprox,
    InnerLoopApproxSet,
    P_IDX,
    Q_IDX,
    R_IDX,
    SweepExtract,
    THRUST_IDX,
)

_CHANNEL_IDX = {
    "thrust": THRUST_IDX,
    "p": P_IDX,
    "q": Q_IDX,
    "r": R_IDX,
}


def extract_sweeps(
    npz_path: str | Path,
    metadata_path: str | Path | None = None,
) -> list[SweepExtract]:
    """Extract per-channel chirp sweeps from experiment log.npz.

    Args:
        npz_path: Path to log.npz from run_chirp_experiment.py.
        metadata_path: Path to metadata.json.  If None, looks for
            metadata.json alongside npz_path.

    Returns:
        Four SweepExtract objects (thrust, p, q, r), one per chirp segment.
    """
    npz_path = Path(npz_path)
    if metadata_path is None:
        metadata_path = npz_path.with_name("metadata.json")
    else:
        metadata_path = Path(metadata_path)

    log = dict(np.load(str(npz_path), allow_pickle=False))
    with open(metadata_path) as f:
        meta = json.load(f)

    segments = meta.get("chirp_segments", [])
    if len(segments) != 4:
        raise ValueError(f"Expected 4 chirp segments in metadata, got {len(segments)}")

    t_s = log["t_s"]
    segment_id = log["segment_id"]
    action_total = log["action_total"]
    chirp_offset_log = log["chirp_offset"]

    omega_body = log.get("omega_body_radps")
    thrust_est = log.get("ext_thrust_est_N")

    dt = float(np.median(np.diff(t_s)))
    fs = 1.0 / dt

    extracts: list[SweepExtract] = []
    for i, seg in enumerate(segments, start=1):
        channel = seg["channel"]
        ch_idx = _CHANNEL_IDX[channel]

        mask = segment_id == i
        if not np.any(mask):
            print(
                f"  WARNING: segment {i} ({channel}) has no data; skipping",
                file=sys.stderr,
            )
            continue

        idx = np.flatnonzero(mask)
        start, stop = int(idx[0]), int(idx[-1]) + 1

        # extend window by 2×window_s on each side to capture Tukey taper
        window_s = float(seg.get("window_s", 2.0))
        pad_start = max(0, int(window_s / dt))
        pad_stop = min(len(t_s), stop + pad_start)
        start = max(0, start - pad_start)
        stop = pad_stop

        t_seg = t_s[start:stop].copy()
        t_seg -= t_seg[0]

        if channel == "thrust":
            if thrust_est is None:
                raise KeyError("ext_thrust_est_N missing from log.npz")
            output = thrust_est[start:stop].copy()
        else:
            if omega_body is None:
                raise KeyError("omega_body_radps missing from log.npz")
            # action index [1,2,3] → omega index [0,1,2]
            omega_idx = ch_idx - 1
            output = omega_body[start:stop, omega_idx].copy()

        extracts.append(
            SweepExtract(
                channel=channel,
                time_s=t_seg,
                chirp_injected=chirp_offset_log[start:stop, ch_idx].copy(),
                input_u=action_total[start:stop, ch_idx].copy(),
                output_y=output,
                fs_hz=fs,
                segment_meta=dict(seg),
            )
        )

    return extracts


def to_sweep_result(extract: SweepExtract) -> SweepResult:
    """Convert a SweepExtract into a plant_analysis SweepResult.

    All signals are de-meaned before packaging.  Since there is no separate
    external reference signal, ``reference`` is filled with zeros.
    """
    time = extract.time_s
    chirp_vals = extract.chirp_injected - np.mean(extract.chirp_injected)
    u_vals = extract.input_u - np.mean(extract.input_u)
    y_vals = extract.output_y - np.mean(extract.output_y)

    return SweepResult(
        chirp=TimeSeries(
            time_s=time,
            values=chirp_vals,
            signal_name=f"chirp_{extract.channel}",
        ),
        reference=TimeSeries(
            time_s=time,
            values=np.zeros_like(time),
            signal_name="reference",
        ),
        control=TimeSeries(
            time_s=time,
            values=u_vals,
            signal_name=f"control_{extract.channel}",
        ),
        output_filtered=TimeSeries(
            time_s=time,
            values=y_vals,
            signal_name=f"output_{extract.channel}",
        ),
        output_raw=TimeSeries(
            time_s=time,
            values=y_vals,
            signal_name=f"output_{extract.channel}",
        ),
        fs_hz=extract.fs_hz,
    )


def _save_bode_comparison(
    freq_frd: np.ndarray,
    H_frd: np.ndarray,
    coh: np.ndarray,
    dfo,  # DelayedFirstOrder
    channel: str,
    output_dir: Path,
) -> None:
    """Save a Bode comparison figure: FRD + coherence + fitted model."""
    omega_fit = (
        2.0
        * np.pi
        * np.logspace(
            np.log10(dfo.freq_min_hz * 0.5),
            np.log10(dfo.freq_max_hz * 2.0),
            400,
        )
    )
    H_fit = dfo.K * np.exp(-1j * omega_fit * dfo.delay) / (1j * omega_fit * dfo.tau + 1.0)
    mag_fit = 20.0 * np.log10(np.maximum(np.abs(H_fit), 1e-15))
    phase_fit = np.unwrap(np.angle(H_fit)) * 180.0 / np.pi

    mag_frd = 20.0 * np.log10(np.maximum(np.abs(H_frd), 1e-15))
    phase_frd = np.unwrap(np.angle(H_frd)) * 180.0 / np.pi

    fig, ax_mag, ax_phase = bode_subplot_grid()
    ax_mag.semilogx(freq_frd, mag_frd, ".", markersize=3, alpha=0.6, label="FRD")
    ax_mag.semilogx(omega_fit / (2.0 * np.pi), mag_fit, "-", linewidth=1.2, label="fitted DFO")
    ax_mag.axvline(dfo.freq_min_hz, color="gray", linestyle=":", linewidth=0.8)
    ax_mag.axvline(dfo.freq_max_hz, color="gray", linestyle=":", linewidth=0.8)
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.legend(fontsize=7)

    ax_phase.semilogx(freq_frd, phase_frd, ".", markersize=3, alpha=0.6, label="FRD")
    ax_phase.semilogx(omega_fit / (2.0 * np.pi), phase_fit, "-", linewidth=1.2, label="fitted DFO")
    ax_phase.axvline(dfo.freq_min_hz, color="gray", linestyle=":", linewidth=0.8)
    ax_phase.axvline(dfo.freq_max_hz, color="gray", linestyle=":", linewidth=0.8)
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.legend(fontsize=7)

    fig.suptitle(
        f"Channel: {channel}  |  K={dfo.K:.3f}  τ={dfo.tau:.4f}s  "
        f"delay={dfo.delay:.4f}s  |  mag RMSE={dfo.magnitude_rmse_db:.1f}dB  "
        f"phase RMSE={dfo.phase_rmse_deg:.1f}°"
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_dir / f"bode_{channel}.png"), dpi=200)
    plt.close(fig)


def run_analysis(
    npz_path: str | Path,
    output_dir: str | Path,
    setting: str = "full",
    freq_range: tuple[float, float] = (1.0, 4.0),
) -> InnerLoopApproxSet:
    """End-to-end analysis: extract → fit → save.

    Args:
        npz_path: Path to experiment log.npz.
        output_dir: Directory for inner_loop_approx.json and Bode PNGs.
        setting: Simulator setting name (e.g. "full", "innerloop").
        freq_range: Fitting frequency band [Hz].

    Returns:
        InnerLoopApproxSet with four fitted channels.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracts = extract_sweeps(npz_path)
    print(f"Extracted {len(extracts)} sweeps from {npz_path}")

    channels: list[InnerLoopApprox] = []
    for ext in extracts:
        print(
            f"  [{ext.channel}] n={len(ext.time_s)}  "
            f"t=[{ext.time_s[0]:.1f}, {ext.time_s[-1]:.1f}]s  fs={ext.fs_hz:.1f}Hz"
        )

        if ext.channel == "thrust":
            # thrust closed-loop response is non-minimum-phase (SE3 controller
            # creates phase lead); use unity gain with zero dynamics.
            dfo = None  # skip fitting
            approx = InnerLoopApprox(
                channel=ext.channel,
                K=1.0,
                tau=0.0,
                delay=0.0,
                freq_min_hz=freq_range[0],
                freq_max_hz=freq_range[1],
                magnitude_rmse_db=0.0,
                phase_rmse_deg=0.0,
            )
            print(f"    → K=1.000  τ=0.0000s  delay=0.0000s  (thrust fixed, not fitted)")
        else:
            dfo = fit_delayed_first_order(
                ext.input_u, ext.output_y, ext.fs_hz, freq_range=freq_range
            )
            approx = InnerLoopApprox(
                channel=ext.channel,
                K=dfo.K,
                tau=dfo.tau,
                delay=dfo.delay,
                freq_min_hz=dfo.freq_min_hz,
                freq_max_hz=dfo.freq_max_hz,
                magnitude_rmse_db=dfo.magnitude_rmse_db,
                phase_rmse_deg=dfo.phase_rmse_deg,
            )
            print(
                f"    → K={approx.K:.4f}  τ={approx.tau:.4f}s  delay={approx.delay:.4f}s  "
                f"mag_rmse={approx.magnitude_rmse_db:.2f}dB  phase_rmse={approx.phase_rmse_deg:.2f}°"
            )

        channels.append(approx)

        if dfo is not None:
            freq, H, coh = tfestimate(ext.input_u, ext.output_y, ext.fs_hz)
            _save_bode_comparison(freq, H, coh, dfo, ext.channel, output_dir)

    result = InnerLoopApproxSet(
        source_setting=setting,
        source_log_path=str(npz_path),
        channels=channels,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    json_path = output_dir / "inner_loop_approx.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "source_setting": result.source_setting,
                "source_log_path": result.source_log_path,
                "created_at": result.created_at,
                "channels": [
                    {
                        "channel": ch.channel,
                        "K": ch.K,
                        "tau": ch.tau,
                        "delay": ch.delay,
                        "freq_min_hz": ch.freq_min_hz,
                        "freq_max_hz": ch.freq_max_hz,
                        "magnitude_rmse_db": ch.magnitude_rmse_db,
                        "phase_rmse_deg": ch.phase_rmse_deg,
                    }
                    for ch in channels
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved {json_path}")

    return result
