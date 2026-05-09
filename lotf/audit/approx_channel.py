"""Channel-by-channel audit for fitted inner-loop approximation models."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lotf import resolve_path
from lotf.audit.schema import ApproxChannelAuditConfig


def _artifact_path(config: ApproxChannelAuditConfig, name: str) -> Path:
    output_dir = resolve_path(config.artifacts.output_dir)
    return output_dir / name


def _load_channels(path: str) -> dict[str, dict]:
    resolved = resolve_path(path)
    with open(resolved) as f:
        data = json.load(f)
    return {ch["channel"]: ch for ch in data["channels"]}


def _chirp_phase(t: np.ndarray, f0_hz: float, f1_hz: float, duration_s: float, kind: str):
    if kind == "linear_chirp":
        k = (f1_hz - f0_hz) / duration_s
        return 2.0 * np.pi * (f0_hz * t + 0.5 * k * t**2)
    ratio = f1_hz / f0_hz
    return 2.0 * np.pi * f0_hz * duration_s / np.log(ratio) * (ratio ** (t / duration_s) - 1.0)


def _taper(t: np.ndarray, duration_s: float, window_s: float) -> np.ndarray:
    if window_s <= 0.0:
        return np.ones_like(t)
    window_s = min(window_s, duration_s / 2.0)
    w = np.ones_like(t)
    start = t < window_s
    end = t > duration_s - window_s
    w[start] = 0.5 * (1.0 - np.cos(np.pi * t[start] / window_s))
    w[end] = 0.5 * (1.0 - np.cos(np.pi * (duration_s - t[end]) / window_s))
    return w


def _excitation(config: ApproxChannelAuditConfig, t: np.ndarray) -> np.ndarray:
    exc = config.excitation
    if exc.kind == "step":
        return exc.amplitude * (t >= exc.step_time_s)
    if exc.kind == "sine":
        return exc.amplitude * np.sin(2.0 * np.pi * exc.f0_hz * t)

    phase = _chirp_phase(t, exc.f0_hz, exc.f1_hz, config.environment.duration_s, exc.kind)
    return exc.amplitude * np.sin(phase) * _taper(t, config.environment.duration_s, exc.window_s)


def _simulate_delayed_first_order(u: np.ndarray, dt: float, K: float, tau: float, delay: float):
    delay_steps = max(int(np.ceil(delay / dt)), 0)
    u_delayed = np.zeros_like(u)
    if delay_steps == 0:
        u_delayed[:] = u
    elif delay_steps < len(u):
        u_delayed[delay_steps:] = u[:-delay_steps]

    if tau <= 1e-9:
        return K * u_delayed

    alpha = np.exp(-dt / tau)
    y = np.zeros_like(u)
    for i in range(1, len(u)):
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * K * u_delayed[i]
    return y


def run_approx_channel_audit(config: ApproxChannelAuditConfig) -> dict[str, Path]:
    """Run the preliminary channel-by-channel approximation audit."""
    output_dir = resolve_path(config.artifacts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    channels = _load_channels(config.environment.approx_path)
    dt = config.environment.dt
    t = np.arange(0.0, config.environment.duration_s, dt)
    u = _excitation(config, t)

    responses: dict[str, np.ndarray] = {}
    summary_channels = []
    for channel in config.excitation.channels:
        params = channels[channel]
        y = _simulate_delayed_first_order(u, dt, params["K"], params["tau"], params["delay"])
        responses[channel] = y
        summary_channels.append(
            {
                "channel": channel,
                "K": params["K"],
                "tau": params["tau"],
                "delay": params["delay"],
                "peak_input": float(np.max(np.abs(u))),
                "peak_output": float(np.max(np.abs(y))),
            }
        )

    written: dict[str, Path] = {}
    if config.output.save_timeseries:
        path = _artifact_path(config, config.artifacts.timeseries_npz)
        np.savez(path, time_s=t, input_u=u, **{f"response_{k}": v for k, v in responses.items()})
        written["timeseries"] = path

    if config.output.save_summary:
        path = _artifact_path(config, config.artifacts.summary_json)
        with open(path, "w") as f:
            json.dump({"channels": summary_channels}, f, indent=2)
        written["summary"] = path

    config_path = _artifact_path(config, config.artifacts.config_json)
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    written["config"] = config_path

    if config.output.save_figure:
        fig, axes = plt.subplots(len(responses), 1, figsize=(10, 2.4 * len(responses)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (channel, y) in zip(axes, responses.items()):
            ax.plot(t, u, label="input", linewidth=1.0)
            ax.plot(t, y, label="approx output", linewidth=1.0)
            ax.set_ylabel(channel)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("time [s]")
        fig.tight_layout()
        path = _artifact_path(config, config.artifacts.figure)
        fig.savefig(path, dpi=config.output.dpi, format=config.output.figure_format)
        if config.output.show:
            plt.show()
        plt.close(fig)
        written["figure"] = path

    return written
