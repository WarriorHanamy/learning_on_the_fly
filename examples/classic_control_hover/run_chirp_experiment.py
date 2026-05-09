#!/usr/bin/env python3
"""SE(3) hover + closed-loop chirp experiment entrypoint.

Usage::

    uv run python -m examples.classic_control_hover.run_chirp_experiment
    uv run python -m examples.classic_control_hover.run_chirp_experiment --setting full
    uv run python -m examples.classic_control_hover.run_chirp_experiment --output outputs/my_run

Generates:
  - ``log.npz``           full experiment trace
  - ``metadata.json``     experiment metadata + chirp segment definitions
  - ``segment_*.png``     per-channel command-vs-response figures
  - ``overview.png``      chirp schedule + error norms
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .chirp import chirp_vector, default_chirp_segments, segment_id
from .controller import se3_hover_controller
from .plotting import plot_all
from .recorder import append_log, init_log, save_log
from .schema import ExperimentConfig, HoverTarget, OUTPUT_DIR
from .sim_adapter import LotfAdapterConfig, SimAdapter


def _build_default_config() -> ExperimentConfig:
    """Return the recommended experiment configuration."""
    return ExperimentConfig(
        target=HoverTarget(p_world=(0.0, 0.0, 1.5), yaw_rad=0.0),
        chirp_segments=default_chirp_segments(),
        output_dir=OUTPUT_DIR,
    )


def run_experiment(cfg: ExperimentConfig, adapter_cfg: LotfAdapterConfig) -> int:
    """Execute the full closed-loop chirp experiment."""
    print("=" * 60)
    print("SE(3) Hover + Closed-Loop Chirp Experiment")
    print("=" * 60)
    print(f"  setting          : {adapter_cfg.setting}")
    print(f"  duration         : {adapter_cfg.duration:.1f} s")
    print(f"  dt               : {adapter_cfg.dt:.3f} s")
    print(f"  target position  : {cfg.target.p_world}")
    print(f"  target yaw       : {cfg.target.yaw_rad:.2f} rad")
    print(f"  output directory : {cfg.output_dir}")
    print(f"  chirp segments   : {len(cfg.chirp_segments)}")
    for s in cfg.chirp_segments:
        print(
            f"    [{s.channel}]  {s.f0_hz:.1f}→{s.f1_hz:.1f} Hz  "
            f"A={s.amplitude:.3f}  t=[{s.t_start:.0f},{s.t_start + s.duration:.0f}] s"
        )
    print("-" * 60)

    # --- build adaptor ---
    print("Building simulator adaptor ...")
    t0 = time.time()
    adaptor = SimAdapter(adapter_cfg)
    cm = adaptor.control_model
    print(f"  adaptor ready  ({time.time() - t0:.1f} s)")

    # --- initialise ---
    sample = adaptor.initialize(cfg.target)
    num_steps = adapter_cfg.num_steps()
    log = init_log(num_steps)

    print(f"  initial position : {sample.p_world_m}")
    print(f"  num steps        : {num_steps}")
    print("-" * 60)
    print("Running simulation loop ...")

    dt = adapter_cfg.dt
    t_loop = time.time()

    for i in range(num_steps):
        t = i * dt

        # 1) compute SE(3) hover control
        ctrl = se3_hover_controller(sample, cfg.target, cfg.gains, cm)

        # 2) compute chirp injection
        chirp_off = chirp_vector(t, cfg.chirp_segments)

        # 3) combine and saturate (use canonical ControlModel limits)
        action = np.zeros(4, dtype=np.float64)
        action[0] = np.clip(
            ctrl.f_cmd_N + chirp_off[0],
            cm.thrust_limits_N[0],
            cm.thrust_limits_N[1],
        )
        rate_max = cm.rate_limits_body_radps
        for j in range(3):
            action[j + 1] = np.clip(
                ctrl.omega_cmd_body_radps[j] + chirp_off[j + 1],
                -rate_max[j],
                rate_max[j],
            )

        # 4) step plant
        sample = adaptor.step(action)

        # 5) record
        seg_id = segment_id(t, cfg.chirp_segments)
        append_log(
            log,
            i,
            t=t,
            sample=sample,
            ctrl=ctrl,
            chirp_off=chirp_off,
            action_total=action,
            seg_id=seg_id,
        )

        # progress
        if (i + 1) % max(1, num_steps // 10) == 0:
            pct = (i + 1) / num_steps * 100
            elapsed = time.time() - t_loop
            eta = elapsed / (i + 1) * (num_steps - i - 1)
            print(f"  {pct:5.1f}%  |  t={t:6.1f}s  |  elapsed={elapsed:.1f}s  |  eta={eta:.1f}s")

    loop_time = time.time() - t_loop
    total_time = time.time() - t0
    print("-" * 60)
    print(f"  simulation loop  : {loop_time:.1f} s  ({num_steps / loop_time:.0f} steps/s)")
    print(f"  total wall time  : {total_time:.1f} s")

    # --- save ---
    print("\nSaving results ...")
    npz_path = save_log(log, cfg.output_dir, segments=cfg.chirp_segments)
    print(f"  log : {npz_path}")

    # --- plot ---
    print("Generating plots ...")
    plot_all(log, cfg.chirp_segments, cfg.output_dir)
    for seg in cfg.chirp_segments:
        name = "thrust" if seg.channel == "thrust" else seg.channel
        print(f"  {Path(cfg.output_dir) / f'segment_{name}.png'}")
    print(f"  {Path(cfg.output_dir) / 'overview.png'}")

    print("\nDone.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SE(3) hover + closed-loop chirp response experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python -m examples.classic_control_hover.run_chirp_experiment
    uv run python -m examples.classic_control_hover.run_chirp_experiment --setting full
    uv run python -m examples.classic_control_hover.run_chirp_experiment --output outputs/my_run
""",
    )
    parser.add_argument(
        "--setting",
        choices=["nominal", "resacc", "innerloop", "full", "approx", "approx_resacc"],
        default="full",
        help="Simulator forward-model setting (default: full)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for logs and plots",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=140.0,
        help="Experiment duration [s] (default: 140)",
    )
    parser.add_argument(
        "--residual-checkpoint",
        type=str,
        default=None,
        help="Override residual dynamics checkpoint path",
    )
    parser.add_argument(
        "--approx-path",
        type=str,
        default=None,
        help="Path to inner_loop_approx.json (required for --setting approx)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _build_default_config()
    cfg.output_dir = args.output
    adapter_cfg = LotfAdapterConfig(
        dt=0.02,
        duration=args.duration,
        setting=args.setting,
        residual_checkpoint=args.residual_checkpoint,
        approx_path=args.approx_path,
    )
    return run_experiment(cfg, adapter_cfg)


if __name__ == "__main__":
    sys.exit(main())
