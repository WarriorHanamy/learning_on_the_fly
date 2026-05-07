#!/usr/bin/env python3
"""Evaluation script for trajectory tracking policy benchmarks.

Evaluates a trained policy against a fixed simulator benchmark
(residual acceleration + inner-loop dynamics, FIG8 reference trajectory).

Usage:
    uv run python -m lotf.scripts.evaluate_policy track \\
        --checkpoint checkpoints/policy/traj_tracking_params
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lotf import LOTF_ROOT, resolve_path
from lotf.eval.runner import run_benchmark
from lotf.forward_model_config import ForwardModelConfig
from lotf.traj_tracking_setup import (
    PolicyNetConfig,
    TrajTrackingConfig,
    build_traj_tracking_env,
    load_policy_fn,
    load_residual_params,
)

# ---------------------------------------------------------------------------
# Benchmark config (a thin dataclass for the benchmark YAML)
# ---------------------------------------------------------------------------


@dataclass
class _BenchmarkConfig:
    ref_traj_name: str = "fig8"
    sim_dt: float = 0.02
    delay: float = 0.04
    max_sim_time: float = 10.0
    skip_start: bool = True
    yaw_scale: float = 0.0
    pitch_roll_scale: float = 0.0
    position_std: float = 0.0
    velocity_std: float = 0.0
    omega_std: float = 0.0
    policy_net_hidden_layers: list[int] = field(default_factory=lambda: [512, 512])
    policy_net_initial_scale: float = 0.01
    num_rollouts: int = 20
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> _BenchmarkConfig:
        path = resolve_path(path)
        if not Path(path).exists():
            raise FileNotFoundError(f"Benchmark config not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        env = raw.get("env", {})
        pnet = raw.get("policy_net", {})
        bench = raw.get("benchmark", {})

        return cls(
            ref_traj_name=raw.get("ref_traj_name", "fig8"),
            sim_dt=env.get("sim_dt", 0.02),
            delay=env.get("delay", 0.04),
            max_sim_time=env.get("max_sim_time", 10.0),
            skip_start=env.get("skip_start", True),
            yaw_scale=env.get("yaw_scale", 0.0),
            pitch_roll_scale=env.get("pitch_roll_scale", 0.0),
            position_std=env.get("position_std", 0.0),
            velocity_std=env.get("velocity_std", 0.0),
            omega_std=env.get("omega_std", 0.0),
            policy_net_hidden_layers=pnet.get("hidden_layers", [512, 512]),
            policy_net_initial_scale=pnet.get("initial_scale", 0.01),
            num_rollouts=bench.get("num_rollouts", 20),
            seed=bench.get("seed", 0),
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval",
        description="Evaluate a trained trajectory tracking policy against a fixed benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run eval track --checkpoint checkpoints/policy/traj_tracking_params
    uv run eval track --checkpoint checkpoints/policy/traj_tracking_params --output results.json
    uv run eval track --checkpoint checkpoints/policy/traj_tracking_params --plot
        """,
    )

    parser.add_argument(
        "env_type",
        choices=["track"],
        help="Policy type (only track is supported)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained policy checkpoint",
    )

    parser.add_argument(
        "--benchmark-config",
        type=str,
        default="configs/benchmark_traj_fig8.yaml",
        help="Path to the benchmark YAML configuration",
    )

    parser.add_argument(
        "--policy-config",
        type=str,
        default="configs/traj_tracking.yaml",
        help="Path to the training YAML config (used for policy network architecture)",
    )

    parser.add_argument(
        "--residual-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to residual dynamics checkpoint "
            "(default: checkpoints/residual_dynamics/residual_params)"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save benchmark summary as JSON (optional)",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Save trajectory plot to file (optional)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override benchmark seed (default: use seed from benchmark config)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # --- load benchmark config ---
    try:
        benchmark_cfg = _BenchmarkConfig.from_yaml(args.benchmark_config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    seed = args.seed if args.seed is not None else benchmark_cfg.seed

    # --- build benchmark simulator (fixed forward model) ---
    benchmark_forward_model = ForwardModelConfig(
        enable_residual_acceleration=True,
        enable_inner_loop_dynamics=True,
    )

    # construct a TrajTrackingConfig for the benchmark env
    benchmark_env_config = TrajTrackingConfig(
        sim_dt=benchmark_cfg.sim_dt,
        max_sim_time=benchmark_cfg.max_sim_time,
        delay=benchmark_cfg.delay,
        ref_traj_name=benchmark_cfg.ref_traj_name,
        skip_start=benchmark_cfg.skip_start,
        forward_model_config=benchmark_forward_model,
        yaw_scale=benchmark_cfg.yaw_scale,
        pitch_roll_scale=benchmark_cfg.pitch_roll_scale,
        position_std=benchmark_cfg.position_std,
        velocity_std=benchmark_cfg.velocity_std,
        omega_std=benchmark_cfg.omega_std,
        policy_net=PolicyNetConfig(
            hidden_layers=benchmark_cfg.policy_net_hidden_layers,
            initial_scale=benchmark_cfg.policy_net_initial_scale,
        ),
    )

    env = build_traj_tracking_env(
        benchmark_env_config, with_log_wrapper=False, with_vec_wrapper=False
    )

    # --- load policy checkpoint ---
    print(f"Loading policy checkpoint: {args.checkpoint}")

    # policy network architecture comes from policy config
    try:
        policy_config = TrajTrackingConfig.from_yaml(args.policy_config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    policy_config.forward_model_config = benchmark_forward_model
    policy_fn = load_policy_fn(args.checkpoint, policy_config, env)

    # --- load residual checkpoint ---
    residual_path = args.residual_checkpoint or str(
        LOTF_ROOT / "checkpoints" / "residual_dynamics" / "residual_params"
    )
    print(f"Loading residual checkpoint: {residual_path}")
    try:
        residual_params = load_residual_params(residual_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "Hint: train a residual model first (uv run train residual) "
            "or pass --residual-checkpoint explicitly.",
            file=sys.stderr,
        )
        return 1

    # --- run benchmark ---
    print(f"\nRunning benchmark: FIG8 | num_rollouts={benchmark_cfg.num_rollouts} | seed={seed}")
    print("Forward model: residual_acceleration=true, inner_loop_dynamics=true")
    print("-" * 50)

    metrics, transitions = run_benchmark(
        env,
        policy_fn,
        residual_params,
        ref_traj=env.ref_traj,
        num_rollouts=benchmark_cfg.num_rollouts,
        seed=seed,
    )

    print("\n" + metrics.summary())

    # --- optional outputs ---
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "mean_episodic_return": metrics.mean_episodic_return,
            "collision_rate": metrics.collision_rate,
            "mean_episode_length": metrics.mean_episode_length,
            "position_rmse": metrics.position_rmse,
            "velocity_rmse": metrics.velocity_rmse,
            "num_rollouts": benchmark_cfg.num_rollouts,
            "seed": seed,
            "checkpoint": args.checkpoint,
            "ref_traj_name": benchmark_cfg.ref_traj_name,
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {args.output}")

    if args.plot:
        env.plot_trajectories(transitions, save_path="benchmark_plot.png")
        print("Plot saved to: benchmark_plot.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
