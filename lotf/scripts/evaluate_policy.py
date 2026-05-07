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
from lotf.eval.plotting import plot_benchmark_comparison
from lotf.eval.runner import (
    BenchmarkPolicySpec,
    BenchmarkRunResult,
    run_benchmark,
    run_benchmark_suite,
)
from lotf.forward_model_config import (
    SETTING_ORDER,
    ForwardModelConfig,
    checkpoint_name_for_setting,
    get_forward_model_config,
)
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
        default=None,
        help="Path to one trained policy checkpoint (single-model mode)",
    )

    parser.add_argument(
        "--checkpoint-stem",
        type=str,
        default="checkpoints/policy/traj_tracking_params",
        help="Base checkpoint stem for default four-setting compare mode",
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
        help="Save legacy single-checkpoint trajectory plot to benchmark_plot.png",
    )

    parser.add_argument(
        "--plot-output",
        type=str,
        default="benchmark_compare.png",
        help="Path to save suite comparison figure (default: benchmark_compare.png)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override benchmark seed (default: use seed from benchmark config)",
    )

    return parser.parse_args()


def _discover_latest_setting_checkpoints(checkpoint_stem: str) -> dict[str, Path]:
    stem = resolve_path(checkpoint_stem)
    stem = Path(stem)
    checkpoints = {}
    for setting_name in SETTING_ORDER:
        expected = checkpoint_name_for_setting(stem, setting_name)
        candidates = sorted(
            expected.parent.glob(f"{expected.name}*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            checkpoints[setting_name] = candidates[0]
    return checkpoints


def _print_missing_settings(checkpoint_stem: str, found: dict[str, Path]) -> None:
    stem = Path(resolve_path(checkpoint_stem))
    missing = [name for name in SETTING_ORDER if name not in found]
    print("Missing trained settings for benchmark:", file=sys.stderr)
    for setting_name in missing:
        expected = checkpoint_name_for_setting(stem, setting_name)
        print(
            f"- {setting_name:<10} -> expected pattern: {expected.parent / (expected.name + '*')}",
            file=sys.stderr,
        )

    print("Train them with:", file=sys.stderr)
    for setting_name in missing:
        print(f"  uv run train track --setting {setting_name}", file=sys.stderr)

    legacy = sorted(stem.parent.glob(f"{stem.name}*"))
    legacy = [p for p in legacy if not any(f"__{name}" in p.name for name in SETTING_ORDER)]
    if legacy:
        print(
            "Found legacy unsuffixed checkpoints, but their training setting cannot be inferred; "
            "please regenerate standard named checkpoints with the new train CLI.",
            file=sys.stderr,
        )


def _print_suite_table(results: list[BenchmarkRunResult]) -> None:
    print("\nBenchmark comparison:")
    header = (
        f"{'setting':<10} {'resacc':<6} {'innerloop':<9} {'return':>10} "
        f"{'collision':>10} {'ep_len':>8} {'pos_rmse':>9} {'vel_rmse':>9} checkpoint"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        cfg = result.train_forward_model_config
        m = result.metrics
        print(
            f"{result.label:<10} "
            f"{str(cfg.enable_residual_acceleration)[0]:<6} "
            f"{str(cfg.enable_inner_loop_dynamics)[0]:<9} "
            f"{m.mean_episodic_return:>10.2f} "
            f"{m.collision_rate:>10.2f} "
            f"{m.mean_episode_length:>8.1f} "
            f"{m.position_rmse:>9.2f} "
            f"{m.velocity_rmse:>9.2f} "
            f"{result.checkpoint_path}"
        )


def _suite_summary(results: list[BenchmarkRunResult], benchmark_cfg: _BenchmarkConfig, seed: int):
    return {
        "num_rollouts": benchmark_cfg.num_rollouts,
        "seed": seed,
        "ref_traj_name": benchmark_cfg.ref_traj_name,
        "results": [
            {
                "setting": r.label,
                "checkpoint": r.checkpoint_path,
                "forward_model_config": r.train_forward_model_config.to_dict(),
                "mean_episodic_return": r.metrics.mean_episodic_return,
                "collision_rate": r.metrics.collision_rate,
                "mean_episode_length": r.metrics.mean_episode_length,
                "position_rmse": r.metrics.position_rmse,
                "velocity_rmse": r.metrics.velocity_rmse,
            }
            for r in results
        ],
    }


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

    # policy network architecture comes from policy config
    try:
        policy_config = TrajTrackingConfig.from_yaml(args.policy_config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    discovered = None
    if not args.checkpoint:
        discovered = _discover_latest_setting_checkpoints(args.checkpoint_stem)
        if len(discovered) != len(SETTING_ORDER):
            _print_missing_settings(args.checkpoint_stem, discovered)
            return 1

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
    print(
        f"\nRunning benchmark: {benchmark_cfg.ref_traj_name} | "
        f"num_rollouts={benchmark_cfg.num_rollouts} | seed={seed}"
    )
    print("Forward model: residual_acceleration=true, inner_loop_dynamics=true")
    print("-" * 50)

    if args.checkpoint:
        print(f"Loading policy checkpoint: {args.checkpoint}")
        policy_config.forward_model_config = benchmark_forward_model
        policy_fn = load_policy_fn(args.checkpoint, policy_config, env)

        metrics, transitions = run_benchmark(
            env,
            policy_fn,
            residual_params,
            ref_traj=env.ref_traj,
            num_rollouts=benchmark_cfg.num_rollouts,
            seed=seed,
        )

        print("\n" + metrics.summary())

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

    policy_specs = []
    policy_config.forward_model_config = benchmark_forward_model
    for setting_name in SETTING_ORDER:
        checkpoint_path = discovered[setting_name]
        print(f"Loading {setting_name} policy checkpoint: {checkpoint_path}")
        policy_specs.append(
            BenchmarkPolicySpec(
                label=setting_name,
                checkpoint_path=str(checkpoint_path),
                train_forward_model_config=get_forward_model_config(setting_name),
                policy_fn=load_policy_fn(checkpoint_path, policy_config, env),
            )
        )

    results = run_benchmark_suite(
        env,
        policy_specs,
        residual_params,
        ref_traj=env.ref_traj,
        num_rollouts=benchmark_cfg.num_rollouts,
        seed=seed,
    )

    _print_suite_table(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(_suite_summary(results, benchmark_cfg, seed), f, indent=2)
        print(f"\nSummary saved to: {args.output}")

    if args.plot_output:
        plot_benchmark_comparison(results, env.ref_traj, args.plot_output)
        print(f"Comparison plot saved to: {args.plot_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
