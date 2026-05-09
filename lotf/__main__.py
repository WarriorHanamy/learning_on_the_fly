#!/usr/bin/env python3
"""Unified entry point for LOTF training commands.

This module provides a single CLI interface that dispatches to specific
training tasks based on command arguments.

Usage:
    uv run train --help
    uv run train --version
    uv run train --list-configs
    uv run train track --config configs/traj_tracking.yaml
    uv run train residual --config configs/residual_dynamics.yaml --dataset data.csv

Subcommands:
    track     Train trajectory tracking policy
    residual  Train residual dynamics ensemble model
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from typing import Sequence

from lotf import LOTF_ROOT
from lotf.forward_model_config import SETTING_ORDER


def get_version() -> str:
    """Get the current package version.

    Returns:
        Version string from package metadata.
    """
    try:
        return importlib.metadata.version("lotf")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0 (dev)"


def list_configs() -> int:
    """List available YAML configuration files.

    Prints all .yaml files found in the configs/ directory.

    Returns:
        Exit code (0 for success).
    """
    configs_dir = LOTF_ROOT / "configs"

    if not configs_dir.exists():
        print("Error: configs/ directory not found", file=sys.stderr)
        return 1

    yaml_files = sorted(configs_dir.glob("*.yaml"))

    if not yaml_files:
        print("No configuration files found in configs/")
        return 0

    print("Available configuration files:")
    for yaml_file in yaml_files:
        rel_path = yaml_file.relative_to(LOTF_ROOT)
        print(f"  {rel_path}")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="train",
        description="LOTF: Learning Agile Flight with Differentiable Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train trajectory tracking (all schemes)
    uv run train

    # Train a single scheme
    uv run train --setting simplest
    uv run train --checkpoint checkpoints/my_policy

    # Train residual dynamics
    uv run train residual --dataset data.csv

    # Show version
    uv run train --version
        """,
    )

    # Global flags
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show package version and exit",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configuration files and exit",
    )

    # Track training args (top-level)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/traj_tracking.yaml",
        help="Path to YAML configuration file (default: configs/traj_tracking.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/policy/traj_tracking_params",
        help="Path to save the trained policy checkpoint or base stem for --setting all",
    )
    parser.add_argument(
        "--setting",
        choices=["all", *SETTING_ORDER, "innerloop"],
        default="all",
        help="Scheme to train (default: all)",
    )
    parser.add_argument(
        "--residual-checkpoint",
        type=str,
        default="checkpoints/residual_dynamics/residual_params",
        help="Residual dynamics checkpoint for resacc/full settings",
    )
    parser.add_argument(
        "--trajectory-output",
        type=str,
        default=None,
        help="Path to export trajectory CSV file (optional)",
    )
    parser.add_argument(
        "--approx-path",
        type=str,
        default=None,
        help="Path to inner_loop_approx.json (required for --setting approx)",
    )

    # Residual subcommand
    subparsers = parser.add_subparsers(
        dest="command",
        title="subcommands",
        metavar="COMMAND",
    )
    residual_parser = subparsers.add_parser(
        "residual",
        help="Train residual dynamics ensemble model",
        description="Train an ensemble of residual dynamics neural networks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run train residual
    uv run train residual --dataset my_data.csv
        """,
    )
    residual_parser.add_argument(
        "--config",
        type=str,
        default="configs/residual_dynamics.yaml",
        help="Path to YAML configuration file (default: configs/residual_dynamics.yaml)",
    )
    residual_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to CSV dataset file",
    )
    residual_parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/residual_dynamics/dummy_params",
        help="Path to save trained model checkpoint",
    )

    return parser


def _run_with_argv(train_main, new_argv: list[str]) -> int:
    """Run a training main function with modified sys.argv."""
    original_argv = sys.argv
    try:
        sys.argv = new_argv
        return train_main()
    finally:
        sys.argv = original_argv


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the LOTF CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"lotf {get_version()}")
        return 0
    if args.list_configs:
        return list_configs()

    if args.command == "residual":
        from lotf.scripts.train_residual import main as train_residual

        train_argv = [
            "train_residual",
            "--config",
            args.config,
            "--dataset",
            args.dataset,
            "--output",
            args.output,
        ]
        return _run_with_argv(train_residual, train_argv)

    # Default: track training
    from lotf.scripts.train_traj_tracking import main as train_track

    train_argv = [
        "train_traj_tracking",
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--setting",
        str(args.setting),
        "--residual-checkpoint",
        args.residual_checkpoint,
    ]
    if args.trajectory_output:
        train_argv += ["--trajectory-output", args.trajectory_output]
    if args.approx_path:
        train_argv += ["--approx-path", args.approx_path]
    return _run_with_argv(train_track, train_argv)

    # Dispatch to subcommand handlers
    if args.command == "track":
        from lotf.scripts.train_traj_tracking import main as train_track

        train_argv = [
            "train_traj_tracking",
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
            "--setting",
            str(args.setting),
            "--residual-checkpoint",
            args.residual_checkpoint,
        ]
        if args.trajectory_output:
            train_argv += ["--trajectory-output", args.trajectory_output]
        if args.approx_path:
            train_argv += ["--approx-path", args.approx_path]
        return _run_with_argv(train_track, train_argv)

    elif args.command == "residual":
        from lotf.scripts.train_residual import main as train_residual

        train_argv = [
            "train_residual",
            "--config",
            args.config,
            "--dataset",
            args.dataset,
            "--output",
            args.output,
        ]
        return _run_with_argv(train_residual, train_argv)

    elif args.command == "eval":
        from lotf.scripts.evaluate_policy import main as eval_policy

        eval_argv = [
            "evaluate_policy",
            "--benchmark-config",
            args.benchmark_config,
            "--setting",
            args.setting,
        ]
        if args.checkpoint:
            eval_argv += ["--checkpoint", args.checkpoint]
        else:
            eval_argv += ["--checkpoint-stem", args.checkpoint_stem]
        if args.residual_checkpoint:
            eval_argv += ["--residual-checkpoint", args.residual_checkpoint]
        if args.output:
            eval_argv += ["--output", args.output]
        if args.plot_output:
            eval_argv += ["--plot-output", args.plot_output]
        if args.seed is not None:
            eval_argv += ["--seed", str(args.seed)]
        return _run_with_argv(eval_policy, eval_argv)

    else:
        print(f"Error: Unknown command '{args.command}'", file=sys.stderr)
        return 2


def main_play(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `play` command — 3D policy visualization.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    from lotf.scripts.visualize_policy import main as play_policy

    play_argv = ["play"] + (list(argv) if argv else sys.argv[1:])
    return _run_with_argv(play_policy, play_argv)


def main_eval(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `eval` command — trajectory tracking benchmark.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    from lotf.scripts.evaluate_policy import main as eval_policy

    eval_argv = ["eval"] + (list(argv) if argv else sys.argv[1:])
    return _run_with_argv(eval_policy, eval_argv)


if __name__ == "__main__":
    sys.exit(main())
