#!/usr/bin/env python3
"""Trajectory tracking policy training script.

This script trains a neural network policy for quadrotor trajectory tracking using
backpropagation through time (BPTT).

Usage:
    uv run python -m lotf.scripts.train_traj_tracking \\
        --config configs/traj_tracking.yaml
    uv run python -m lotf.scripts.train_traj_tracking \\
        --config configs/traj_tracking.yaml --checkpoint checkpoints/policy/my_policy

CLI Arguments:
    --config: Path to YAML config (default: configs/traj_tracking.yaml)
    --checkpoint: Path to save the trained policy checkpoint
    --trajectory-output: Path to export trajectory CSV file (optional)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import jax
from orbax.checkpoint import PyTreeCheckpointer

from lotf import LOTF_ROOT, resolve_path
from lotf.algos import bptt
from lotf.envs import TrajTrackingStateEnv
from lotf.traj_tracking_setup import (
    TrajTrackingConfig,
    build_policy_train_state,
    build_traj_tracking_env,
)


def load_dummy_residual_params() -> Any:
    """Load dummy residual dynamics parameters.

    For base policy training, we do not use residual acceleration for forward
    simulation or backpropagation.  However, the environment requires
    residual parameters to be passed, so we load a dummy checkpoint.
    """
    path = LOTF_ROOT / "checkpoints" / "residual_dynamics" / "dummy_params"
    ckptr = PyTreeCheckpointer()
    return ckptr.restore(path)


def get_unique_checkpoint_path(base_path: Path) -> Path:
    """Generate a unique checkpoint path by appending timestamp if directory exists."""
    path = resolve_path(base_path)
    if not path.exists():
        return path

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = path.parent / f"{path.name}_{timestamp}"

    print(f"Checkpoint directory exists, using: {new_path}")
    return new_path


def save_checkpoint(output_path: str, params: Any) -> None:
    """Save policy parameters to a unique checkpoint path."""
    path = get_unique_checkpoint_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckptr = PyTreeCheckpointer()
    ckptr.save(str(path), params)
    print(f"Policy saved successfully to: {path}")


def export_trajectory(traj: Any, output_path: str) -> None:
    """Export trajectory to CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    TrajTrackingStateEnv.generate_csv(traj, output_path)
    print(f"Trajectory exported successfully to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train trajectory tracking policy using BPTT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python -m lotf.scripts.train_traj_tracking
    uv run python -m lotf.scripts.train_traj_tracking --config my_config.yaml
    uv run python -m lotf.scripts.train_traj_tracking --checkpoint checkpoints/policy/my_policy
    uv run python -m lotf.scripts.train_traj_tracking --trajectory-output outputs/traj.csv
        """,
    )

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
        help="Path to save the trained policy checkpoint",
    )

    parser.add_argument(
        "--trajectory-output",
        type=str,
        default=None,
        help="Path to export trajectory CSV file (optional)",
    )

    return parser.parse_args()


def main() -> int:
    """Main training function.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_args()

    print(f"Loading configuration from: {args.config}")
    try:
        config = TrajTrackingConfig.from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error parsing config: {e}", file=sys.stderr)
        return 1

    print(f"Initializing with seed: {config.seed}")
    key = jax.random.key(config.seed)
    key_init, key_bptt = jax.random.split(key, 2)

    print("Creating environment...")
    env = build_traj_tracking_env(config)

    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    print("Environment info:")
    print(f"  action_dim: {action_dim}")
    print(f"  obs_dim: {obs_dim}")
    print(f"  ref_traj_name: {config.ref_traj_name}")
    print(f"  max_steps_in_episode: {env.max_steps_in_episode}")

    print("Creating policy network...")
    train_state = build_policy_train_state(config, env, key_init)

    print("Loading dummy residual dynamics parameters...")
    dummy_residual_params = load_dummy_residual_params()

    print(f"Initializing {config.num_envs} parallel environments...")
    key_bptt, key_ = jax.random.split(key_bptt)
    key_reset = jax.random.split(key_, config.num_envs)
    init_env_state, init_obs = env.reset(key_reset, None)

    print(f"\nStarting training for {config.max_epochs} epochs...")
    print("-" * 50)

    time_start = time.time()
    res_dict = bptt.train(
        env,
        init_env_state,
        init_obs,
        train_state,
        num_epochs=config.max_epochs,
        num_steps_per_epoch=env.max_steps_in_episode,
        num_envs=config.num_envs,
        res_model_params=dummy_residual_params,
        key=key_bptt,
    )
    time_train_compile = time.time() - time_start

    print("-" * 50)
    print(f"Compile + Training time: {time_train_compile:.2f}s")

    losses = res_dict["metrics"]
    returns = -losses
    final_reward = returns[-1]
    print(f"Final reward: {final_reward:.2f}")

    trained_policy_params = res_dict["runner_state"].train_state.params
    save_checkpoint(args.checkpoint, trained_policy_params)

    if args.trajectory_output:
        print(f"\nExporting trajectory to: {args.trajectory_output}")
        print("Note: Trajectory export requires running a separate rollout.")
        print("Use lotf.envs.rollout() with the trained policy to generate trajectory data.")

    print("\nTraining complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
