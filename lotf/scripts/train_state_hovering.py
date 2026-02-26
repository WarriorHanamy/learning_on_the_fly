#!/usr/bin/env python3
"""State-based hovering policy training script.

This script trains a neural network policy for quadrotor hovering using
backpropagation through time (BPTT). The implementation is modularized from
the original notebook examples/state_hovering/1_train_base_policy.ipynb.

Usage:
    uv run python -m lotf.scripts.train_state_hovering --config configs/state_hovering.yaml
    uv run python -m lotf.scripts.train_state_hovering --config configs/state_hovering.yaml --output checkpoints/policy/my_policy

CLI Arguments:
    --config: Path to YAML configuration file (default: configs/state_hovering.yaml)
    --output: Path to save the trained policy checkpoint (default: checkpoints/policy/state_hovering_params)

Examples:
    # Train with default configuration
    uv run python -m lotf.scripts.train_state_hovering

    # Train with custom config and output location
    uv run python -m lotf.scripts.train_state_hovering \\
        --config configs/state_hovering.yaml \\
        --output checkpoints/policy/custom_hovering

    # Quick training run with fewer epochs (override via config)
    uv run python -m lotf.scripts.train_state_hovering --config configs/state_hovering.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import optax
import yaml
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer

from lotf import LOTF_PATH
from lotf.algos import bptt
from lotf.envs import HoveringStateEnv
from lotf.envs.wrappers import LogWrapper, MinMaxObservationWrapper, VecEnv
from lotf.modules import MLP
from lotf.objects import Quadrotor


@dataclass
class SimDynConfig:
    """Simulation dynamics configuration."""

    use_high_fidelity: bool = False
    use_forward_residual: bool = False


@dataclass
class PolicyNetConfig:
    """Policy network architecture configuration."""

    hidden_layers: list[int] = field(default_factory=lambda: [512, 512])
    initial_scale: float = 0.01


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    initial_lr: float = 0.005
    scheduler: str = "cosine_decay"


@dataclass
class StateHoveringConfig:
    """Complete configuration for state hovering training.

    Attributes:
        seed: Random seed for reproducibility
        num_envs: Number of parallel environments
        max_epochs: Maximum number of training epochs
        sim_dt: Simulation time step in seconds
        max_sim_time: Maximum simulation time per episode in seconds
        delay: Action delay in seconds
        reward_sharpness: Sharpness parameter for reward function
        action_penalty_weight: Weight for action penalty in reward
        hover_target: Target hovering position [x, y, z]
        sim_dyn_config: Simulation dynamics configuration
        yaw_scale: Scale for yaw randomization
        pitch_roll_scale: Scale for pitch/roll randomization
        velocity_std: Standard deviation for velocity randomization
        omega_std: Standard deviation for angular velocity randomization
        margin: Margin for initial position randomization
        policy_net: Policy network configuration
        optimizer: Optimizer configuration
    """

    seed: int = 0
    num_envs: int = 200
    max_epochs: int = 200
    sim_dt: float = 0.02
    max_sim_time: float = 3.0
    delay: float = 0.04
    reward_sharpness: float = 3.0
    action_penalty_weight: float = 0.5
    hover_target: list[float] = field(default_factory=lambda: [1.5, 0.0, 1.5])
    sim_dyn_config: SimDynConfig = field(default_factory=SimDynConfig)
    yaw_scale: float = 1.0
    pitch_roll_scale: float = 0.1
    velocity_std: float = 0.1
    omega_std: float = 0.1
    margin: float = 0.5
    policy_net: PolicyNetConfig = field(default_factory=PolicyNetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StateHoveringConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            StateHoveringConfig instance with loaded values.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If YAML parsing fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:
            raise ValueError(f"Configuration file is empty: {path}")

        # Parse nested configs
        sim_dyn_dict = raw_config.get("sim_dyn_config", {})
        sim_dyn_config = SimDynConfig(
            use_high_fidelity=sim_dyn_dict.get("use_high_fidelity", False),
            use_forward_residual=sim_dyn_dict.get("use_forward_residual", False),
        )

        policy_net_dict = raw_config.get("policy_net", {})
        policy_net_config = PolicyNetConfig(
            hidden_layers=policy_net_dict.get("hidden_layers", [512, 512]),
            initial_scale=policy_net_dict.get("initial_scale", 0.01),
        )

        optimizer_dict = raw_config.get("optimizer", {})
        optimizer_config = OptimizerConfig(
            initial_lr=optimizer_dict.get("initial_lr", 0.005),
            scheduler=optimizer_dict.get("scheduler", "cosine_decay"),
        )

        return cls(
            seed=raw_config.get("seed", 0),
            num_envs=raw_config.get("num_envs", 200),
            max_epochs=raw_config.get("max_epochs", 200),
            sim_dt=raw_config.get("sim_dt", 0.02),
            max_sim_time=raw_config.get("max_sim_time", 3.0),
            delay=raw_config.get("delay", 0.04),
            reward_sharpness=raw_config.get("reward_sharpness", 3.0),
            action_penalty_weight=raw_config.get("action_penalty_weight", 0.5),
            hover_target=raw_config.get("hover_target", [1.5, 0.0, 1.5]),
            sim_dyn_config=sim_dyn_config,
            yaw_scale=raw_config.get("yaw_scale", 1.0),
            pitch_roll_scale=raw_config.get("pitch_roll_scale", 0.1),
            velocity_std=raw_config.get("velocity_std", 0.1),
            omega_std=raw_config.get("omega_std", 0.1),
            margin=raw_config.get("margin", 0.5),
            policy_net=policy_net_config,
            optimizer=optimizer_config,
        )


def create_env(config: StateHoveringConfig) -> HoveringStateEnv:
    """Create the hovering environment with wrappers.

    This function builds a HoveringStateEnv configured according to the
    provided config, wrapped with MinMaxObservationWrapper, LogWrapper,
    and VecEnv for parallel execution.

    Args:
        config: Training configuration containing environment parameters.

    Returns:
        Wrapped HoveringStateEnv ready for vectorized training.
    """
    # Create quadrotor object with dynamics config
    sim_dyn_config_dict = {
        "use_high_fidelity": config.sim_dyn_config.use_high_fidelity,
        "use_forward_residual": config.sim_dyn_config.use_forward_residual,
    }
    quad_obj = Quadrotor.from_name("example_quad", sim_dyn_config_dict)

    # Create base environment
    env = HoveringStateEnv(
        max_steps_in_episode=int(config.max_sim_time / config.sim_dt),
        dt=config.sim_dt,
        delay=config.delay,
        yaw_scale=config.yaw_scale,
        pitch_roll_scale=config.pitch_roll_scale,
        velocity_std=config.velocity_std,
        omega_std=config.omega_std,
        quad_obj=quad_obj,
        reward_sharpness=config.reward_sharpness,
        action_penalty_weight=config.action_penalty_weight,
        margin=config.margin,
        hover_target=config.hover_target,
    )

    # Apply min-max observation wrapper
    env = MinMaxObservationWrapper(env)

    # Apply logging and vectorization wrappers
    env = LogWrapper(env)
    env = VecEnv(env)

    return env


def create_policy(
    config: StateHoveringConfig,
    env: HoveringStateEnv,
    key: jax.Array,
) -> TrainState:
    """Create policy network and training state.

    This function initializes an MLP policy network with the architecture
    specified in the config, creates an optimizer with cosine decay schedule,
    and returns a TrainState object ready for training.

    Args:
        config: Training configuration containing policy network and optimizer settings.
        env: The environment (used to get observation/action dimensions and hovering action).
        key: JAX random key for parameter initialization.

    Returns:
        TrainState with initialized policy parameters and optimizer.
    """
    # Get dimensions from environment
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    # Build network architecture: [obs_dim, hidden1, hidden2, ..., action_dim]
    layer_sizes = [obs_dim] + config.policy_net.hidden_layers + [action_dim]

    # Create policy network
    policy_net = MLP(
        layer_sizes,
        initial_scale=config.policy_net.initial_scale,
        action_bias=env.hovering_action,
    )
    policy_params = policy_net.initialize(key)

    # Create optimizer with cosine decay schedule
    scheduler = optax.cosine_decay_schedule(config.optimizer.initial_lr, config.max_epochs)
    tx = optax.adam(scheduler)

    # Create training state
    train_state = TrainState.create(apply_fn=policy_net.apply, params=policy_params, tx=tx)

    return train_state


def load_dummy_residual_params() -> Any:
    """Load dummy residual dynamics parameters.

    For base policy training, we don't use residual dynamics for forward
    simulation or backpropagation. However, the environment requires
    residual parameters to be passed, so we load a dummy checkpoint.

    Returns:
        Dummy residual dynamics parameters.
    """
    path = LOTF_PATH + "/../checkpoints/residual_dynamics/dummy_params"
    ckptr = PyTreeCheckpointer()
    return ckptr.restore(path)


def save_checkpoint(output_path: str, params: Any) -> None:
    """Save policy parameters to checkpoint.

    Args:
        output_path: Path to save the checkpoint (without extension).
        params: Policy parameters to save.
    """
    # Ensure parent directory exists
    path = Path(output_path)
    # Convert to absolute path (required by orbax)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    ckptr = PyTreeCheckpointer()
    ckptr.save(str(path), params)
    print(f"Policy saved successfully to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train state-based hovering policy using BPTT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default configuration
    uv run python -m lotf.scripts.train_state_hovering

    # Train with custom config
    uv run python -m lotf.scripts.train_state_hovering --config my_config.yaml

    # Save to custom location
    uv run python -m lotf.scripts.train_state_hovering --output checkpoints/policy/my_policy
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/state_hovering.yaml",
        help="Path to YAML configuration file (default: configs/state_hovering.yaml)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/policy/state_hovering_params",
        help="Path to save the trained policy checkpoint (default: checkpoints/policy/state_hovering_params)",
    )

    return parser.parse_args()


def main() -> int:
    """Main training function.

    This function orchestrates the complete training pipeline:
    1. Parse CLI arguments
    2. Load configuration from YAML
    3. Create environment and policy
    4. Run training loop
    5. Save final checkpoint

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Parse arguments
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    try:
        config = StateHoveringConfig.from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error parsing config: {e}", file=sys.stderr)
        return 1

    # Initialize random keys
    print(f"Initializing with seed: {config.seed}")
    key = jax.random.key(config.seed)
    key_init, key_bptt = jax.random.split(key, 2)

    # Create environment
    print("Creating environment...")
    env = create_env(config)

    # Print environment info
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    print(f"Environment info:")
    print(f"  action_dim: {action_dim}")
    print(f"  obs_dim: {obs_dim}")
    print(f"  target hover goal: {env.goal}")
    print(f"  max_steps_in_episode: {env.max_steps_in_episode}")

    # Create policy
    print("Creating policy network...")
    train_state = create_policy(config, env, key_init)

    # Load dummy residual dynamics parameters
    print("Loading dummy residual dynamics parameters...")
    dummy_residual_params = load_dummy_residual_params()

    # Initialize environments
    print(f"Initializing {config.num_envs} parallel environments...")
    key_bptt, key_ = jax.random.split(key_bptt)
    key_reset = jax.random.split(key_, config.num_envs)
    init_env_state, init_obs = env.reset(key_reset, None)

    # Run training
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

    # Compute final reward
    losses = res_dict["metrics"]
    returns = -losses
    final_reward = returns[-1]
    print(f"Final reward: {final_reward:.2f}")

    # Save checkpoint
    trained_policy_params = res_dict["runner_state"].train_state.params
    save_checkpoint(args.output, trained_policy_params)

    print("\nTraining complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
