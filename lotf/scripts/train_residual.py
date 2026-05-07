#!/usr/bin/env python3
"""Residual dynamics ensemble training script.

This script trains an ensemble of residual dynamics neural networks for
learning quadrotor dynamics residuals. The implementation is modularized from
the original notebook examples/residual_dynamics/train_ensemble_model.ipynb.

Usage:
    uv run python -m lotf.scripts.train_residual --dataset path/to/dataset.csv
    uv run python -m lotf.scripts.train_residual --config configs/residual_dynamics.yaml --dataset examples/residual_dynamics/example_dataset.csv
    uv run python -m lotf.scripts.train_residual --config configs/residual_dynamics.yaml --dataset examples/residual_dynamics/example_dataset.csv --output checkpoints/residual_dynamics/my_model

CLI Arguments:
    --config: Path to YAML configuration file (default: configs/residual_dynamics.yaml)
    --dataset: Path to CSV dataset file (required)
    --output: Path to save the trained ensemble checkpoint (default: checkpoints/residual_dynamics/residual_params)

Examples:
    # Train with default configuration
    uv run python -m lotf.scripts.train_residual --dataset examples/residual_dynamics/example_dataset.csv

    # Train with custom config and output location
    uv run python -m lotf.scripts.train_residual \\
        --config configs/residual_dynamics.yaml \\
        --dataset examples/residual_dynamics/example_dataset.csv \\
        --output checkpoints/residual_dynamics/custom_model

    # Quick training run with fewer epochs (modify config file)
    uv run python -m lotf.scripts.train_residual --config configs/residual_dynamics.yaml --dataset examples/residual_dynamics/example_dataset.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
import pandas as pd
import yaml
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer

from lotf import LOTF_ROOT, resolve_path
from lotf.utils.residual_dynamics import create_vec_funcs


@dataclass
class ResidualDynamicsConfig:
    """Configuration for residual dynamics ensemble training.

    Attributes:
        num_models: Number of ensemble members
        input_dim: Input dimension (state + action features)
        output_dim: Output dimension (residual prediction)
        learning_rate: Optimizer learning rate
        lambda_reg: Weight regularization coefficient
        num_epochs: Number of training epochs
        batch_size: Training batch size
        eval_every: Log metrics every N epochs
        weight_init_scale: Scale for weight initialization
    """

    num_models: int = 3
    input_dim: int = 19
    output_dim: int = 3
    learning_rate: float = 0.01
    lambda_reg: float = 0.001
    num_epochs: int = 100
    batch_size: int = 256
    eval_every: int = 10
    weight_init_scale: float = 1.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> ResidualDynamicsConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            ResidualDynamicsConfig instance with loaded values.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If YAML parsing fails.
        """
        path = Path(path)
        if not path.is_absolute():
            path = LOTF_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:
            raise ValueError(f"Configuration file is empty: {path}")

        return cls(
            num_models=raw_config.get("num_models", 3),
            input_dim=raw_config.get("input_dim", 19),
            output_dim=raw_config.get("output_dim", 3),
            learning_rate=raw_config.get("learning_rate", 0.01),
            lambda_reg=raw_config.get("lambda_reg", 0.001),
            num_epochs=raw_config.get("num_epochs", 100),
            batch_size=raw_config.get("batch_size", 256),
            eval_every=raw_config.get("eval_every", 10),
            weight_init_scale=raw_config.get("weight_init_scale", 1.0),
        )


def load_dataset(path: str, input_dim: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load dataset from CSV file and return JAX arrays.

    Args:
        path: Path to the CSV file (no header, comma-separated).
        input_dim: Number of input features (columns for X).

    Returns:
        Tuple of (X, y) JAX arrays with float32 dtype.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    file_path = resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Read CSV without header
    df = pd.read_csv(file_path, header=None)
    dataset = df.to_numpy()

    # Split into input and output
    X = dataset[:, :input_dim]
    y = dataset[:, input_dim:]

    # Convert to JAX arrays with float32
    X = jnp.array(X, dtype=jnp.float32)
    y = jnp.array(y, dtype=jnp.float32)

    return X, y


def create_ensemble(
    config: ResidualDynamicsConfig,
) -> Tuple[jnp.ndarray, TrainState]:
    """Create ensemble of residual dynamics models.

    Uses vectorized initialization from create_vec_funcs() to initialize
    multiple ensemble members with different random seeds.

    Args:
        config: Training configuration with num_models and learning_rate.

    Returns:
        Tuple of (model_params, train_states) for all ensemble members.
    """
    # Get vectorized functions
    init_fn, _, _ = create_vec_funcs()

    # Create seeds for each ensemble member
    seeds = jnp.arange(config.num_models, dtype=jnp.int32)

    # Initialize all ensemble members in parallel
    model_params, train_states = init_fn(config.learning_rate, seeds)

    return model_params, train_states


def get_unique_checkpoint_path(base_path: Path) -> Path:
    """Generate a unique checkpoint path by appending timestamp if directory exists.

    Args:
        base_path: Base checkpoint path (without extension).

    Returns:
        Unique checkpoint path (either original or with timestamp suffix).
    """
    path = resolve_path(base_path) if isinstance(base_path, Path) else resolve_path(base_path)
    if not path.exists():
        return path

    # Generate timestamp suffix
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = path.parent / f"{path.name}_{timestamp}"

    print(f"Checkpoint directory exists, using: {new_path}")
    return new_path


def save_checkpoint(output_path: str, params: jnp.ndarray) -> None:
    """Save ensemble parameters to checkpoint.

    Args:
        output_path: Path to save the checkpoint (without extension).
        params: Ensemble parameters to save.
    """
    # Ensure parent directory exists and generate unique path
    path = get_unique_checkpoint_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckptr = PyTreeCheckpointer()
    ckptr.save(str(path), params)
    print(f"Saved model params to: {path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train residual dynamics ensemble model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default dataset and configuration
    uv run python -m lotf.scripts.train_residual

    # Train with custom dataset
    uv run python -m lotf.scripts.train_residual --dataset my_data.csv

    # Train with custom config and output
    uv run python -m lotf.scripts.train_residual --config my_config.yaml --dataset my_data.csv --output checkpoints/my_model
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/residual_dynamics.yaml",
        help="Path to YAML configuration file (default: configs/residual_dynamics.yaml)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="examples/residual_dynamics/example_dataset.csv",
        help="Path to CSV dataset file (default: examples/residual_dynamics/example_dataset.csv)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/residual_dynamics/residual_params",
        help="Path to save the trained ensemble checkpoint (default: checkpoints/residual_dynamics/residual_params)",
    )

    return parser.parse_args()


def main() -> int:
    """Main training function.

    This function orchestrates the complete training pipeline:
    1. Parse CLI arguments
    2. Load configuration from YAML
    3. Load dataset
    4. Create ensemble models
    5. Run training loop
    6. Save final checkpoint

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Parse arguments
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    try:
        config = ResidualDynamicsConfig.from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error parsing config: {e}", file=sys.stderr)
        return 1

    # Print configuration
    print(f"Configuration:")
    print(f"  num_models: {config.num_models}")
    print(f"  input_dim: {config.input_dim}")
    print(f"  output_dim: {config.output_dim}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  lambda_reg: {config.lambda_reg}")
    print(f"  num_epochs: {config.num_epochs}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  eval_every: {config.eval_every}")

    # Load dataset
    print(f"\nLoading dataset from: {args.dataset}")
    try:
        X, y = load_dataset(args.dataset, config.input_dim)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Create ensemble
    print(f"\nCreating ensemble with {config.num_models} models...")
    model_params, train_states = create_ensemble(config)

    # Get training function
    _, train_fn, _ = create_vec_funcs()

    # Run training
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print("-" * 60)

    tic = time.time()
    train_states = train_fn(
        train_states, X, y, config.lambda_reg, config.num_epochs, config.eval_every
    )
    elapsed = time.time() - tic

    print("-" * 60)
    print(f"Residual model training took {elapsed:.2f} seconds")

    # Save checkpoint
    residual_params = train_states.params
    save_checkpoint(args.output, residual_params)

    print("\nTraining complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
