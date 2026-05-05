# Learning on the Fly (LOTF)

Learning on the Fly (LOTF) is a JAX-based differentiable simulation library for agile quadrotor flight. The project uses a two-stage training approach: first learning residual dynamics from real hardware data, then training policies in differentiable simulation with the learned residual model.

## Key Features

- **Differentiable physics simulation** with automatic differentiation
- **Residual dynamics learning** to bridge sim-to-real gap
- **BPTT (Backpropagation Through Time)** policy optimization
- Support for state hovering, trajectory tracking, and vision-based hovering
- JAX JIT compilation with GPU acceleration
- Ensemble learning for uncertainty quantification

## Requirements

| Component | Version |
|-----------|---------|
| Ubuntu    | 22.04 LTS |
| Python    | 3.10 (uv managed) |
| CUDA      | 12.x |
| GPU       | NVIDIA GPU with CUDA support |

## Installation

```bash
git clone https://github.com/WarriorHanamy/learning_on_the_fly.git
cd learning_on_the_fly

# CPU only
uv sync

# GPU support (CUDA 12)
uv sync --extra cuda12
```

## CLI Commands

All commands use `uv run train`:

```bash
# Global
uv run train --help
uv run train --version
uv run train --list-configs

# Residual dynamics training
uv run train residual --dataset data.csv
uv run train residual --config configs/residual_dynamics.yaml --dataset data.csv

# State-based hovering
uv run train hover
uv run train hover --config configs/state_hovering.yaml

# Trajectory tracking
uv run train track
uv run train track --config configs/traj_tracking.yaml

# Subcommand help
uv run train hover --help
uv run train track --help
uv run train residual --help
```

## Project Structure

```
lotf/
├── algos/          # BPTT algorithm with differentiable simulation
├── configs/        # YAML configuration loader
├── envs/           # Simulation environments
├── modules/        # MLP networks (policy, residual dynamics, LoRA)
├── objects/        # Quadrotor, reference trajectory, world box
├── scripts/        # Training scripts (CLI entry points)
├── sensors/        # Double sphere camera model
├── simulation/     # High-fidelity quadrotor dynamics
├── utils/          # Utility functions
├── __init__.py     # Package init (LOTF_ROOT, resolve_path)
└── __main__.py     # CLI entry point (uv run train)
```
