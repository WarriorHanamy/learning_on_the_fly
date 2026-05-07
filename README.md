# Learning on the Fly (LOTF)

Learning on the Fly (LOTF) is a JAX-based differentiable simulation library for agile quadrotor flight. The project uses a two-stage training approach: first learning residual dynamics from real hardware data, then training policies in differentiable simulation with the learned residual model.

## Key Features

- **Differentiable physics simulation** with automatic differentiation
- **Residual acceleration learning** to bridge sim-to-real gap
- **BPTT (Backpropagation Through Time)** policy optimization
- Trajectory tracking and residual dynamics training
- JAX JIT compilation with GPU acceleration
- Ensemble learning for uncertainty quantification

## Requirements

| Component | Version |
|-----------|---------|
| Ubuntu    | 22.04 LTS |
| Python    | 3.12 (uv managed) |
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

# Trajectory tracking: default trains all four standard settings
uv run train track
uv run train track --config configs/traj_tracking.yaml
uv run train track --setting full

# Benchmark comparison: default discovers all four standard checkpoints
uv run eval track
uv run eval track --checkpoint checkpoints/policy/traj_tracking_params__full

# Interactive playback: defaults to latest full checkpoint
uv run play track
uv run play track --setting nominal

# Subcommand help
uv run train track --help
uv run train residual --help
```

## Project Structure

```
lotf/
├── algos/          # BPTT algorithm with differentiable simulation
├── envs/           # Simulation environments
├── eval/           # Benchmark evaluation runner
├── modules/        # MLP networks (policy, residual dynamics, LoRA)
├── objects/        # Quadrotor, reference trajectory, world box
├── scripts/        # Training and evaluation scripts (CLI entry points)
├── simulation/     # Quadrotor rotor dynamics augmentation
├── utils/          # Utility functions
├── forward_model_config.py     # Forward model fidelity config
├── traj_tracking_setup.py      # Shared trajectory tracking builders
├── __init__.py     # Package init (LOTF_ROOT, resolve_path)
└── __main__.py     # CLI entry points (train, play, eval)
```
