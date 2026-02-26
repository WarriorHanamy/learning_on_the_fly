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
# Clone the repository
git clone https://github.com/WarriorHanamy/learning_on_the_fly.git
cd learning_on_the_fly

# Install dependencies (CPU only)
uv sync

# Install with GPU support (CUDA 12)
uv sync --extra cuda12
```

## CLI Commands

### Global Commands
```bash
uv run lotf --help                 # Show all commands
uv run lotf --version              # Show package version
uv run lotf --list-configs         # List available configuration files
```

### Residual Dynamics Training
```bash
uv run lotf residual --dataset data.csv
uv run lotf residual --config configs/residual_dynamics.yaml --dataset data.csv
uv run lotf residual --dataset data.csv --output checkpoints/my_model
```

### State-Based Hovering Training
```bash
uv run lotf hover
uv run lotf hover --config configs/state_hovering.yaml
uv run lotf hover --output checkpoints/my_hovering_policy
```

### Trajectory Tracking Training
```bash
uv run lotf track
uv run lotf track --config configs/traj_tracking.yaml
uv run lotf track --checkpoint checkpoints/my_tracking_policy
uv run lotf track --trajectory-output outputs/trajectory.csv
```

### Subcommand Help
```bash
uv run lotf hover --help
uv run lotf track --help
uv run lotf residual --help
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
└── utils/          # Utility functions
```

## Documentation

- [Installation Guide](docs/installation.md) - Setup and environment configuration
- [Configuration Guide](docs/configuration.md) - Complete parameter reference
- [Training Guide](docs/training.md) - Training workflows and examples
- [Deployment Guide](docs/deployment.md) - Docker and ROS2 integration
- [CODEBASE.md](CODEBASE.md) - Detailed codebase documentation
- [USAGE.md](USAGE.md) - Comprehensive CLI usage guide
- [examples/](examples/) - Jupyter notebooks with step-by-step tutorials

## Citation

```bibtex
@inproceedings{pan2026learning,
  title={Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation},
  author={Pan, Jiahe and Xing, Jiaxu and Reiter, Rudolf and Zhai, Yifan and Aljalbout, Elie and Scaramuzza, Davide},
  booktitle = {IEEE Robotics and Automation Letters},
  year={2026}
}
```

## Acknowledgements

We thank the authors of [flightning](https://github.com/uzh-rpg/rpg_flightning) for open-sourcing their code, which provided the foundation of this codebase.

## Contact

For questions, use the [GitHub issue tracker](https://github.com/uzh-rpg/learning_on_the_fly/issues) or contact [Michael Pan](mailto:michael.pan31415@gmail.com).
