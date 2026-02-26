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

## Python Execution Wrapper

The project includes a `bin/python_exec` wrapper script that automatically handles environment setup and cleanup:

```bash
./bin/python_exec <script>
```

**Features:**
- Automatically sets `PYTHONPATH` to project root
- Executes using `uv run python` for dependency management
- Always runs from project root directory
- Simplifies script execution without manual environment setup

**Examples:**
```bash
./bin/python_exec -c "import lotf; print('Success')"
./bin/python_exec examples/some_analysis.py
```

**IMPORTANT:** Never use `python` or `python3` directly. Always use `./bin/python_exec` for the environment.

## CLI Commands

### Global Commands
```bash
./bin/python_exec -m lotf --help           # Show all commands
./bin/python_exec -m lotf --version        # Show package version
./bin/python_exec -m lotf --list-configs   # List available configuration files
```

### Residual Dynamics Training
```bash
./bin/python_exec -m lotf residual --dataset data.csv
./bin/python_exec -m lotf residual --config configs/residual_dynamics.yaml --dataset data.csv
./bin/python_exec -m lotf residual --dataset data.csv --output checkpoints/my_model
```

### State-Based Hovering Training
```bash
./bin/python_exec -m lotf hover
./bin/python_exec -m lotf hover --config configs/state_hovering.yaml
./bin/python_exec -m lotf hover --output checkpoints/my_hovering_policy
```

### Trajectory Tracking Training
```bash
./bin/python_exec -m lotf track
./bin/python_exec -m lotf track --config configs/traj_tracking.yaml
./bin/python_exec -m lotf track --checkpoint checkpoints/my_tracking_policy
./bin/python_exec -m lotf track --trajectory-output outputs/trajectory.csv
```

### Subcommand Help
```bash
./bin/python_exec -m lotf hover --help
./bin/python_exec -m lotf track --help
./bin/python_exec -m lotf residual --help
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
- [Design Decisions](DESIGN_DECISIONS.md) - Key architectural decisions and rationale
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
