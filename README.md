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

## Docker Tools

The project includes two Docker utilities for streamlined container management:

### dockrun.py - Simplified Docker Command Runner

`dockrun.py` provides a simplified interface for running commands in the LOTF Docker container with pre-configured parameters. It eliminates the need to manually specify common Docker options.

**Fixed Parameters (hardcoded):**
- **Image**: `lotf:latest`
- **GPU Support**: `--gpus=all` (all available GPUs)
- **Volume Mount**: `-v $(pwd):/app` (mounts current directory to /app)
- **Working Directory**: `-w /app`
- **Auto-Remove**: `--rm` (automatically removes container after execution)

**Basic Usage:**
```bash
# Run a command in non-interactive mode (recommended)
python3 dockrun.py --non-interactive [command]

# Show version
python3 dockrun.py --version

# Show help
python3 dockrun.py --help
```

**Examples:**
```bash
# Run residual dynamics training
python3 dockrun.py --non-interactive ./bin/python_exec -m lotf residual --dataset data.csv

# Run state hovering training
python3 dockrun.py --non-interactive ./bin/python_exec -m lotf hover

# Run with custom config
python3 dockrun.py --non-interactive ./bin/python_exec -m lotf track --config configs/my_config.yaml

# Show nested command help (e.g., for lotf hover)
python3 dockrun.py --non-interactive ./bin/python_exec -m lotf hover --help
```

**Key Features:**
- **Non-interactive mode**: Default behavior, suitable for CI/CD pipelines
- **Nested command support**: Passes flags to nested commands (e.g., `--help` works for `lotf` subcommands)
- **Fixed volume mounting**: Automatically mounts current working directory for seamless file access
- **GPU-enabled**: All GPUs are available by default for JAX/CUDA operations
- **Automatic cleanup**: Containers are removed after execution to prevent resource accumulation

**Note on Fixed Parameters:**
All Docker parameters are hardcoded and cannot be changed via command-line arguments. This design ensures consistency across runs and simplifies usage. If you need different parameters, use `docker run` directly (see [Docker Setup](docker/README.md)).

### dockerfile_utils.py - Dockerfile Management

`dockerfile_utils.py` provides utilities for managing Dockerfile and building Docker images.

**Commands:**

1. **init** - Create Dockerfile in `.dockman/` directory:
```bash
python3 dockerfile_utils.py init

# Force overwrite existing Dockerfile
python3 dockerfile_utils.py init --force
```

This creates a minimal Dockerfile based on `lotf:latest` image with:
- Working directory set to `/app`
- PATH configured to include uv
- Default command to show lotf help

2. **build** - Build Docker image tagged as `lotf:latest`:
```bash
python3 dockerfile_utils.py build
```

This builds the Docker image using the `Dockerfile` in the project root directory. The resulting image is tagged as `lotf:latest`, which is required by `dockrun.py`.

3. **version** - Show tool version:
```bash
python3 dockerfile_utils.py version
```

**Typical Workflow:**
```bash
# 1. Build the lotf:latest image
python3 dockerfile_utils.py build

# 2. Use dockrun to run commands in the container
python3 dockrun.py --non-interactive ./bin/python_exec -m lotf hover
```

**Note:** The `build` command requires a `Dockerfile` to exist in the project root. See [Docker Setup](docker/README.md) for more details on the Dockerfile contents.

### Troubleshooting Docker Tools

**Issue: "lotf:latest" image not found**
```
Error: Unable to find image 'lotf:latest' locally
```
**Solution:** Build the image first using `dockerfile_utils.py build`:
```bash
python3 dockerfile_utils.py build
```

**Issue: Volume mount permission errors**
```
PermissionError: [Errno 13] Permission denied: '/app/checkpoints/...'
```
**Solution:** The volume mount is bidirectional. Files created inside the container are visible on the host, but permissions may be affected by UID/GID differences. Consider:
1. Using `sudo` for host operations if needed
2. Ensuring proper file permissions on the host side
3. Running containers with user mapping if required

**Issue: GPU not available inside container**
```
RuntimeError: No GPU devices found
```
**Solution:** Verify that:
1. NVIDIA Docker runtime is installed: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
2. Host has CUDA-capable GPUs: `nvidia-smi`
3. Docker daemon is running with GPU support enabled

**Issue: Nested command flags not working**
```
Error: unrecognized arguments: --help
```
**Solution:** Use `--non-interactive` flag when running commands that have their own `--help`:
```bash
# Correct: dockrun passes --help to the nested command
python3 dockrun.py --non-interactive ./bin/python_exec -m lotf hover --help
```

**Issue: Container cleanup (dangling containers)**
```
WARNING: Multiple containers with name found
```
**Solution:** While `dockrun.py` uses `--rm` to auto-remove containers, you may occasionally need to clean up manually:
```bash
# Remove stopped containers
docker container prune

# Remove all unused containers
docker system prune -a
```

**Issue: Changes not persisted between container runs**
**Solution:** Remember that `dockrun.py` mounts `$(pwd):/app`. Any changes made inside `/app` in the container are automatically reflected on the host. However, changes outside `/app` are lost when the container exits (due to `--rm`).

For more detailed Docker configuration and troubleshooting, see [Docker Setup](docker/README.md).

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
 - [Design Decisions](https://github.com/WarriorHanamy/learning_on_the_fly/wiki/Design-Decisions) - Key architectural decisions and rationale
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
