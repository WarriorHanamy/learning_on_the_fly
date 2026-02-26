# Deployment Guide

This guide covers Docker deployment, checkpoint management, and ROS2 integration for the Learning on the Fly (LOTF) project.

## Docker Deployment

### Building the Docker Image

Build the Docker image with the custom UV_INDEX_URL argument:

```bash
# Build with default Tsinghua mirror (China)
docker build -t lotf:latest --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .

# Build with PyPI (international)
docker build -t lotf:latest --build-arg UV_INDEX_URL=https://pypi.org/simple .

# Build with custom mirror
docker build -t lotf:latest --build-arg UV_INDEX_URL=https://your-mirror.com/simple .
```

### Running the Container

#### Basic GPU Usage

```bash
docker run --gpus all -it -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple lotf:latest
```

#### Interactive Shell

```bash
docker run --gpus all -it -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple --entrypoint /bin/bash lotf:latest
```

#### Running Training Scripts

```bash
# Train state hovering
docker run --gpus all -it -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -v $(pwd)/checkpoints:/app/checkpoints lotf:latest uv run python -m lotf.scripts.train_state_hovering --config configs/state_hovering.yaml --output checkpoints/policy/my_policy

# Train trajectory tracking
docker run --gpus all -it -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -v $(pwd)/checkpoints:/app/checkpoints lotf:latest uv run python -m lotf.scripts.train_traj_tracking --config configs/traj_tracking.yaml --checkpoint checkpoints/policy/my_policy

# Train residual dynamics
docker run --gpus all -it -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -v $(pwd)/checkpoints:/app/checkpoints lotf:latest uv run python -m lotf.scripts.train_residual --config configs/residual_dynamics.yaml --dataset data.csv --output checkpoints/residual_dynamics/my_model
```

#### Running Jupyter Notebook

```bash
docker run --gpus all -it -p 8888:8888 -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -v $(pwd):/app lotf:latest uv run python -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### Running Tests

```bash
docker run --gpus all -it -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple lotf:latest uv run python -m pytest tests/
```

### Docker Volume Mounting

Mount local directories for persistent storage:

```bash
# Mount checkpoints directory
docker run --gpus all -it -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -v $(pwd)/checkpoints:/app/checkpoints lotf:latest

# Mount examples directory
docker run --gpus all -it -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -v $(pwd)/examples:/app/examples lotf:latest

# Mount entire project for development
docker run --gpus all -it -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -v $(pwd):/app lotf:latest
```

### Essential Docker and uv Patterns

#### Build Time vs Runtime Arguments

```bash
# Build-time: use --build-arg for UV_INDEX_URL (embedded in image)
docker build -t lotf:latest --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .

# Runtime: use -e for UV_INDEX_URL (overrides environment variable)
docker run --gpus all -it -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple lotf:latest
```

#### Entry Point Pattern

Always use `uv run python` as the entry point for running Python code:

```bash
# Correct: uv run python ensures proper environment
docker run lotf:latest uv run python -m lotf --help

# Incorrect: direct python may use system Python
docker run lotf:latest python -m lotf --help
```

#### GPU Access Pattern

Always include `--gpus all` for GPU-accelerated workloads:

```bash
docker run --gpus all -it lotf:latest uv run python -c "import jax; print(jax.devices())"
```

#### Mirror Selection

Use the appropriate mirror based on your location:

- **China**: `https://pypi.tuna.tsinghua.edu.cn/simple`
- **International**: `https://pypi.org/simple`
- **Custom**: Your organization's PyPI mirror

## Checkpoint Management

LOTF uses Orbax's `PyTreeCheckpointer` for saving and loading model parameters.

### Checkpoint Directory Structure

```
checkpoints/
├── policy/
│   ├── state_hovering_params/
│   │   ├── _CHECKPOINT_METADATA
│   │   ├── _METADATA
│   │   ├── _sharding
│   │   └── manifest.ocdbt
│   └── traj_tracking_params/
└── residual_dynamics/
    ├── example_params/
    ├── dummy_params/
    └── residual_params/
```

### Saving Checkpoints

#### Save Policy Parameters

```python
from pathlib import Path
from orbax.checkpoint import PyTreeCheckpointer

def save_checkpoint(output_path: str, params) -> None:
    """Save policy parameters to checkpoint.

    Args:
        output_path: Path to save the checkpoint (without extension).
        params: Policy parameters to save.
    """
    path = Path(output_path)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    ckptr = PyTreeCheckpointer()
    ckptr.save(str(path), params)
    print(f"Checkpoint saved to: {path}")
```

#### Save Residual Dynamics Parameters

```python
def save_residual_checkpoint(output_path: str, params: jnp.ndarray) -> None:
    """Save ensemble parameters to checkpoint.

    Args:
        output_path: Path to save the checkpoint (without extension).
        params: Residual dynamics parameters to save.
    """
    path = Path(output_path)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    ckptr = PyTreeCheckpointer()
    ckptr.save(str(path), params)
    print(f"Residual checkpoint saved to: {path}")
```

### Loading Checkpoints

#### Load Policy Parameters

```python
from orbax.checkpoint import PyTreeCheckpointer

def load_checkpoint(checkpoint_path: str):
    """Load policy parameters from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory.

    Returns:
        Loaded parameters.
    """
    path = Path(checkpoint_path).resolve()
    ckptr = PyTreeCheckpointer()
    params = ckptr.restore(str(path))
    print(f"Checkpoint loaded from: {path}")
    return params
```

#### Load Residual Dynamics Parameters

```python
def load_residual_checkpoint(checkpoint_path: str) -> jnp.ndarray:
    """Load residual dynamics parameters from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory.

    Returns:
        Loaded residual dynamics parameters.
    """
    path = Path(checkpoint_path).resolve()
    ckptr = PyTreeCheckpointer()
    params = ckptr.restore(str(path))
    print(f"Residual checkpoint loaded from: {path}")
    return params
```

### Checkpoint Operations in Training Scripts

#### State Hovering Training

```python
# Save trained policy
trained_policy_params = runner_state.train_state.params
save_checkpoint("checkpoints/policy/my_hovering_policy", trained_policy_params)

# Load dummy residual dynamics (for initialization)
path = "checkpoints/residual_dynamics/dummy_params"
ckptr = PyTreeCheckpointer()
dummy_residual = ckptr.restore(path)
```

#### Trajectory Tracking Training

```python
# Save trained policy
trained_policy_params = runner_state.train_state.params
save_checkpoint("checkpoints/policy/my_tracking_policy", trained_policy_params)

# Load residual dynamics for simulation
residual_params = load_residual_checkpoint("checkpoints/residual_dynamics/residual_params")
```

#### Residual Dynamics Training

```python
# Save trained ensemble
residual_params = trained_ensemble
save_residual_checkpoint("checkpoints/residual_dynamics/my_ensemble", residual_params)
```

### Checkpoint Verification

Verify checkpoint integrity:

```python
from orbax.checkpoint import PyTreeCheckpointer
import jax.numpy as jnp

def verify_checkpoint(checkpoint_path: str) -> bool:
    """Verify checkpoint can be loaded.

    Args:
        checkpoint_path: Path to the checkpoint directory.

    Returns:
        True if checkpoint is valid, False otherwise.
    """
    try:
        ckptr = PyTreeCheckpointer()
        params = ckptr.restore(checkpoint_path)
        print(f"Checkpoint valid. Shape: {params.shape if hasattr(params, 'shape') else 'N/A'}")
        return True
    except Exception as e:
        print(f"Checkpoint invalid: {e}")
        return False
```

### Checkpoint Backup

```bash
# Backup checkpoints directory
tar -czf checkpoints_backup_$(date +%Y%m%d).tar.gz checkpoints/

# Restore from backup
tar -xzf checkpoints_backup_20250101.tar.gz
```

## ROS2 Integration

### ROS2 Humble Compatibility

LOTF requires **Python 3.10** which is compatible with ROS2 Humble Hawksbill. The project's `pyproject.toml` enforces this constraint:

```toml
requires-python = ">=3.10,<3.11"
```

### ROS2 Workspace Setup

#### Install ROS2 Humble

```bash
# Add ROS2 apt repository (Ubuntu 22.04)
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop -y
```

#### Source ROS2 Environment

Before using ROS2 tools or running ROS2-dependent code, you must source the ROS2 environment:

```bash
# Source ROS2 environment (required for all ROS2 commands)
source /opt/ros/humble/setup.bash

# Verify installation
ros2 --version
# Expected output: ros2 version x.x.x
```

#### Source Workspace (if using a ROS2 workspace)

If you have a custom ROS2 workspace for sensor drivers or custom messages:

```bash
# Source workspace setup script
source /path/to/your/workspace/install/setup.bash

# This makes custom packages and messages available
ros2 pkg list | grep your_package
```

### ROS2 Integration with LOTF

#### Sourcing Requirements

For scripts that interact with ROS2 sensors or messages:

```bash
# 1. Source ROS2 environment
source /opt/ros/humble/setup.bash

# 2. Source workspace (if using custom ROS2 packages)
source /path/to/workspace/install/setup.bash

# 3. Run LOTF with ROS2 integration
uv run lotf track --config configs/traj_tracking.yaml
```

#### Python Scripts with ROS2

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from lotf.scripts.train_traj_tracking import main as train_main

def main():
    # Initialize ROS2
    rclpy.init()
    
    # Run LOTF training
    train_main()
    
    # Shutdown ROS2
    rclpy.shutdown()

if __name__ == "__main__":
    main()
```

#### Docker with ROS2

Build Docker image with ROS2 Humble support:

```dockerfile
# Add to Dockerfile after base image setup
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y ros-humble-desktop \
    && rm -rf /var/lib/apt/lists/*
```

Run Docker container with ROS2:

```bash
docker run --gpus all -it -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple lotf:latest bash -c "source /opt/ros/humble/setup.bash && uv run python -m lotf --help"
```

### ROS2-Specific Builds

Keep ROS2-specific builds isolated in dedicated workspaces. The main LOTF repository expects only the Python package to be editable.

```bash
# Create ROS2 workspace
mkdir -p ros2_ws/src
cd ros2_ws

# Clone sensor drivers or custom ROS2 packages
cd src
git clone https://github.com/your-org/your-ros2-package.git

# Build ROS2 workspace
cd ..
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# Source workspace
source install/setup.bash

# Use LOTF with ROS2 integration
cd /path/to/lotf
uv run lotf track --config configs/traj_tracking.yaml
```

## Troubleshooting

### Docker Issues

#### GPU Not Available in Container

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Check NVIDIA container toolkit
dpkg -l | grep nvidia-container-toolkit

# Install if missing
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Permission Denied on Checkpoints

```bash
# Fix checkpoint directory permissions
sudo chown -R $USER:$USER checkpoints/
chmod -R 755 checkpoints/
```

### Checkpoint Issues

#### Checkpoint Load Failure

```python
# Verify checkpoint structure
import os
checkpoint_path = "checkpoints/policy/my_policy"
print(os.listdir(checkpoint_path))

# Should contain: _CHECKPOINT_METADATA, _METADATA, manifest.ocdbt, etc.
```

#### Orbax Version Mismatch

```bash
# Reinstall with correct Orbax version
uv pip install orbax-checkpoint==0.6.4 --force-reinstall
```

### ROS2 Issues

#### ROS2 Environment Not Sourced

```bash
# Check if ROS2 is sourced
echo $ROS_DISTRO
# Expected output: humble

# If empty, source ROS2 environment
source /opt/ros/humble/setup.bash
```

#### ROS2 Workspace Build Errors

```bash
# Clean build
cd ros2_ws
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# Check for missing dependencies
rosdep install --from-paths src --ignore-src -y
```

## Additional Resources

- [Installation Guide](installation.md) - Setup instructions for local development
- [Configuration Guide](configuration.md) - Configuration file reference
- [Training Guide](training.md) - Training workflows and best practices
- [Docker Hub](https://hub.docker.com/) - Official Docker images
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/) - ROS2 documentation
