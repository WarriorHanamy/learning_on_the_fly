# Installation Guide

This guide provides step-by-step instructions for installing Learning on the Fly (LOTF) with both CPU and GPU support, along with environment verification and troubleshooting.

## System Requirements

| Component | Minimum Version | Recommended Version | Notes |
|-----------|----------------|-------------------|-------|
| Operating System | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS | Linux only (Windows/Mac not tested) |
| Python | 3.12 | 3.12 | Managed by uv, other versions not supported |
| CUDA | 11.8+ | 12.4 | Required for GPU acceleration |
| GPU | NVIDIA GPU with compute capability 7.0+ | RTX 3090, A100, or similar | CUDA-compatible NVIDIA GPU |
| RAM | 8 GB | 16 GB+ | More memory for larger training runs |
| Storage | 10 GB | 50 GB+ | Space for checkpoints and datasets |

### CPU-Only Requirements
For CPU-only installation, CUDA and GPU are not required. Minimum RAM of 8 GB is sufficient for development and testing.

## Prerequisites

Before installing LOTF, ensure you have the following:

1. **Git**: For cloning the repository
   ```bash
   sudo apt-get update
   sudo apt-get install git
   ```

2. **uv**: Python package manager (fast and reliable)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

3. **NVIDIA Drivers** (for GPU support):
   ```bash
   # Check if NVIDIA drivers are installed
   nvidia-smi

   # If not installed, follow NVIDIA's guide:
   # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   ```

## Running Commands

All commands use `uv run`, which automatically manages the virtual environment:

```bash
# Basic Python execution
uv run python --version

# Run training via CLI
uv run train hover --config configs/state_hovering.yaml

# Run tests
uv run pytest

# Install packages
uv pip install package_name
```

## Installation

### CPU Installation

For development and testing without GPU acceleration:

```bash
# Clone the repository
git clone https://github.com/WarriorHanamy/learning_on_the_fly.git
cd learning_on_the_fly

# Install dependencies (CPU only)
uv sync

# Verify installation
uv run python --version
uv run python -c "import lotf; print('LOTF installed successfully')"
```

### GPU Installation (CUDA 12)

For GPU acceleration with CUDA 12 support:

```bash
# Clone the repository
git clone https://github.com/WarriorHanamy/learning_on_the_fly.git
cd learning_on_the_fly

# Install dependencies with CUDA 12 support
uv sync --extra cuda12

# Verify GPU installation
uv run python -c "import jax; print(jax.devices())"
```

### Development Installation

For development with testing tools:

```bash
# Install with development dependencies
uv sync --extra dev --extra cuda12

# Run tests to verify installation
uv run pytest
```

## Environment Verification

After installation, verify your environment with the following commands:

### 1. Python Version Check

```bash
# Check Python version
uv run python --version
```

Expected output:
```
Python 3.12.x
```

### 2. JAX Devices Check

Verify that JAX can detect your hardware:

```bash
# Check available JAX devices
uv run python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
```

For CPU-only installation:
```
JAX version: 0.6.2
Devices: [cpu(id=0)]
```

For GPU installation:
```
JAX version: 0.6.2
Devices: [cuda(id=0)]
```

### 3. CUDA Availability Check

Verify CUDA is properly configured (GPU only):

```bash
# Check CUDA version
nvcc --version

# Check GPU availability with JAX
uv run python -c "import jax; print('GPU Available:', len(jax.devices('gpu')) > 0)"
```

Expected output:
```
nvcc (GCC) 11.x.x
...
release 12.4, V12.4.x
...
GPU Available: True
```

### 4. Package Import Check

Verify all required packages can be imported:

```bash
uv run python << 'EOF'
import jax
import jaxlib
import flax
import optax
import orbax_checkpoint
import chex
import numpy
import scipy
print("All packages imported successfully!")
EOF
```

### 5. CLI Functionality Check

Test the LOTF CLI:

```bash
# Show help
uv run train --help

# Show version
uv run train --version

# List available configs
uv run train --list-configs
```

## Troubleshooting

### GPU JVP Errors

**Problem**: `jaxlib.xla_extension.XlaRuntimeError: GPU is required but not available` or errors with JVP (Jacobian-Vector Product) operations.

**Solutions**:

1. **Check JAX backend configuration**:
    ```bash
    uv run python -c "import os; os.environ['JAX_PLATFORMS']='cpu'; import jax; print(jax.devices())"
    ```

2. **Force GPU backend**:
    ```bash
    export JAX_PLATFORMS=cuda
    uv run python -c "import jax; print(jax.devices())"
    ```

3. **Reinstall with CUDA support**:
    ```bash
    # Uninstall and reinstall with CUDA
    rm -rf .venv
    uv sync --extra cuda12
    ```

4. **Check CUDA installation**:
    ```bash
    # Verify CUDA libraries are in PATH
    echo $LD_LIBRARY_PATH

    # Add CUDA libraries to PATH if missing
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    ```

### Missing Dependencies

**Problem**: `ModuleNotFoundError: No module named '...'`

**Solutions**:

1. **Reinstall dependencies**:
   ```bash
   uv sync --reinstall
   ```

2. **Check uv installation**:
   ```bash
   # Verify uv is working
   uv --version

   # Update uv to latest version
   uv self update
   ```

3. **Clear Python cache**:
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} +
   find . -type f -name "*.pyc" -delete
   ```

4. **Verify virtual environment**:
   ```bash
   # Check if .venv exists
   ls -la .venv/

   # Recreate virtual environment
   rm -rf .venv
   uv sync
   ```

### CUDA Compatibility Issues

**Problem**: `RuntimeError: CUDA out of memory` or CUDA version mismatch.

**Solutions**:

1. **Check CUDA version compatibility**:
   ```bash
   # JAX requires CUDA 11.8+ for jaxlib 0.4.30+
   nvidia-smi
   nvcc --version
   ```

2. **Reduce memory usage**:
   ```bash
   # Set environment variables to limit GPU memory preallocation
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_MEM_FRACTION=.8
   ```

3. **Use CPU fallback**:
    ```bash
    # Run training on CPU if GPU issues persist
    export JAX_PLATFORMS=cpu
    uv run train hover
    ```

### Import Errors

**Problem**: `ImportError: cannot import name '...' from '...'`

**Solutions**:

1. **Check package versions**:
   ```bash
   uv pip list | grep -E "jax|flax|optax"
   ```

2. **Update conflicting packages**:
   ```bash
   uv pip install --upgrade jax jaxlib flax optax
   ```

3. **Reinstall from scratch**:
    ```bash
    rm -rf .venv uv.lock
    uv sync --extra cuda12
    ```


## Additional Resources

- [Project README](../README.md) - Overview and quick start
- [CODEBASE.md](../CODEBASE.md) - Detailed codebase documentation
- [USAGE.md](../USAGE.md) - CLI usage guide
- [examples/](../examples/) - Jupyter notebook tutorials

## Getting Help

If you encounter issues not covered in this guide:

1. Search existing [GitHub Issues](https://github.com/uzh-rpg/learning_on_the_fly/issues)
2. Create a new issue with:
   - System information (OS, Python version, CUDA version)
   - Error messages
   - Steps to reproduce
   - Installation method (CPU/GPU)

3. Contact: [Michael Pan](mailto:michael.pan31415@gmail.com)
