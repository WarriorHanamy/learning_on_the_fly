# Installation Guide

This guide provides step-by-step instructions for installing Learning on the Fly (LOTF) with both CPU and GPU support, along with environment verification and troubleshooting.

## System Requirements

| Component | Minimum Version | Recommended Version | Notes |
|-----------|----------------|-------------------|-------|
| Operating System | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS | Linux only (Windows/Mac not tested) |
| Python | 3.10 | 3.10 | Managed by uv, other versions not supported |
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

## Python Execution Wrapper

LOTF provides a convenient `python_exec` wrapper script that simplifies Python script execution within the project environment. This wrapper automatically handles environment setup and ensures consistent execution across different contexts.

### Features

- **Automatic PYTHONPATH Configuration**: Sets `PYTHONPATH` to include the project root directory
- **Project Root Context**: Always executes from the project root directory, ensuring consistent imports
- **Seamless uv Integration**: Uses `uv run python` internally for dependency management

### Usage

The `python_exec` script is located at `./bin/python_exec` and can be used as a drop-in replacement for `python` commands:

```bash
# Basic Python execution
./bin/python_exec --version

# Run Python scripts
./bin/python_exec train.py

# Execute inline Python code
./bin/python_exec -c "import jax; print(jax.devices())"

# Install packages through the wrapper
./bin/python_exec -m pip install package_name
```

### Comparison with Direct uv Commands

| Aspect | `uv run python` | `./bin/python_exec` |
|--------|-----------------|---------------------|
| PYTHONPATH Configuration | Manual | Automatic |
| Directory Context | Current directory | Always project root |
| Recommended For | One-off scripts, debugging | Daily development and training workflows |

## Installation

### CPU Installation

For development and testing without GPU acceleration:

```bash
# Clone the repository
git clone https://github.com/WarriorHanamy/learning_on_the_fly.git
cd learning_on_the_fly

# Install dependencies (CPU only)
uv sync

# Verify installation using python_exec wrapper
./bin/python_exec --version
./bin/python_exec -c "import lotf; print('LOTF installed successfully')"
```

### GPU Installation (CUDA 12)

For GPU acceleration with CUDA 12 support:

```bash
# Clone the repository
git clone https://github.com/WarriorHanamy/learning_on_the_fly.git
cd learning_on_the_fly

# Install dependencies with CUDA 12 support
uv sync --extra cuda12

# Verify GPU installation using python_exec wrapper
./bin/python_exec -c "import jax; print(jax.devices())"
```

### Development Installation

For development with testing tools:

```bash
# Install with development dependencies
uv sync --extra dev --extra cuda12

# Run tests to verify installation using python_exec wrapper
./bin/python_exec -m pytest
```

## Environment Verification

After installation, verify your environment with the following commands:

### 1. Python Version Check

```bash
# Check Python version using python_exec wrapper
./bin/python_exec --version
```

Expected output:
```
Python 3.10.x
```

### 2. JAX Devices Check

Verify that JAX can detect your hardware:

```bash
# Check available JAX devices using python_exec wrapper
./bin/python_exec -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
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

# Check GPU availability with JAX using python_exec wrapper
./bin/python_exec -c "import jax; print('GPU Available:', len(jax.devices('gpu')) > 0)"
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
./bin/python_exec << 'EOF'
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

Test the LOTF CLI using python_exec wrapper:

```bash
# Show help
./bin/python_exec -m lotf --help

# Show version
./bin/python_exec -m lotf --version

# List available configs
./bin/python_exec -m lotf --list-configs
```

## Troubleshooting

### GPU JVP Errors

**Problem**: `jaxlib.xla_extension.XlaRuntimeError: GPU is required but not available` or errors with JVP (Jacobian-Vector Product) operations.

**Solutions**:

1. **Check JAX backend configuration**:
    ```bash
    ./bin/python_exec -c "import os; os.environ['JAX_PLATFORMS']='cpu'; import jax; print(jax.devices())"
    ```

2. **Force GPU backend**:
    ```bash
    export JAX_PLATFORMS=cuda
    ./bin/python_exec -c "import jax; print(jax.devices())"
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
    ./bin/python_exec -m lotf hover
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

### python_exec Wrapper Issues

**Problem**: `bash: ./bin/python_exec: Permission denied`

**Solution**: Make the script executable:
```bash
chmod +x ./bin/python_exec
```

---

**Problem**: `Virtual environment not found. Setting up...` appears every time

**Solution**: This is normal behavior - python_exec automatically creates the virtual environment if it doesn't exist. To ensure it persists, run:
```bash
# Verify virtual environment exists
ls -la .venv/

# If missing, explicitly create it
uv sync --extra cuda12
```

---

**Problem**: `ModuleNotFoundError` when using python_exec

**Solutions**:

1. **Check PYTHONPATH configuration**:
    ```bash
    # Manually set PYTHONPATH as a fallback
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    ./bin/python_exec -c "import lotf; print('LOTF module found')"
    ```

2. **Verify virtual environment is synced**:
    ```bash
    # Re-sync dependencies
    uv sync --reinstall
    ```

3. **Check script integrity**:
    ```bash
    # Verify python_exec script exists and is correct
    cat ./bin/python_exec
    ```

---

**Problem**: `Error: Could not find a matching version` or similar dependency conflicts

**Solutions**:

1. **Clear lock file and resync**:
    ```bash
    rm -rf .venv uv.lock
    uv sync --extra cuda12
    ```

2. **Use python_exec's automatic setup**:
    ```bash
    # Remove existing environment and let python_exec recreate it
    rm -rf .venv
    ./bin/python_exec --version
    ```

3. **Check uv version**:
    ```bash
    uv --version
    uv self update
    ```

---

**Problem**: Python scripts cannot import LOTF modules

**Solutions**:

1. **Verify project root context**:
    ```bash
    # python_exec should always execute from project root
    ./bin/python_exec -c "import os; print('Current directory:', os.getcwd())"
    ```

2. **Check LOTF installation**:
    ```bash
    # Verify LOTF is installed in the virtual environment
    ./bin/python_exec -c "import sys; print(sys.path)"
    ```

3. **Reinstall LOTF in development mode**:
    ```bash
    uv pip install --use-pep517 -e .
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
