# LOTF CLI Usage Guide

This guide provides detailed instructions for using the LOTF command-line interface for training tasks.

## Prerequisites

Ensure you have [uv](https://docs.astral.sh/uv/) installed. The project uses `uv` for dependency management and virtual environment handling.

## Installation

```bash
# Clone the repository
git clone git@github.com:uzh-rpg/learning_on_the_fly.git
cd learning_on_the_fly

# Sync dependencies (creates .venv if needed)
uv sync

# For GPU support, install CUDA variant
uv sync --extra cuda12
```

## Quick Reference

```bash
# Show available commands
./bin/python_exec -m lotf --help

# List available configuration files
./bin/python_exec -m lotf --list-configs

# Show package version
./bin/python_exec -m lotf --version
```

## Training Commands

### 1. Residual Dynamics Training

Train an ensemble of residual dynamics neural networks to model unmodeled dynamics.

**Basic Usage:**
```bash
./bin/python_exec -m lotf residual --dataset examples/residual_dynamics/example_dataset.csv
```

**With Custom Config:**
```bash
./bin/python_exec -m lotf residual \
  --config configs/residual_dynamics.yaml \
  --dataset path/to/your_dataset.csv \
  --output checkpoints/residual_dynamics/my_model
```

**Parameters:**
- `--config`: Path to YAML configuration file (default: `configs/residual_dynamics.yaml`)
- `--dataset`: **Required**. Path to CSV dataset file
- `--output`: Path to save the trained ensemble checkpoint (default: `checkpoints/residual_dynamics/residual_params`)

**Dataset Format:**
The CSV file should contain 22 columns:
- Input (19-dim): position (3), rotation matrix (9), linear velocity (3), commands (4)
- Target (3-dim): residual acceleration

**Example Output:**
```
Loading configuration from: configs/residual_dynamics.yaml
Configuration:
  num_models: 3
  input_dim: 19
  output_dim: 3
  learning_rate: 0.01
  lambda_reg: 0.001
  num_epochs: 100
  batch_size: 256

Loading dataset from: examples/residual_dynamics/example_dataset.csv
Dataset shape: X=(1000, 19), y=(1000, 3)

Creating ensemble with 3 models...

Starting training for 100 epochs...
------------------------------------------------------------
Epoch 0/100 | Train MSE: 3.354 | Total Loss: 3.359
...
Epoch 99/100 | Train MSE: 0.201 | Total Loss: 0.206
------------------------------------------------------------
Residual model training took 0.42 seconds
Saved model params to: checkpoints/residual_dynamics/residual_params

Training complete!
```

### 2. State-Based Hovering Training

Train a neural network policy for quadrotor hovering using Backpropagation Through Time (BPTT).

**Basic Usage:**
```bash
./bin/python_exec -m lotf hover --config configs/state_hovering.yaml
```

**With Custom Output:**
```bash
./bin/python_exec -m lotf hover \
  --config configs/state_hovering.yaml \
  --output checkpoints/policy/my_hovering_policy
```

**Parameters:**
- `--config`: Path to YAML configuration file (default: `configs/state_hovering.yaml`)
- `--output`: Path to save the trained policy checkpoint (default: `checkpoints/policy/state_hovering_params`)

**Configuration Options:**
Key parameters in `configs/state_hovering.yaml`:
- `num_envs`: Number of parallel environments (default: 200)
- `max_epochs`: Maximum training epochs (default: 200)
- `sim_dt`: Simulation timestep (default: 0.02)
- `max_sim_time`: Maximum simulation time per episode (default: 3.0)
- `hover_target`: Target position [x, y, z] (default: [1.5, 0.0, 1.5])

**Note:** This training task currently requires a GPU environment due to JAX automatic differentiation requirements. CPU-only execution may encounter JVP (Jacobian-Vector Product) errors related to PRNG key handling.

### 3. Trajectory Tracking Training

Train a neural network policy for following a reference trajectory.

**Basic Usage:**
```bash
./bin/python_exec -m lotf track --config configs/traj_tracking.yaml
```

**With Trajectory Export:**
```bash
./bin/python_exec -m lotf track \
  --config configs/traj_tracking.yaml \
  --checkpoint checkpoints/policy/my_tracking_policy \
  --trajectory-output outputs/trajectory.csv
```

**Parameters:**
- `--config`: Path to YAML configuration file (default: `configs/traj_tracking.yaml`)
- `--checkpoint`: Path to save the trained policy checkpoint (default: `checkpoints/policy/traj_tracking_params`)
- `--trajectory-output`: Optional path to export trajectory as CSV file

## Loading Trained Checkpoints

All checkpoints are saved using Orbax's PyTreeCheckpointer and can be loaded as follows:

```python
from orbax.checkpoint import PyTreeCheckpointer

# Load residual dynamics model
ckptr = PyTreeCheckpointer()
residual_params = ckptr.restore("checkpoints/residual_dynamics/residual_params")

# Load policy checkpoint
policy_params = ckptr.restore("checkpoints/policy/state_hovering_params")
```

## Configuration Files

Available configuration files in `configs/`:
- `residual_dynamics.yaml` - Residual dynamics ensemble training
- `state_hovering.yaml` - State-based hovering policy
- `traj_tracking.yaml` - Trajectory tracking policy
- `vision_hovering.yaml` - Vision-based hovering policy

## Common Issues

### GPU Requirements

Some training tasks (hovering, tracking) require GPU support due to JAX's automatic differentiation requirements. If you encounter errors like:

```
TypeError: Custom JVP rule must produce primal and tangent outputs...
```

Ensure you have:
1. A CUDA-capable GPU
2. CUDA toolkit installed (12.x recommended)
3. JAX with CUDA support: `uv sync --extra cuda12`

### Checkpoint Loading Warnings

When loading checkpoints, you may see warnings about sharding info:

```
UserWarning: Sharding info not provided when restoring...
```

This is expected when loading CPU-saved checkpoints on GPU or vice versa. The checkpoints will still load correctly, but may have slightly increased restoration time.

### Virtual Environment Conflicts

If you see warnings about `VIRTUAL_ENV` not matching:

```
warning: `VIRTUAL_ENV=/path/to/other/env` does not match the project environment path `.venv`
```

This indicates another virtual environment is active. You can:
- Deactivate the other environment: `conda deactivate` or `deactivate`
- Or simply ignore - uv will use `.venv` in the project directory

## Advanced Usage

### Custom Training Configuration

Create your own configuration file by copying and modifying an existing one:

```bash
cp configs/residual_dynamics.yaml configs/my_custom_config.yaml
# Edit my_custom_config.yaml with your parameters
./bin/python_exec -m lotf residual --config configs/my_custom_config.yaml --dataset data.csv
```

### Minimal Smoke Testing

For quick verification, create a minimal config with reduced parameters:

```yaml
# minimal_test.yaml
seed: 0
num_envs: 10
max_epochs: 2
# ... other minimal parameters
```

```bash
./bin/python_exec -m lotf hover --config minimal_test.yaml --output /tmp/test_checkpoint
```

## Additional Resources

- **Examples:** See `examples/` directory for Jupyter notebooks with detailed walkthroughs
- **Pretrained Models:** Check `checkpoints/` for pretrained policy and residual dynamics models
- **Paper:** [Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation](https://arxiv.org/abs/2508.21065)

## Support

For issues and questions:
1. Check the [GitHub Issues](https://github.com/uzh-rpg/learning_on_the_fly/issues)
2. Review the example notebooks in `examples/`
3. Consult the paper for algorithmic details
