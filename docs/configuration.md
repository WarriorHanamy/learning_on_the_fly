# Configuration Reference

Complete guide to all configuration files and parameters in Learning on the Fly (LOTF).

## Table of Contents

- [Configuration System Overview](#configuration-system-overview)
- [File Locations](#file-locations)
- [Common Parameters](#common-parameters)
- [Task-Specific Parameters](#task-specific-parameters)
  - [Trajectory Tracking (`configs/traj_tracking.yaml`)](#trajectory-tracking-configstraj_trackingyaml)
### Trajectory Tracking (`configs/traj_tracking.yaml`)

Trains a policy to track predefined reference trajectories (figure-8, circle, star).

**Unique Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ref_traj_name` | str | "fig8" | Reference trajectory name. Options: "fig8", "circle", "star". |
| `skip_start` | bool | true | Skip initial speedup portion of trajectory. Set to true for smoother tracking. |
| `position_std` | float | 0.1 | Standard deviation for position noise in meters. |
| `yaw_scale` | float | 0.1 | Scale for yaw randomization. |

**Complete Parameter List:**

```yaml
seed: int                    # 0
num_envs: int                # 300
max_epochs: int              # 300
sim_dt: float                # 0.02
max_sim_time: float          # 5.0
delay: float                 # 0.04
ref_traj_name: str           # "fig8"
skip_start: bool             # true
forward_model_config:              # Nested config
  enable_inner_loop_dynamics: bool    # false
  enable_residual_acceleration: bool # false
yaw_scale: float             # 0.1
pitch_roll_scale: float      # 0.1
position_std: float          # 0.1
velocity_std: float          # 0.1
omega_std: float             # 0.1
policy_net:                  # Nested config
  hidden_layers: list[int]   # [512, 512]
  initial_scale: float       # 0.01
optimizer:                   # Nested config
  initial_lr: float          # 0.001
  scheduler: str             # cosine_decay
```

**Default Configuration:**

```yaml
seed: 0
num_envs: 300
max_epochs: 300
sim_dt: 0.02
max_sim_time: 5.0
delay: 0.04
ref_traj_name: fig8
skip_start: true
forward_model_config:
  enable_inner_loop_dynamics: false
  enable_residual_acceleration: false
yaw_scale: 0.1
pitch_roll_scale: 0.1
position_std: 0.1
velocity_std: 0.1
omega_std: 0.1
policy_net:
  hidden_layers:
    - 512
    - 512
  initial_scale: 0.01
optimizer:
  initial_lr: 0.001
  scheduler: cosine_decay
```

### Residual Dynamics (`configs/residual_dynamics.yaml`)

Trains an ensemble of neural networks to learn physics residuals from real-world data.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_models` | int | 3 | Number of ensemble members. More models improve uncertainty estimation. |
| `input_dim` | int | 19 | Input dimension (state + action features). Must match dataset. |
| `output_dim` | int | 3 | Output dimension (residual prediction vector). |
| `learning_rate` | float | 0.01 | Learning rate for residual dynamics training. |
| `lambda_reg` | float | 0.001 | L2 regularization coefficient. Higher values prevent overfitting. |
| `num_epochs` | int | 100 | Number of training epochs. |
| `batch_size` | int | 256 | Batch size for training. Larger values improve stability. |
| `eval_every` | int | 10 | Log evaluation metrics every N epochs. |
| `weight_init_scale` | float | 1.0 | Scale for weight initialization. |
| `dataset_name` | str | "example_dataset.csv" | Path to CSV dataset file. |

**Complete Parameter List:**

```yaml
num_models: int              # 3
input_dim: int                # 19
output_dim: int               # 3
learning_rate: float          # 0.01
lambda_reg: float             # 0.001
num_epochs: int               # 100
batch_size: int               # 256
eval_every: int               # 10
weight_init_scale: float      # 1.0
dataset_name: str            # "example_dataset.csv"
```

**Default Configuration:**

```yaml
num_models: 3
input_dim: 19
output_dim: 3
learning_rate: 0.01
lambda_reg: 0.001
num_epochs: 100
batch_size: 256
eval_every: 10
weight_init_scale: 1.0
dataset_name: example_dataset.csv
```

## Complete YAML Examples

### Trajectory Tracking (`configs/traj_tracking.yaml`)

```yaml
# Trajectory Tracking Training Configuration
# Extracted from examples/traj_tracking/1_train_base_policy.ipynb

# Random seed for reproducibility
seed: 0

# Training parameters
num_envs: 300
max_epochs: 300

# Simulation parameters
sim_dt: 0.02
max_sim_time: 5.0
delay: 0.04

# Reference trajectory
ref_traj_name: fig8
skip_start: true

# Forward model config
forward_model_config:
  enable_inner_loop_dynamics: false
  enable_residual_acceleration: false

# Environment noise parameters
yaw_scale: 0.1
pitch_roll_scale: 0.1
position_std: 0.1
velocity_std: 0.1
omega_std: 0.1

# Policy network architecture
policy_net:
  hidden_layers:
    - 512
    - 512
  initial_scale: 0.01

# Optimizer settings
optimizer:
  initial_lr: 0.001
  scheduler: cosine_decay
```

### Residual Dynamics (`configs/residual_dynamics.yaml`)

```yaml
# Residual Dynamics Ensemble Model Training Configuration
# Extracted from examples/residual_dynamics/train_ensemble_model.ipynb

# Ensemble model parameters
num_models: 3
input_dim: 19
output_dim: 3

# Training hyperparameters
learning_rate: 0.01
lambda_reg: 0.001
num_epochs: 100
batch_size: 256
eval_every: 10

# Weight initialization
weight_init_scale: 1.0

# Dataset settings
dataset_name: example_dataset.csv
```

## Parameter Tuning Guidelines

This section provides guidance on tuning the most important parameters for different scenarios.

### 1. `num_envs` - Number of Parallel Environments

**Purpose:** Controls the batch size for training by running multiple environments in parallel.

**Trade-offs:**
- **Higher values (500-1000):** Faster convergence due to more samples per update, but requires more GPU memory
- **Lower values (50-100):** Slower convergence, but uses less memory and is more stable

**Recommendations by Task:**

| Task | Recommended Range | Notes |
|------|-------------------|-------|

| Trajectory Tracking | 200-500 | Default 300 for stable tracking |
| Residual Dynamics | N/A | Controlled by `batch_size` instead |

**Tuning Guidelines:**
- Start with default values and monitor GPU memory usage
- If you run out of memory: Decrease `num_envs` by 50-100
- If training is too slow: Increase `num_envs` up to available memory limit
- For debugging: Use very low values (10-20) for faster iterations

**Memory Estimation:**
```
GPU Memory (GB) ≈ num_envs * 0.01 GB (rough estimate for state tasks)
```

### 2. `max_epochs` - Maximum Training Epochs

**Purpose:** Controls how long training runs before stopping.

**Trade-offs:**
- **Higher values:** May achieve better performance, but wastes time if converged early
- **Lower values:** Faster training, but may underfit if insufficient

**Recommendations by Task:**

| Task | Recommended Range | Notes |
|------|-------------------|-------|

| Trajectory Tracking | 200-500 | Default 300 for complex trajectories |
| Residual Dynamics | 50-200 | Default 100 for ensemble |

**Tuning Guidelines:**
- Monitor training loss/reward curves
- Stop early if metrics plateau for 20-30 epochs
- For quick experimentation: Reduce by 50-100 epochs
- For production models: Increase by 50-100 epochs

**Early Stopping:**
The training scripts automatically detect convergence and may stop before `max_epochs`. Key indicators:
- Loss plateaus (change < 1% over 10 epochs)
- Reward stabilizes (change < 0.01 over 10 epochs)

### 3. `learning_rate` / `initial_lr` - Learning Rate

**Purpose:** Controls step size for optimizer updates.

**Trade-offs:**
- **Higher values:** Faster initial learning, but may be unstable or diverge
- **Lower values:** More stable convergence, but slower training

**Recommendations by Task:**

| Task | Recommended Range | Default | Notes |
|------|-------------------|---------|-------|

| Trajectory Tracking | 0.0005-0.002 | 0.001 | Lower for stable tracking |
| Residual Dynamics | 0.005-0.02 | 0.01 | Supervised learning tolerates higher rates |

**Tuning Guidelines:**

**If training is unstable (loss spikes or NaN):**
- Decrease learning rate by factor of 2-5
- Example: 0.005 → 0.001

**If training is too slow (loss decreases gradually):**
- Increase learning rate by factor of 2
- Example: 0.001 → 0.002

**Common Patterns:**
1. **Start High:** Use default learning rate
2. **Monitor:** Watch loss curve for first 10-20 epochs
3. **Adjust:** Decrease if unstable, increase if too slow
4. **Fine-tune:** Once stable, adjust by 10-20% for optimal convergence

**Learning Rate Schedules:**

All tasks use `cosine_decay` scheduler, which gradually reduces learning rate:

```python
learning_rate(t) = initial_lr * 0.5 * (1 + cos(pi * t / max_epochs))
```

This provides:
- High learning rate at start for fast exploration
- Gradual decay for fine-tuning
- Near-zero rate at end for convergence

### 4. Network Architecture - `policy_net.hidden_layers`

**Purpose:** Defines the capacity of the policy network (hidden layer sizes).

**Trade-offs:**
- **Larger networks:** Can learn more complex policies, but require more data and memory
- **Smaller networks:** Faster training and less memory, but may underfit complex tasks

**Recommendations by Task:**

| Task | Recommended Architecture | Default | Notes |
|------|-------------------------|---------|-------|

| Trajectory Tracking | [256, 256] or [512, 512] | [512, 512] | 2 layers sufficient |
| Residual Dynamics | N/A | N/A | Fixed architecture in code |

**Common Architectures:**

```yaml
# Small (faster training, less memory)
hidden_layers:
  - 128
  - 128

# Medium (default)
hidden_layers:
  - 256
  - 256

# Large (default for most tasks)
hidden_layers:
  - 512
  - 512

# Very Large (for complex tasks)
hidden_layers:
  - 512
  - 512
  - 256
```

**Tuning Guidelines:**

**If policy underperforms (low reward, fails to learn):**
- Increase network size (e.g., [256, 256] → [512, 512])
- Add more layers (e.g., [512, 512] → [512, 512, 256])

**If training is too slow or memory is limited:**
- Decrease network size (e.g., [512, 512] → [256, 256])
- Reduce layer count (e.g., [512, 512, 256] → [512, 512])

**Rule of Thumb:**
- Start with [512, 512] (default)
- Only increase if you see clear evidence of underfitting
- Decrease if memory is constrained or training is too slow

**Network Size vs Training Data:**

The relationship between network size and training data (controlled by `num_envs`):

| Network Size | Min `num_envs` | Recommended `num_envs` |
|--------------|----------------|------------------------|
| [128, 128]   | 50             | 100-200                |
| [256, 256]   | 100            | 200-400                |
| [512, 512]   | 200            | 300-600                |
| [512, 512, 256] | 300        | 400-800                |

### 5. Task-Specific Tuning Tips

#### Trajectory Tracking

**Reward Sharpness (`reward_sharpness`):**
- Default: 3.0
- Range: 1.0-5.0
- Higher values create steeper gradients for faster learning
- Lower values provide smoother but slower learning

**Action Penalty Weight (`action_penalty_weight`):**
- Default: 0.5
- Range: 0.1-1.0
- Higher values encourage smoother control but may reduce agility
- Lower values allow more aggressive control

#### Trajectory Tracking

**Reference Trajectory (`ref_traj_name`):**
- Options: "fig8", "circle", "star"
- Complexity: star > fig8 > circle
- Start with "circle" for debugging

**Position Noise (`position_std`):**
- Default: 0.1
- Range: 0.0-0.5
- Higher values increase robustness but difficulty

#### Residual Dynamics

**Ensemble Size (`num_models`):**
- Default: 3
- Range: 1-10
- More models improve uncertainty estimation but increase training time
- For final deployment: Use 5-10 models

**Regularization (`lambda_reg`):**
- Default: 0.001
- Range: 0.0001-0.01
- Higher values prevent overfitting but may underfit
- Use cross-validation to tune

### 6. Common Tuning Workflows

**Workflow 1: Quick Experimentation**
1. Reduce `num_envs` to 50-100 for faster iterations
2. Reduce `max_epochs` to 50-100
3. Use small network: [256, 256]
4. Once satisfied, scale up to defaults

**Workflow 2: Maximum Performance**
1. Increase `num_envs` to maximum memory limit
2. Increase `max_epochs` to 300-500
3. Use large network: [512, 512, 256]
4. Fine-tune learning rate ±20%

**Workflow 3: Memory-Constrained Training**
1. Reduce `num_envs` until memory fits
2. Reduce network size: [256, 256] or [128, 128]
3. Reduce `max_epochs` if training is too slow
4. Consider CPU-only training if GPU is unavailable

**Workflow 4: Robustness Testing**
1. Increase noise parameters (yaw_scale, velocity_std, etc.)
2. Increase `action_penalty_weight` for smoother control
3. Use smaller learning rate for stable convergence
4. Train with multiple random seeds

### 7. Troubleshooting Common Issues

**Issue: Out of Memory (OOM)**
- Solutions:
  - Reduce `num_envs` by 50-100
  - Reduce `policy_net.hidden_layers` size
  - Reduce `batch_size` (for residual dynamics)
  - Reduce `max_sim_time` for shorter episodes

**Issue: Training Diverges (loss goes to NaN)**
- Solutions:
  - Decrease learning rate by factor of 2-5
  - Check for NaN in dataset (residual dynamics)
  - Reduce `reward_sharpness`
  - Reduce network size

**Issue: Slow Convergence**
- Solutions:
  - Increase learning rate by factor of 2
  - Increase `num_envs` for more samples per update
  - Increase `reward_sharpness` for steeper gradients
  - Check if task is too difficult for current architecture

**Issue: Policy Not Learning (reward stays low)**
- Solutions:
  - Increase network capacity (larger `hidden_layers`)
  - Reduce `action_penalty_weight`
  - Check simulation parameters are reasonable
  - Verify trajectory configuration is valid

**Issue: Overfitting**
- Solutions:
  - Increase `lambda_reg` (residual dynamics)
  - Reduce network size
  - Increase noise parameters
  - Use ensemble (residual dynamics)

## Configuration Validation

Before starting full training, it's recommended to validate your configuration files to ensure they are correctly formatted and compatible with the training scripts. All validation commands use `uv run`, which automatically manages the virtual environment.

### Configuration Validation

### Validating Configuration Syntax

Check that your YAML files are syntactically correct:

```bash
# Validate trajectory tracking configuration
uv run python -c "
import yaml
with open('configs/traj_tracking.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print('Configuration loaded successfully!')
    print(f'Number of environments: {config.get(\"num_envs\")}')
    print(f'Max epochs: {config.get(\"max_epochs\")}')
"
```

```bash
# Validate trajectory tracking configuration
uv run python -c "
import yaml
with open('configs/traj_tracking.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print('Configuration loaded successfully!')
    print(f'Reference trajectory: {config.get(\"ref_traj_name\")}')
    print(f'Number of environments: {config.get(\"num_envs\")}')
"
```

```bash
# Validate residual dynamics configuration
uv run python -c "
import yaml
with open('configs/residual_dynamics.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print('Configuration loaded successfully!')
    print(f'Number of models: {config.get(\"num_models\")}')
    print(f'Input dimension: {config.get(\"input_dim\")}')
"
```

### Testing Configurations with Short Runs

Run a short test training to verify configuration parameters work correctly:

```bash
# Test trajectory tracking config with reduced epochs
uv run python -c "
import yaml
with open('configs/traj_tracking.yaml', 'r') as f:
    config = yaml.safe_load(f)
    # Reduce epochs for quick test
    config['max_epochs'] = 2
    config['num_envs'] = 10
with open('configs/configs/traj_tracking_test.yaml', 'w') as f:
    yaml.dump(config, f)
print('Test configuration created: configs/configs/traj_tracking_test.yaml')
"
```

```bash
# Run quick test with modified config
uv run train track --config configs/configs/traj_tracking_test.yaml
```

### Checking Configuration Parameter Types

Verify that all configuration parameters have the correct data types:

```bash
# Validate parameter types for trajectory tracking
uv run python -c "
import yaml
from typing import get_type_hints

expected_types = {
    'seed': int,
    'num_envs': int,
    'max_epochs': int,
    'sim_dt': float,
    'max_sim_time': float,
    'delay': float,
}

with open('configs/traj_tracking.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('Validating parameter types:')
for param, expected_type in expected_types.items():
    if param in config:
        actual_type = type(config[param])
        match = actual_type == expected_type
        status = '✓' if match else '✗'
        print(f'{status} {param}: expected {expected_type.__name__}, got {actual_type.__name__}')
    else:
        print(f'⚠ {param}: not found in config')
"
```

### Verifying Configuration Parameters Against Training Script

Check that all required parameters are present in the configuration:

```bash
# Verify residual dynamics configuration
uv run python -c "
import yaml

required_params = [
    'num_models',
    'input_dim',
    'output_dim',
    'learning_rate',
    'lambda_reg',
    'num_epochs',
    'batch_size',
    'eval_every',
    'weight_init_scale',
    'dataset_name',
]

with open('configs/residual_dynamics.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('Checking required parameters:')
all_present = True
for param in required_params:
    present = param in config
    status = '✓' if present else '✗'
    print(f'{status} {param}')
    if not present:
        all_present = False

if all_present:
    print('\\nAll required parameters are present!')
else:
    print('\\nWarning: Some required parameters are missing!')
"
```

### Quick Configuration Sanity Check

Perform a quick sanity check on configuration values:

```bash
# Sanity check for trajectory tracking config
uv run python -c "
import yaml

with open('configs/traj_tracking.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('Configuration sanity check:')
print(f'num_envs ({config[\"num_envs\"]}): ', end='')
if 1 <= config['num_envs'] <= 1000:
    print('✓ Valid range [1, 1000]')
else:
    print('✗ Out of range [1, 1000]')

print(f'max_epochs ({config[\"max_epochs\"]}): ', end='')
if 1 <= config['max_epochs'] <= 10000:
    print('✓ Valid range [1, 10000]')
else:
    print('✗ Out of range [1, 10000]')

print(f'sim_dt ({config[\"sim_dt\"]}): ', end='')
if 0.001 <= config['sim_dt'] <= 0.1:
    print('✓ Valid range [0.001, 0.1]')
else:
    print('✗ Out of range [0.001, 0.1]')

print(f'delay ({config[\"delay\"]}): ', end='')
if 0.0 <= config['delay'] <= 1.0:
    print('✓ Valid range [0.0, 1.0]')
else:
    print('✗ Out of range [0.0, 1.0]')

print(f'reward_sharpness ({config[\"reward_sharpness\"]}): ', end='')
if 0.1 <= config['reward_sharpness'] <= 10.0:
    print('✓ Valid range [0.1, 10.0]')
else:
    print('✗ Out of range [0.1, 10.0]')
"
```

### Comparing Configuration Files

Compare two configuration files to identify differences:

```bash
# Compare default and custom configurations
uv run python -c "
import yaml

with open('configs/traj_tracking.yaml', 'r') as f:
    default_config = yaml.safe_load(f)

with open('configs/traj_tracking_custom.yaml', 'r') as f:
    custom_config = yaml.safe_load(f)

print('Comparing configurations:')
print('\\nParameters changed in custom config:')
for key in default_config:
    if key in custom_config:
        if default_config[key] != custom_config[key]:
            print(f'  {key}: {default_config[key]} -> {custom_config[key]}')
    else:
        print(f'  {key}: removed in custom config')

print('\\nNew parameters in custom config:')
for key in custom_config:
    if key not in default_config:
        print(f'  {key}: {custom_config[key]}')
"
```

### Validation Before Training

Always validate your configuration before starting a full training run:

```bash
# Complete validation workflow
uv run python -c "
import yaml
import sys

def validate_config(config_path):
    print(f'Validating {config_path}...')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print('✓ YAML syntax is valid')
        print(f'✓ Configuration loaded: {len(config)} top-level parameters')
        return True
    except yaml.YAMLError as e:
        print(f'✗ YAML syntax error: {e}')
        return False
    except FileNotFoundError:
        print(f'✗ Configuration file not found: {config_path}')
        return False

configs = [
    'configs/traj_tracking.yaml',
    'configs/traj_tracking.yaml',
    'configs/residual_dynamics.yaml',
]

print('=' * 60)
all_valid = True
for config in configs:
    if not validate_config(config):
        all_valid = False
    print()

print('=' * 60)
if all_valid:
    print('All configurations validated successfully!')
    sys.exit(0)
else:
    print('Some configurations failed validation. Please check errors above.')
    sys.exit(1)
"
```

## Additional Resources

- [Training Guide](training.md) - Complete training workflows and examples
- [Installation Guide](installation.md) - Setup and environment configuration
- [Codebase Documentation](../CODEBASE.md) - API reference and architecture

## Configuration Best Practices

1. **Always use version control** for your config files
2. **Document changes** with comments in YAML files
3. **Save successful configs** with descriptive names (e.g., `traj_tracking_best.yaml`)
4. **Use config inheritance** by creating base configs and task-specific overrides
5. **Validate configs** using `uv run python` before full training (see [Configuration Validation](#configuration-validation))
6. **Monitor GPU memory** usage when scaling up `num_envs`
7. **Use consistent naming** across config files for easy comparison
8. **Keep configs simple** - only override necessary parameters
9. **Test configs** on small scale before full training
10. **Backup checkpoints** with corresponding configs for reproducibility

### Configuration Management

```bash
# Quick config validation
uv run python -c "import yaml; yaml.safe_load(open('configs/traj_tracking.yaml')); print('Config valid!')"

# Test config with reduced parameters
uv run train track --config configs/traj_tracking.yaml --output checkpoints/test

# Compare configs
uv run python -c "import yaml; print(yaml.safe_load(open('configs/traj_tracking.yaml')))"
```

See [Configuration Validation](#configuration-validation) for detailed examples.
