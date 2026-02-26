# Configuration Reference

Complete guide to all configuration files and parameters in Learning on the Fly (LOTF).

## Table of Contents

- [Configuration System Overview](#configuration-system-overview)
- [File Locations](#file-locations)
- [Common Parameters](#common-parameters)
- [Task-Specific Parameters](#task-specific-parameters)
  - [State-Based Hovering](#state-based-hovering-configsstate_hoveringyaml)
  - [Trajectory Tracking](#trajectory-tracking-configstraj_trackingyaml)
  - [Vision-Based Hovering](#vision-based-hovering-configsvision_hoveringyaml)
  - [Residual Dynamics](#residual-dynamics-configsresidual_dynamicsyaml)
- [Complete YAML Examples](#complete-yaml-examples)
- [Parameter Tuning Guidelines](#parameter-tuning-guidelines)

## Configuration System Overview

LOTF uses YAML configuration files to control training parameters across all four training tasks. Configuration files are loaded using Python dataclasses with validation and type checking, ensuring reproducibility and preventing runtime errors.

### Configuration Architecture

The configuration system is organized into nested dataclasses:

```
Training Config (YAML)
├── Common Parameters (seed, num_envs, max_epochs, etc.)
├── Simulation Parameters (sim_dt, max_sim_time, delay)
├── Reward Parameters (reward_sharpness, action_penalty_weight)
├── Environment Noise (yaw_scale, velocity_std, omega_std, etc.)
├── Simulation Dynamics (sim_dyn_config)
├── Policy Network (policy_net)
└── Optimizer (optimizer)
```

### Key Features

- **Type Safety**: All configs are validated against typed dataclasses
- **Nested Structure**: Related parameters are grouped in sub-configurations
- **Default Values**: Reasonable defaults are provided for all parameters
- **Runtime Overrides**: Configs can be modified via command-line arguments
- **Reproducibility**: Seed control ensures deterministic behavior

## File Locations

All training configuration files are located in the `configs/` directory at the repository root:

```
configs/
├── state_hovering.yaml      # State-based hovering policy training
├── traj_tracking.yaml       # Trajectory tracking policy training
├── vision_hovering.yaml     # Vision-based hovering policy training
└── residual_dynamics.yaml   # Residual dynamics ensemble training
```

### Config File Mappings

| Config File | Task Type | Training Script | Notebook |
|-------------|----------|-----------------|----------|
| `state_hovering.yaml` | State Hovering | `lotf/scripts/train_state_hovering.py` | `examples/state_hovering/1_train_base_policy.ipynb` |
| `traj_tracking.yaml` | Trajectory Tracking | `lotf/scripts/train_traj_tracking.py` | `examples/traj_tracking/1_train_base_policy.ipynb` |
| `vision_hovering.yaml` | Vision Hovering | (Notebook only) | `examples/vision_hovering/2_train_base_policy.ipynb` |
| `residual_dynamics.yaml` | Residual Dynamics | `lotf/scripts/train_residual.py` | `examples/residual_dynamics/train_ensemble_model.ipynb` |

## Common Parameters

These parameters appear across multiple training configurations:

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 0 | Random seed for reproducibility. Set to 0 for deterministic behavior. |
| `num_envs` | int | 200-300 | Number of parallel environments for vectorized training. Higher values improve sample efficiency but require more memory. |
| `max_epochs` | int | 100-300 | Maximum number of training epochs. Training stops early if convergence is detected. |
| `learning_rate` | float | 0.001-0.01 | Learning rate for the optimizer. Lower values (0.001) for vision tasks, higher (0.01) for residual dynamics. |

### Simulation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sim_dt` | float | 0.02 | Simulation timestep in seconds (50 Hz physics). Smaller values improve accuracy but increase computation. |
| `max_sim_time` | float | 3.0-5.0 | Maximum simulation time per episode in seconds. Longer episodes allow more complex behaviors. |
| `delay` | float | 0.04 | Action delay in seconds, simulating real-world latency. Typical value: 0.04 (40ms). |

### Reward Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reward_sharpness` | float | 2.0-3.0 | Sharpness of the exponential reward function. Higher values (3.0) create steeper gradients for faster learning. |
| `action_penalty_weight` | float | 0.5 | Weight for action regularization in reward function. Higher values encourage smoother control. |

### Simulation Dynamics Configuration

Nested under `sim_dyn_config`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_high_fidelity` | bool | false | Enable high-fidelity physics simulation. Increases realism and computational cost. |
| `use_forward_residual` | bool | false | Use learned residual dynamics in forward simulation. Requires pretrained residual model. |

### Policy Network Configuration

Nested under `policy_net`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_layers` | list[int] | [512, 512] | List of hidden layer sizes for the policy network. More layers increase capacity. |
| `initial_scale` | float | 0.01 | Scale for initial weight initialization. Smaller values (0.01) are better for pretrained policies. |

### Optimizer Configuration

Nested under `optimizer`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_lr` | float | 0.001-0.005 | Initial learning rate. Higher (0.005) for state-based tasks, lower (0.001) for vision tasks. |
| `scheduler` | str | cosine_decay | Learning rate schedule. Currently supports: `cosine_decay`. |

## Task-Specific Parameters

### State-Based Hovering (`configs/state_hovering.yaml`)

Trains a hovering policy using full state observations (position, orientation, velocity).

**Unique Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hover_target` | list[float] | [1.5, 0.0, 1.5] | Target hover position [x, y, z] in meters. |
| `margin` | float | 0.5 | Margin around hover target for initial position randomization in meters. |
| `yaw_scale` | float | 1.0 | Scale for yaw randomization (radians). Higher values increase difficulty. |
| `pitch_roll_scale` | float | 0.1 | Scale for pitch/roll randomization (radians). |
| `velocity_std` | float | 0.1 | Standard deviation for velocity randomization (m/s). |
| `omega_std` | float | 0.1 | Standard deviation for angular velocity randomization (rad/s). |

**Complete Parameter List:**

```yaml
seed: int                    # 0
num_envs: int                # 200
max_epochs: int              # 200
sim_dt: float                # 0.02
max_sim_time: float          # 3.0
delay: float                 # 0.04
reward_sharpness: float      # 3.0
action_penalty_weight: float # 0.5
hover_target: list[float]    # [1.5, 0.0, 1.5]
sim_dyn_config:              # Nested config
  use_high_fidelity: bool    # false
  use_forward_residual: bool # false
yaw_scale: float             # 1.0
pitch_roll_scale: float      # 0.1
velocity_std: float          # 0.1
omega_std: float             # 0.1
margin: float                # 0.5
policy_net:                  # Nested config
  hidden_layers: list[int]   # [512, 512]
  initial_scale: float       # 0.01
optimizer:                   # Nested config
  initial_lr: float          # 0.005
  scheduler: str             # cosine_decay
```

**Default Configuration:**

```yaml
seed: 0
num_envs: 200
max_epochs: 200
sim_dt: 0.02
max_sim_time: 3.0
delay: 0.04
reward_sharpness: 3.0
action_penalty_weight: 0.5
hover_target:
  - 1.5
  - 0.0
  - 1.5
sim_dyn_config:
  use_high_fidelity: false
  use_forward_residual: false
yaw_scale: 1.0
pitch_roll_scale: 0.1
velocity_std: 0.1
omega_std: 0.1
margin: 0.5
policy_net:
  hidden_layers:
    - 512
    - 512
  initial_scale: 0.01
optimizer:
  initial_lr: 0.005
  scheduler: cosine_decay
```

### Trajectory Tracking (`configs/traj_tracking.yaml`)

Trains a policy to track predefined reference trajectories (figure-8, circle, star).

**Unique Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ref_traj_name` | str | "fig8" | Reference trajectory name. Options: "fig8", "circle", "star". |
| `skip_start` | bool | true | Skip initial speedup portion of trajectory. Set to true for smoother tracking. |
| `position_std` | float | 0.1 | Standard deviation for position noise in meters. |
| `yaw_scale` | float | 0.1 | Scale for yaw randomization (lower than hovering for tracking stability). |

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
sim_dyn_config:              # Nested config
  use_high_fidelity: bool    # false
  use_forward_residual: bool # false
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
sim_dyn_config:
  use_high_fidelity: false
  use_forward_residual: false
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

### Vision-Based Hovering (`configs/vision_hovering.yaml`)

Trains a hovering policy using visual features extracted from camera observations. Requires pretraining phase.

**Unique Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_last_quad_states` | int | 15 | Number of recent quadrotor states to include in observations. |
| `skip_frames` | int | 3 | Number of frames to skip between observations (reduces temporal correlation). |

**Pretraining Configuration**

Nested under `pretrain`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 500 | Number of pretraining epochs for state prediction task. |
| `batch_size` | int | 1024 | Batch size for pretraining. |
| `learning_rate` | float | 0.001 | Learning rate for pretraining. |
| `num_rollouts` | int | 100 | Number of rollouts to collect for pretraining dataset. |
| `rollout_steps` | int | 1000 | Steps per rollout for data collection. |

**Complete Parameter List:**

```yaml
seed: int                    # 0
num_envs: int                # 300
max_epochs: int              # 200
sim_dt: float                # 0.02
max_sim_time: float          # 3.0
delay: float                 # 0.04
reward_sharpness: float      # 2.0
action_penalty_weight: float # 0.5
hover_target: list[float]    # [1.5, 0.0, 1.5]
sim_dyn_config:              # Nested config
  use_high_fidelity: bool    # false
  use_forward_residual: bool # false
yaw_scale: float             # 1.0
pitch_roll_scale: float      # 0.1
velocity_std: float          # 0.1
omega_std: float             # 0.1
margin: float                # 0.5
num_last_quad_states: int    # 15
skip_frames: int             # 3
policy_net:                  # Nested config
  hidden_layers: list[int]   # [512, 512]
  initial_scale: float       # 0.01
optimizer:                   # Nested config
  initial_lr: float          # 0.001
  scheduler: str             # cosine_decay
pretrain:                    # Nested config
  epochs: int                # 500
  batch_size: int            # 1024
  learning_rate: float       # 0.001
  num_rollouts: int          # 100
  rollout_steps: int         # 1000
```

**Default Configuration:**

```yaml
seed: 0
num_envs: 300
max_epochs: 200
sim_dt: 0.02
max_sim_time: 3.0
delay: 0.04
reward_sharpness: 2.0
action_penalty_weight: 0.5
hover_target:
  - 1.5
  - 0.0
  - 1.5
sim_dyn_config:
  use_high_fidelity: false
  use_forward_residual: false
yaw_scale: 1.0
pitch_roll_scale: 0.1
velocity_std: 0.1
omega_std: 0.1
margin: 0.5
num_last_quad_states: 15
skip_frames: 3
policy_net:
  hidden_layers:
    - 512
    - 512
  initial_scale: 0.01
optimizer:
  initial_lr: 0.001
  scheduler: cosine_decay
pretrain:
  epochs: 500
  batch_size: 1024
  learning_rate: 0.001
  num_rollouts: 100
  rollout_steps: 1000
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

### State-Based Hovering (`configs/state_hovering.yaml`)

```yaml
# State-Based Hovering Training Configuration
# Extracted from examples/state_hovering/1_train_base_policy.ipynb

# Random seed for reproducibility
seed: 0

# Training parameters
num_envs: 200
max_epochs: 200

# Simulation parameters
sim_dt: 0.02
max_sim_time: 3.0
delay: 0.04

# Reward parameters
reward_sharpness: 3.0
action_penalty_weight: 0.5

# Hover target position [x, y, z]
hover_target:
  - 1.5
  - 0.0
  - 1.5

# Simulation dynamics config
sim_dyn_config:
  use_high_fidelity: false
  use_forward_residual: false

# Environment noise parameters
yaw_scale: 1.0
pitch_roll_scale: 0.1
velocity_std: 0.1
omega_std: 0.1
margin: 0.5

# Policy network architecture
policy_net:
  hidden_layers:
    - 512
    - 512
  initial_scale: 0.01

# Optimizer settings
optimizer:
  initial_lr: 0.005
  scheduler: cosine_decay
```

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

# Simulation dynamics config
sim_dyn_config:
  use_high_fidelity: false
  use_forward_residual: false

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

### Vision-Based Hovering (`configs/vision_hovering.yaml`)

```yaml
# Vision-Based Hovering Training Configuration
# Extracted from examples/vision_hovering/2_train_base_policy.ipynb

# Random seed for reproducibility
seed: 0

# Training parameters
num_envs: 300
max_epochs: 200

# Simulation parameters
sim_dt: 0.02
max_sim_time: 3.0
delay: 0.04

# Reward parameters
reward_sharpness: 2.0
action_penalty_weight: 0.5

# Hover target position [x, y, z]
hover_target:
  - 1.5
  - 0.0
  - 1.5

# Simulation dynamics config
sim_dyn_config:
  use_high_fidelity: false
  use_forward_residual: false

# Environment noise parameters
yaw_scale: 1.0
pitch_roll_scale: 0.1
velocity_std: 0.1
omega_std: 0.1
margin: 0.5

# Feature extraction parameters
num_last_quad_states: 15
skip_frames: 3

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

# Pretraining settings (for state prediction task)
pretrain:
  epochs: 500
  batch_size: 1024
  learning_rate: 0.001
  num_rollouts: 100
  rollout_steps: 1000
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
| State Hovering | 100-300 | Default 200 is good balance |
| Trajectory Tracking | 200-500 | Default 300 for stable tracking |
| Vision Hovering | 200-400 | Default 300, reduce if OOM |
| Residual Dynamics | N/A | Controlled by `batch_size` instead |

**Tuning Guidelines:**
- Start with default values and monitor GPU memory usage
- If you run out of memory: Decrease `num_envs` by 50-100
- If training is too slow: Increase `num_envs` up to available memory limit
- For debugging: Use very low values (10-20) for faster iterations

**Memory Estimation:**
```
GPU Memory (GB) ≈ num_envs * 0.01 GB (rough estimate for state tasks)
GPU Memory (GB) ≈ num_envs * 0.02 GB (for vision tasks with larger observations)
```

### 2. `max_epochs` - Maximum Training Epochs

**Purpose:** Controls how long training runs before stopping.

**Trade-offs:**
- **Higher values:** May achieve better performance, but wastes time if converged early
- **Lower values:** Faster training, but may underfit if insufficient

**Recommendations by Task:**

| Task | Recommended Range | Notes |
|------|-------------------|-------|
| State Hovering | 100-300 | Default 200 typically sufficient |
| Trajectory Tracking | 200-500 | Default 300 for complex trajectories |
| Vision Hovering | 150-300 | Default 200 after pretraining |
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
| State Hovering | 0.001-0.01 | 0.005 | Moderate learning rate works well |
| Trajectory Tracking | 0.0005-0.002 | 0.001 | Lower for stable tracking |
| Vision Hovering | 0.0005-0.002 | 0.001 | Requires lower rate due to complexity |
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
| State Hovering | [256, 256] or [512, 512] | [512, 512] | 2 layers sufficient |
| Trajectory Tracking | [256, 256] or [512, 512] | [512, 512] | Similar to hovering |
| Vision Hovering | [512, 512] or [512, 512, 256] | [512, 512] | May need more capacity |
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

#### State-Based Hovering

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

#### Vision-Based Hovering

**Number of States (`num_last_quad_states`):**
- Default: 15
- Range: 5-30
- More states provide more temporal context but increase observation size

**Frame Skipping (`skip_frames`):**
- Default: 3
- Range: 1-5
- Higher skipping reduces temporal correlation but may miss fast dynamics

**Pretraining:**
- Always complete pretraining phase (500 epochs)
- Use lower learning rate (0.001) for pretraining
- Increase `num_rollouts` for more diverse pretraining data

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
  - Verify target is reachable (check `hover_target`)

**Issue: Overfitting**
- Solutions:
  - Increase `lambda_reg` (residual dynamics)
  - Reduce network size
  - Increase noise parameters
  - Use ensemble (residual dynamics)

## Additional Resources

- [Training Guide](training.md) - Complete training workflows and examples
- [Installation Guide](installation.md) - Setup and environment configuration
- [Codebase Documentation](../CODEBASE.md) - API reference and architecture

## Configuration Best Practices

1. **Always use version control** for your config files
2. **Document changes** with comments in YAML files
3. **Save successful configs** with descriptive names (e.g., `state_hovering_best.yaml`)
4. **Use config inheritance** by creating base configs and task-specific overrides
5. **Validate configs** by running a short test before full training
6. **Monitor GPU memory** usage when scaling up `num_envs`
7. **Use consistent naming** across config files for easy comparison
8. **Keep configs simple** - only override necessary parameters
9. **Test configs** on small scale before full training
10. **Backup checkpoints** with corresponding configs for reproducibility
