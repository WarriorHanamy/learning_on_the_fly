# Training Guide

This guide provides comprehensive documentation for training all four tasks in Learning on the Fly (LOTF): residual dynamics learning, state-based hovering, trajectory tracking, and vision-based hovering.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Task 1: Residual Dynamics Training](#task-1-residual-dynamics-training)
- [Task 2: State-Based Hovering Training](#task-2-state-based-hovering-training)
- [Task 3: Trajectory Tracking Training](#task-3-trajectory-tracking-training)
- [Task 4: Vision-Based Hovering Training](#task-4-vision-based-hovering-training)
- [Checkpoint Loading Examples](#checkpoint-loading-examples)
- [GPU Requirements and Optimization](#gpu-requirements-and-optimization)
- [Troubleshooting](#troubleshooting)

## Overview

LOTF provides training scripts for four different tasks:

| Task | Purpose | Method | GPU Required |
|------|---------|--------|--------------|
| Residual Dynamics | Learn physics residuals from real data | Supervised learning (MSE) | Optional |
| State Hovering | Train hovering policy from state observations | BPTT (reinforcement learning) | Recommended |
| Trajectory Tracking | Track predefined trajectories | BPTT (reinforcement learning) | Recommended |
| Vision Hovering | Train vision-based policy from features | Pretraining + BPTT | Required |

## Prerequisites

Before starting training, ensure you have:

1. **Installed LOTF**: Follow the [installation guide](installation.md)
2. **Prepared datasets**: For residual dynamics training, prepare CSV datasets
3. **Configured GPU**: For BPTT tasks, ensure GPU is available and properly configured
4. **Sufficient disk space**: Checkpoints can be large (100MB - 1GB each)

### Environment Variables

```bash
# Set JAX platform (GPU by default)
export JAX_PLATFORMS=cuda  # Use GPU
# export JAX_PLATFORMS=cpu  # Use CPU only

# Control GPU memory allocation
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8
```

## Task 1: Residual Dynamics Training

Residual dynamics training learns an ensemble of neural networks to predict the difference between real-world physics and simulation dynamics. This is a supervised learning task using Mean Squared Error (MSE) loss.

### Dataset Format

The dataset must be a **22-column CSV file without headers**:

- **Columns 1-19 (Input features)**: State and action features
  - Position (x, y, z)
  - Orientation (quaternion: qw, qx, qy, qz)
  - Linear velocity (vx, vy, vz)
  - Angular velocity (wx, wy, wz)
  - Rotor velocities (4 values)
  - Action inputs (4 values)

- **Columns 20-22 (Output/Residual)**: Residual prediction
  - Residual force (x, y)
  - Residual torque (z)

**Example dataset structure**:
```
-0.001385,0.000185,0.938985,0.999837,-0.003934,0.017602,0.003853,0.999982,0.004615,-0.017620,-0.004547,0.999834,-0.000517,0.000269,0.003566,7.607462,-2.520337,5.610799,-0.048515,0.971864,0.383823,0.633043
...
```

**Example dataset location**: `examples/residual_dynamics/example_dataset.csv`

### Training Commands

#### Basic Training

Train with default configuration and example dataset:

```bash
uv run python -m lotf.scripts.train_residual \
    --dataset examples/residual_dynamics/example_dataset.csv
```

#### Custom Configuration

Train with custom config file:

```bash
uv run python -m lotf.scripts.train_residual \
    --config configs/residual_dynamics.yaml \
    --dataset examples/residual_dynamics/example_dataset.csv
```

#### Custom Output Path

Save checkpoint to custom location:

```bash
uv run python -m lotf.scripts.train_residual \
    --config configs/residual_dynamics.yaml \
    --dataset examples/residual_dynamics/example_dataset.csv \
    --output checkpoints/residual_dynamics/my_model
```

### Configuration File

Edit `configs/residual_dynamics.yaml` to customize training:

```yaml
# Ensemble model parameters
num_models: 3              # Number of ensemble members
input_dim: 19              # Input feature dimension (state + action)
output_dim: 3               # Output dimension (residual prediction)

# Training hyperparameters
learning_rate: 0.01         # Optimizer learning rate
lambda_reg: 0.001          # Weight regularization coefficient
num_epochs: 100            # Number of training epochs
batch_size: 256            # Training batch size
eval_every: 10              # Log metrics every N epochs
weight_init_scale: 1.0      # Weight initialization scale
```

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | No | `configs/residual_dynamics.yaml` | Path to YAML config file |
| `--dataset` | Yes | - | Path to CSV dataset file |
| `--output` | No | `checkpoints/residual_dynamics/residual_params` | Checkpoint output path |

### Expected Output

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
  eval_every: 10

Loading dataset from: examples/residual_dynamics/example_dataset.csv
Dataset shape: X=(1000, 19), y=(1000, 3)

Creating ensemble with 3 models...

Starting training for 100 epochs...
------------------------------------------------------------
Epoch 0/100 | Train MSE: 4.4129 | Total Loss: 4.4181
Epoch 10/100 | Train MSE: 0.1166 | Total Loss: 0.1221
...
Epoch 100/100 | Train MSE: 0.0015 | Total Loss: 0.0081
------------------------------------------------------------
Residual model training took 5.57 seconds
Saved model params to: checkpoints/residual_dynamics/residual_params

Training complete!
```

### Training Time

- **CPU**: ~10-30 seconds for 1000 samples, 100 epochs
- **GPU**: ~2-10 seconds for 1000 samples, 100 epochs

## Task 2: State-Based Hovering Training

State-based hovering training uses Backpropagation Through Time (BPTT) to train a neural network policy that maintains quadrotor position at a target hover point using only state observations.

### Training Commands

#### Basic Training

Train with default configuration:

```bash
uv run python -m lotf.scripts.train_state_hovering
```

#### Custom Configuration

Train with custom config file:

```bash
uv run python -m lotf.scripts.train_state_hovering \
    --config configs/state_hovering.yaml
```

#### Custom Output Path

Save checkpoint to custom location:

```bash
uv run python -m lotf.scripts.train_state_hovering \
    --config configs/state_hovering.yaml \
    --output checkpoints/policy/my_hovering_policy
```

### Configuration File

Edit `configs/state_hovering.yaml` to customize training:

```yaml
# Random seed for reproducibility
seed: 0

# Training parameters
num_envs: 200              # Number of parallel environments
max_epochs: 200            # Number of training epochs

# Simulation parameters
sim_dt: 0.02               # Simulation time step (seconds)
max_sim_time: 3.0          # Maximum simulation time per episode (seconds)
delay: 0.04                # Action delay (seconds)

# Reward parameters
reward_sharpness: 3.0      # Sharpness parameter for reward function
action_penalty_weight: 0.5   # Weight for action penalty in reward

# Hover target position [x, y, z]
hover_target:
  - 1.5
  - 0.0
  - 1.5

# Simulation dynamics config
sim_dyn_config:
  use_high_fidelity: false   # Use high-fidelity dynamics
  use_forward_residual: false # Use residual dynamics in forward sim

# Environment noise parameters
yaw_scale: 1.0             # Yaw randomization scale
pitch_roll_scale: 0.1       # Pitch/roll randomization scale
velocity_std: 0.1           # Velocity noise std
omega_std: 0.1              # Angular velocity noise std
margin: 0.5                 # Initial position randomization margin

# Policy network architecture
policy_net:
  hidden_layers:
    - 512
    - 512
  initial_scale: 0.01       # Weight initialization scale

# Optimizer settings
optimizer:
  initial_lr: 0.005         # Initial learning rate
  scheduler: cosine_decay    # Learning rate scheduler
```

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | No | `configs/state_hovering.yaml` | Path to YAML config file |
| `--output` | No | `checkpoints/policy/state_hovering_params` | Checkpoint output path |

### BPTT Notes

**Warning**: This task uses BPTT which is computationally expensive. GPU is highly recommended.

- **Parallel environments**: 200+ parallel environments for efficient gradient estimation
- **Episode length**: 150 steps (3.0 seconds @ 0.02s/step)
- **Training time**: ~5-15 minutes on GPU, ~30-60 minutes on CPU
- **Memory usage**: ~2-4GB GPU memory for default settings

### Expected Output

```
Loading configuration from: configs/state_hovering.yaml
Initializing with seed: 0
Creating environment...
Environment info:
  action_dim: 4
  obs_dim: 27
  target hover goal: [1.5 0.  1.5]
  max_steps_in_episode: 150
Creating policy network...
Loading dummy residual dynamics parameters...
Initializing 200 parallel environments...

Starting training for 200 epochs...
--------------------------------------------------
Epoch 0/200 | Loss: 1.234 | Return: -1.234
Epoch 10/200 | Loss: 0.856 | Return: -0.856
...
Epoch 200/200 | Loss: 0.123 | Return: -0.123
--------------------------------------------------
Compile + Training time: 325.42s
Final reward: -0.12
Policy saved successfully to: checkpoints/policy/state_hovering_params

Training complete!
```

## Task 3: Trajectory Tracking Training

Trajectory tracking training uses BPTT to train a neural network policy that follows predefined reference trajectories (Circle, Figure-8, Star).

### Training Commands

#### Basic Training

Train with default configuration:

```bash
uv run python -m lotf.scripts.train_traj_tracking
```

#### Custom Configuration

Train with custom config file:

```bash
uv run python -m lotf.scripts.train_traj_tracking \
    --config configs/traj_tracking.yaml
```

#### Custom Output Path

Save checkpoint to custom location:

```bash
uv run python -m lotf.scripts.train_traj_tracking \
    --config configs/traj_tracking.yaml \
    --checkpoint checkpoints/policy/my_tracking_policy
```

### Configuration File

Edit `configs/traj_tracking.yaml` to customize training:

```yaml
# Random seed for reproducibility
seed: 0

# Training parameters
num_envs: 300              # Number of parallel environments
max_epochs: 300            # Number of training epochs

# Simulation parameters
sim_dt: 0.02               # Simulation time step (seconds)
max_sim_time: 5.0          # Maximum simulation time per episode (seconds)
delay: 0.04                # Action delay (seconds)

# Reference trajectory
ref_traj_name: fig8         # Trajectory name: circle, fig8, star
skip_start: true           # Skip initial speedup portion

# Simulation dynamics config
sim_dyn_config:
  use_high_fidelity: false   # Use high-fidelity dynamics
  use_forward_residual: false # Use residual dynamics in forward sim

# Environment noise parameters
yaw_scale: 0.1             # Yaw randomization scale
pitch_roll_scale: 0.1       # Pitch/roll randomization scale
position_std: 0.1           # Position noise std
velocity_std: 0.1           # Velocity noise std
omega_std: 0.1              # Angular velocity noise std

# Policy network architecture
policy_net:
  hidden_layers:
    - 512
    - 512
  initial_scale: 0.01       # Weight initialization scale

# Optimizer settings
optimizer:
  initial_lr: 0.001         # Initial learning rate
  scheduler: cosine_decay    # Learning rate scheduler
```

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | No | `configs/traj_tracking.yaml` | Path to YAML config file |
| `--checkpoint` | No | `checkpoints/policy/traj_tracking_params` | Checkpoint output path |
| `--trajectory-output` | No | None | Export trajectory to CSV (requires separate rollout) |

### Available Trajectories

| Trajectory | Description | Difficulty |
|-------------|-------------|------------|
| `circle` | Circular path | Easy |
| `fig8` | Figure-8 pattern | Medium |
| `star` | Five-point star | Hard |

To change trajectory, modify `ref_traj_name` in config file.

### BPTT Notes

**Warning**: This task uses BPTT with longer episodes than hovering. GPU is highly recommended.

- **Parallel environments**: 300+ parallel environments for efficient gradient estimation
- **Episode length**: 250 steps (5.0 seconds @ 0.02s/step)
- **Training time**: ~10-20 minutes on GPU, ~60-120 minutes on CPU
- **Memory usage**: ~3-6GB GPU memory for default settings

### Expected Output

```
Loading configuration from: configs/traj_tracking.yaml
Initializing with seed: 0
Creating environment...
Environment info:
  action_dim: 4
  obs_dim: 27
  ref_traj_name: fig8
  max_steps_in_episode: 250
Creating policy network...
Loading dummy residual dynamics parameters...
Initializing 300 parallel environments...

Starting training for 300 epochs...
--------------------------------------------------
Epoch 0/300 | Loss: 2.345 | Return: -2.345
Epoch 10/300 | Loss: 1.876 | Return: -1.876
...
Epoch 300/300 | Loss: 0.234 | Return: -0.234
--------------------------------------------------
Compile + Training time: 542.87s
Final reward: -0.23
Policy saved successfully to: checkpoints/policy/traj_tracking_params

Training complete!
```

## Task 4: Vision-Based Hovering Training

Vision-based hovering training is a two-stage process:
1. **Pretraining**: Train a state prediction model using collected rollout data
2. **Fine-tuning**: Fine-tune the policy using BPTT with vision features

### Stage 1: Pretraining (State Prediction)

Pretraining collects rollout data and trains a model to predict future states.

#### Pretraining Workflow

**Note**: Pretraining is typically done via Jupyter notebooks. See `examples/vision_hovering/1_pretrain_base_policy.ipynb`.

```python
import jax
from lotf.envs import HoveringFeaturesEnv, rollout
from lotf.objects import Quadrotor
from lotf.utils.math import normalize

# Setup environment
sim_dyn_config = {
    "use_high_fidelity": False,
    "use_forward_residual": False,
}
quad_obj = Quadrotor.from_name("example_quad", sim_dyn_config)

env = HoveringFeaturesEnv(
    max_steps_in_episode=int(3.0 / 0.02),
    dt=0.02,
    delay=0.04,
    yaw_scale=1.0,
    pitch_roll_scale=0.3,
    velocity_std=2.0,
    omega_std=2.0,
    quad_obj=quad_obj,
    reward_sharpness=5.0,
    action_penalty_weight=0.5,
    num_last_quad_states=15,
    skip_frames=3,
    hover_target=[1.5, 0.0, 1.5],
)

# Collect rollouts
num_rollouts = 100
rollout_keys = jax.random.split(jax.random.key(0), num_rollouts)
transitions = [rollout(env, key, None, None) for key in rollout_keys]

# Train state prediction model
# (See notebook for full implementation)
```

#### Pretraining Configuration

Edit `configs/vision_hovering.yaml` pretraining section:

```yaml
# Pretraining settings (for state prediction task)
pretrain:
  epochs: 500               # Number of pretraining epochs
  batch_size: 1024          # Batch size for pretraining
  learning_rate: 0.001      # Learning rate for pretraining
  num_rollouts: 100         # Number of rollouts to collect
  rollout_steps: 1000       # Steps per rollout
```

### Stage 2: Fine-Tuning (Vision Hovering)

Fine-tune the pretrained policy using BPTT with vision features.

**Note**: Vision hovering does not have a CLI script yet. Use the Jupyter notebook at `examples/vision_hovering/2_train_base_policy.ipynb`.

### Configuration File

Edit `configs/vision_hovering.yaml` to customize training:

```yaml
# Random seed for reproducibility
seed: 0

# Training parameters
num_envs: 300              # Number of parallel environments
max_epochs: 200            # Number of training epochs

# Simulation parameters
sim_dt: 0.02               # Simulation time step (seconds)
max_sim_time: 3.0          # Maximum simulation time per episode (seconds)
delay: 0.04                # Action delay (seconds)

# Reward parameters
reward_sharpness: 2.0      # Sharpness parameter for reward function
action_penalty_weight: 0.5   # Weight for action penalty in reward

# Hover target position [x, y, z]
hover_target:
  - 1.5
  - 0.0
  - 1.5

# Simulation dynamics config
sim_dyn_config:
  use_high_fidelity: false   # Use high-fidelity dynamics
  use_forward_residual: false # Use residual dynamics in forward sim

# Environment noise parameters
yaw_scale: 1.0             # Yaw randomization scale
pitch_roll_scale: 0.1       # Pitch/roll randomization scale
velocity_std: 0.1           # Velocity noise std
omega_std: 0.1              # Angular velocity noise std
margin: 0.5                 # Initial position randomization margin

# Feature extraction parameters
num_last_quad_states: 15     # Number of past states to include
skip_frames: 3               # Frame skipping for feature extraction

# Policy network architecture
policy_net:
  hidden_layers:
    - 512
    - 512
  initial_scale: 0.01       # Weight initialization scale

# Optimizer settings
optimizer:
  initial_lr: 0.001         # Initial learning rate
  scheduler: cosine_decay    # Learning rate scheduler
```

### BPTT Notes

**Warning**: Vision hovering uses BPTT with high-dimensional observations. GPU is required.

- **Observation dimension**: 82 features (compared to 27 for state-based)
- **Parallel environments**: 300+ parallel environments
- **Episode length**: 150 steps (3.0 seconds @ 0.02s/step)
- **Training time**: ~15-30 minutes on GPU, ~90-180 minutes on CPU
- **Memory usage**: ~4-8GB GPU memory for default settings

## Checkpoint Loading Examples

### Loading Residual Dynamics Checkpoints

```python
from lotf import LOTF_PATH
from orbax.checkpoint import PyTreeCheckpointer

# Load residual dynamics ensemble
path = LOTF_PATH + "/../checkpoints/residual_dynamics/residual_params"
ckptr = PyTreeCheckpointer()
residual_params = ckptr.restore(path)

# Use in simulation with high-fidelity dynamics
sim_dyn_config = {
    "use_high_fidelity": True,
    "use_forward_residual": True,  # Use learned residuals
}
```

### Loading Policy Checkpoints (State-Based Hovering)

```python
import jax
import optax
from flax.training.train_state import TrainState
from lotf import LOTF_PATH
from lotf.envs import HoveringStateEnv
from lotf.envs.wrappers import MinMaxObservationWrapper
from lotf.modules import MLP
from lotf.objects import Quadrotor
from orbax.checkpoint import PyTreeCheckpointer

# Create environment
sim_dyn_config = {
    "use_high_fidelity": False,
    "use_forward_residual": False,
}
quad_obj = Quadrotor.from_name("example_quad", sim_dyn_config)

eval_env = HoveringStateEnv(
    max_steps_in_episode=int(5.0 / 0.02),
    dt=0.02,
    delay=0.04,
    quad_obj=quad_obj,
    margin=0.5,
    hover_target=[1.5, 0.0, 1.5],
)
eval_env = MinMaxObservationWrapper(eval_env)

action_dim = eval_env.action_space.shape[0]
obs_dim = eval_env.observation_space.shape[0]

# Create policy network and load parameters
policy_name = "state_hovering_params"

base_policy_net = MLP(
    [obs_dim, 512, 512, action_dim],
    action_bias=eval_env.hovering_action,
)

path = LOTF_PATH + "/../checkpoints/policy/" + policy_name
ckptr = PyTreeCheckpointer()
base_policy_params = ckptr.restore(path)
loaded_train_state = TrainState.create(
    apply_fn=base_policy_net.apply,
    params=base_policy_params,
    tx=optax.adam(1e-3)
)

# Define policy function for rollout
def policy_trained(obs, key):
    return loaded_train_state.apply_fn(loaded_train_state.params, obs)
```

### Loading Policy Checkpoints (Trajectory Tracking)

```python
import jax
import optax
from flax.training.train_state import TrainState
from lotf import LOTF_PATH
from lotf.envs import TrajTrackingStateEnv
from lotf.envs.wrappers import MinMaxObservationWrapper
from lotf.modules import MLP
from lotf.objects import Quadrotor, RefTrajNames
from orbax.checkpoint import PyTreeCheckpointer

# Create environment
sim_dyn_config = {
    "use_high_fidelity": False,
    "use_forward_residual": False,
}
quad_obj = Quadrotor.from_name("example_quad", sim_dyn_config)

eval_env = TrajTrackingStateEnv(
    max_steps_in_episode=int(10.0 / 0.02),
    dt=0.02,
    delay=0.04,
    yaw_scale=0.0,
    pitch_roll_scale=0.0,
    position_std=0.0,
    velocity_std=0.0,
    omega_std=0.0,
    quad_obj=quad_obj,
    ref_traj_name=RefTrajNames.FIG8,
    skip_start=True,
)
eval_env = MinMaxObservationWrapper(eval_env)

action_dim = eval_env.action_space.shape[0]
obs_dim = eval_env.observation_space.shape[0]

# Create policy network and load parameters
policy_name = "traj_tracking_params"

base_policy_net = MLP(
    [obs_dim, 512, 512, action_dim],
    action_bias=eval_env.hovering_action,
)

path = LOTF_PATH + "/../checkpoints/policy/" + policy_name
ckptr = PyTreeCheckpointer()
base_policy_params = ckptr.restore(path)
loaded_train_state = TrainState.create(
    apply_fn=base_policy_net.apply,
    params=base_policy_params,
    tx=optax.adam(1e-3)
)

# Define policy function for rollout
def policy_trained(obs, key):
    return loaded_train_state.apply_fn(loaded_train_state.params, obs)
```

### Loading Policy Checkpoints (Vision-Based Hovering)

```python
import jax
import optax
from flax.training.train_state import TrainState
from lotf import LOTF_PATH
from lotf.envs import HoveringFeaturesEnv
from lotf.modules import MLP
from lotf.objects import Quadrotor
from orbax.checkpoint import PyTreeCheckpointer

# Create environment
sim_dyn_config = {
    "use_high_fidelity": False,
    "use_forward_residual": False,
}
quad_obj = Quadrotor.from_name("example_quad", sim_dyn_config)

eval_env = HoveringFeaturesEnv(
    max_steps_in_episode=int(10.0 / 0.02),
    dt=0.02,
    delay=0.04,
    quad_obj=quad_obj,
    num_last_quad_states=15,
    skip_frames=3,
    margin=0.5,
    hover_target=[1.5, 0.0, 1.5],
)

action_dim = eval_env.action_space.shape[0]
obs_dim = eval_env.observation_space.shape[0]

# Create policy network and load parameters
policy_name = "vision_hovering_params"

base_policy_net = MLP(
    [obs_dim, 512, 512, action_dim],
    action_bias=eval_env.hovering_action,
)

path = LOTF_PATH + "/../checkpoints/policy/" + policy_name
ckptr = PyTreeCheckpointer()
base_policy_params = ckptr.restore(path)
loaded_train_state = TrainState.create(
    apply_fn=base_policy_net.apply,
    params=base_policy_params,
    tx=optax.adam(1e-3)
)

# Define policy function for rollout
def policy_trained(obs, key):
    return loaded_train_state.apply_fn(loaded_train_state.params, obs)
```

### Running Rollouts with Loaded Checkpoints

```python
from lotf.envs import rollout

# Single rollout
key = jax.random.key(0)
transition = rollout(eval_env, key, policy_trained, dummy_residual_params)

# Multiple parallel rollouts
def get_rollouts(env, policy, num_rollouts, key):
    parallel_rollout = jax.vmap(rollout, in_axes=(None, 0, None, None))
    rollout_keys = jax.random.split(key, num_rollouts)
    transitions = parallel_rollout(env, rollout_keys, policy, dummy_residual_params)
    return transitions

transitions_eval = get_rollouts(eval_env, policy_trained, 20, jax.random.key(0))

# Plot trajectories
eval_env.plot_trajectories(transitions_eval)
```

## GPU Requirements and Optimization

### GPU Requirements

| Task | Minimum GPU | Recommended GPU | GPU Memory Required |
|------|--------------|-----------------|-------------------|
| Residual Dynamics | CPU OK | RTX 3060 | 1-2GB |
| State Hovering | GTX 1660 | RTX 3070 | 2-4GB |
| Trajectory Tracking | GTX 1660 | RTX 3070 | 3-6GB |
| Vision Hovering | RTX 3060 | RTX 3080 | 4-8GB |

### GPU Memory Optimization

If you encounter CUDA out of memory errors:

1. **Reduce number of parallel environments**:
   ```yaml
   num_envs: 100  # Reduce from 200/300
   ```

2. **Reduce memory preallocation**:
   ```bash
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_MEM_FRACTION=.6
   ```

3. **Use smaller batch sizes** (for residual dynamics):
   ```yaml
   batch_size: 128  # Reduce from 256
   ```

4. **Use mixed precision training** (advanced):
   ```python
   import jax
   jax.config.update('jax_enable_x64', False)
   ```

### Performance Optimization

**For faster training**:

1. **Use GPU**: BPTT tasks are 5-10x faster on GPU
2. **Increase parallel environments**: More environments = better gradient estimation
   ```yaml
   num_envs: 500  # Increase for faster convergence
   ```
3. **Reduce episode length**: Fewer steps per episode = faster epochs
   ```yaml
   max_sim_time: 2.0  # Reduce from 3.0/5.0
   ```
4. **Use cosine decay**: Default scheduler converges faster than fixed LR

**For better convergence**:

1. **Increase training epochs**: More epochs = better policy
   ```yaml
   max_epochs: 500  # Increase from 200/300
   ```
2. **Adjust learning rate**: Lower LR for stability, higher for speed
   ```yaml
   initial_lr: 0.001  # Lower for stability
   ```
3. **Reduce noise**: Less noise = easier learning
   ```yaml
   velocity_std: 0.05  # Reduce from 0.1
   ```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `num_envs` in config file
- Set `export XLA_PYTHON_CLIENT_MEM_FRACTION=.6`
- Use a smaller model architecture (reduce hidden layers)
- Close other GPU-intensive applications

#### 2. Training is Too Slow

**Possible causes**:
- Running on CPU instead of GPU
- Too few parallel environments
- Too many epochs

**Solutions**:
- Verify GPU is being used: `uv run python -c "import jax; print(jax.devices())"`
- Increase `num_envs` in config file
- Use fewer epochs for testing
- Use `uv run python -m lotf.scripts.train_*` with GPU backend

#### 3. Poor Policy Performance

**Possible causes**:
- Insufficient training epochs
- Wrong hyperparameters
- Simulation dynamics mismatch

**Solutions**:
- Increase `max_epochs` to 500+
- Tune learning rate: try 0.001, 0.0005, 0.0001
- Adjust reward parameters: `reward_sharpness`, `action_penalty_weight`
- Enable high-fidelity or residual dynamics in `sim_dyn_config`

#### 4. Checkpoint Loading Errors

**Error**: `FileNotFoundError` or checkpoint corruption

**Solutions**:
- Verify checkpoint path is correct
- Check if checkpoint file exists: `ls checkpoints/policy/`
- Retrain if checkpoint is corrupted
- Ensure checkpoint matches model architecture

#### 5. Dataset Format Errors (Residual Dynamics)

**Error**: `ValueError: could not convert string to float` or shape mismatches

**Solutions**:
- Verify CSV has no header row
- Check CSV has exactly 22 columns
- Ensure all values are numeric (no text)
- Verify dataset path is correct

### Getting Help

If you encounter issues not covered here:

1. Check [examples/](../examples/) for working notebook examples
2. Review [installation.md](installation.md) for environment issues
3. Search [GitHub Issues](https://github.com/WuzhongyiQian/learning_on_the_fly/issues)
4. Create a new issue with:
   - Task name and configuration
   - Error messages
   - System info (OS, GPU, JAX version)
   - Steps to reproduce

## Additional Resources

- [Installation Guide](installation.md) - Setup and environment configuration
- [CODEBASE.md](../CODEBASE.md) - Detailed codebase documentation
- [USAGE.md](../USAGE.md) - CLI usage reference
- [examples/](../examples/) - Jupyter notebook tutorials for each task
- [configs/](../configs/) - Configuration files for all tasks
