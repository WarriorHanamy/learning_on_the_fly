## Requirements

| Component | Version |
|-----------|---------|
| Ubuntu    | 22.04 LTS |
| Python    | 3.10 (uv managed) |
| CUDA      | 12.x |
| GPU       | NVIDIA GPU with CUDA support |

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast dependency management.

```bash
# Clone the repository
git clone https://github.com/WarriorHanamy/learning_on_the_fly.git
cd learning_on_the_fly

# Install dependencies (CPU only)
uv sync

# Install with GPU support (CUDA 12)
uv sync --extra cuda12
```

## Quick Start

```bash
# Show available commands
uv run lotf --help

# List available configuration files
uv run lotf --list-configs

# Show version
uv run lotf --version
```

## Training

### 1. Residual Dynamics Training

Train an ensemble of neural networks to model unmodeled dynamics:

```bash
# Basic usage
uv run lotf residual --dataset examples/residual_dynamics/example_dataset.csv

# With custom config and output
uv run lotf residual \
  --config configs/residual_dynamics.yaml \
  --dataset path/to/your_dataset.csv \
  --output checkpoints/residual_dynamics/my_model
```

**Dataset Format (CSV, 22 columns):**

| Input (19-dim) | Target (3-dim) |
|----------------|----------------|
| position (3) + rotation matrix (9) + linear vel (3) + commands (4) | residual accel (3) |

### 2. State-Based Hovering Training

Train a hovering policy using Backpropagation Through Time (BPTT):

```bash
# Basic usage
uv run lotf hover --config configs/state_hovering.yaml

# With custom output path
uv run lotf hover \
  --config configs/state_hovering.yaml \
  --output checkpoints/policy/my_hovering_policy
```

**Key Configuration Options** (`configs/state_hovering.yaml`):
- `num_envs`: Parallel environments (default: 200)
- `max_epochs`: Training epochs (default: 200)
- `hover_target`: Target position [x, y, z] (default: [1.5, 0.0, 1.5])

> **Note:** Requires GPU due to JAX automatic differentiation requirements.

### 3. Trajectory Tracking Training

Train a policy to follow reference trajectories:

```bash
# Basic usage
uv run lotf track --config configs/traj_tracking.yaml

# With trajectory export
uv run lotf track \
  --config configs/traj_tracking.yaml \
  --checkpoint checkpoints/policy/my_tracking_policy \
  --trajectory-output outputs/trajectory.csv
```

> **Note:** Requires GPU due to JAX automatic differentiation requirements.

## Checkpoints

Pretrained checkpoints are provided in the `checkpoints/` directory:

```
checkpoints/
├── policy/
│   ├── state_hovering_params
│   ├── traj_tracking_params
│   ├── vision_hovering_params
│   └── vision_hovering_pre_params
└── residual_dynamics/
    ├── dummy_params
    └── example_params
```

### Loading Checkpoints

```python
from orbax.checkpoint import PyTreeCheckpointer

ckptr = PyTreeCheckpointer()

# Load residual dynamics model
residual_params = ckptr.restore("checkpoints/residual_dynamics/example_params")

# Load policy checkpoint
policy_params = ckptr.restore("checkpoints/policy/state_hovering_params")
```

## Configuration Files

Available in `configs/`:

| File | Description |
|------|-------------|
| `residual_dynamics.yaml` | Residual dynamics ensemble training |
| `state_hovering.yaml` | State-based hovering policy |
| `traj_tracking.yaml` | Trajectory tracking policy |
| `vision_hovering.yaml` | Vision-based hovering policy |

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

## Examples

Jupyter notebooks with detailed walkthroughs are available in `examples/`:

| Directory | Description |
|-----------|-------------|
| `residual_dynamics/` | Ensemble model training |
| `state_hovering/` | State-based hovering (train, eval, finetune) |
| `traj_tracking/` | Trajectory tracking (train, eval, finetune) |
| `vision_hovering/` | Vision-based hovering (pretrain, train, finetune) |

### Training Results

**State-Based Hovering:**

| Reward Curve | Policy Rollout |
|---------------|----------------|
| <img src="assets/state_hovering_reward.png" width="300"> | <img src="assets/state_hovering_rollout.png" width="500"> |

**Trajectory Tracking:**

| Reward Curve | Policy Rollout |
|---------------|----------------|
| <img src="assets/tracking_reward.png" width="300"> | <img src="assets/tracking_rollout.png" width="430"> |

**Vision-Based Hovering:**

| Reward Curve | Policy Rollout |
|---------------|----------------|
| <img src="assets/vision_hovering_reward.png" width="300"> | <img src="assets/vision_hovering_rollout.png" width="500"> |

## Reference Trajectories

Predefined trajectories in `lotf/objects/reference_traj_obj.py`:

| Name | Description |
|------|-------------|
| `CIRCLE` | Smooth circle (radius=1m, period=3s) |
| `FIG8` | Smooth figure-8 (3m x 1m, period=5s) |
| `STAR` | Non-smooth star (side=2m, period=6s) |

## ROS2 Integration

This implementation is compatible with **ROS2 Humble** (Python 3.10). For hardware deployment with ROS stacks, ensure your ROS2 workspace is sourced before running training scripts.

## Troubleshooting

### GPU Requirements

If you encounter JVP errors:
```
TypeError: Custom JVP rule must produce primal and tangent outputs...
```

Ensure:
1. CUDA-capable GPU is available
2. CUDA toolkit 12.x is installed
3. JAX with CUDA support: `uv sync --extra cuda12`

### Checkpoint Loading Warnings

Warnings about sharding info are expected when loading checkpoints across CPU/GPU:
```
UserWarning: Sharding info not provided when restoring...
```
Checkpoints will still load correctly.

## Contact

For questions, use the [GitHub issue tracker](https://github.com/uzh-rpg/learning_on_the_fly/issues) or contact [Michael Pan](mailto:michael.pan31415@gmail.com).

## Acknowledgements

We thank the authors of [flightning](https://github.com/uzh-rpg/rpg_flightning) for open-sourcing their code, which provided the foundation of this codebase.

## Citation

```bibtex
@inproceedings{pan2026learning,
  title={Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation},
  author={Pan, Jiahe and Xing, Jiaxu and Reiter, Rudolf and Zhai, Yifan and Aljalbout, Elie and Scaramuzza, Davide},
  booktitle = {IEEE Robotics and Automation Letters},
  year={2026}
}
```
