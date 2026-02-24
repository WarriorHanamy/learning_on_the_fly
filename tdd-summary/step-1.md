# Step 1 - Understand Intent

## Functional Requirements

### FR-1: CLI Module Executable
The script `lotf/scripts/train_traj_tracking.py` must be executable via `uv run python -m lotf.scripts.train_traj_tracking`.

### FR-2: CLI Argument Parsing
The CLI must accept the following arguments:
- `--config`: Path to YAML configuration file (default: `configs/traj_tracking.yaml`)
- `--checkpoint`: Path to save the trained policy checkpoint (default: `checkpoints/policy/traj_tracking_params`)
- `--trajectory-output`: Path to export trajectory CSV file (optional)

### FR-3: Environment Creation
The `create_env` function must create `TrajTrackingStateEnv` from config with:
- Correct observation space dimensions
- Reference trajectory loaded from config
- Properly wrapped with MinMaxObservationWrapper, LogWrapper, VecEnv

### FR-4: Reference Trajectory Loading
Reference trajectory must load correctly from configured `ref_traj_name` (CIRCLE, FIG8, STAR).

### FR-5: Training Loop
Script must run training loop with:
- BPTT algorithm from `lotf.algos.bptt`
- Policy checkpoint saving after training
- Progress output with epoch metrics

### FR-6: Trajectory Export
Trajectory export generates CSV with columns:
- `index`, `t` (time)
- `px`, `py`, `pz` (position)
- `qw`, `qx`, `qy`, `qz` (quaternion)
- `vx`, `vy`, `vz` (velocity)

## Assumptions

- Follow existing `train_state_hovering.py` structure and patterns
- Use dataclass config with `from_yaml` classmethod
- Environment uses wrappers: MinMaxObservationWrapper, LogWrapper, VecEnv
- Policy: MLP with hidden layers from config, action bias from hovering_action
- Optimizer: Adam with cosine decay schedule
- Use `TrajTrackingStateEnv.generate_csv` for trajectory export
