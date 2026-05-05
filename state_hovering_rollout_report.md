# State Hovering Rollout Report

## Scope
This report summarizes how the current state-hovering rollout is defined (inputs/outputs), and whether a heading target is fixed.

## Rollout API (Generic)
Source: `lotf/envs/env_base.py` (`rollout`).

Inputs:
- `env`: environment instance (e.g., `HoveringStateEnv` or a wrapper around it).
- `key`: JAX PRNG key for the rollout.
- `policy`: callable mapping `(obs, key) -> action`.
- `res_model_params`: residual dynamics parameters (a `FrozenDict`).
- `state` (optional): initial `EnvState` to override `reset`.
- `real_step` (optional, default `False`): whether to use `env.step` (auto-reset) vs `env._step` (raw transition).
- `num_steps` (optional): length of rollout; defaults to `env.max_steps_in_episode`.

Outputs:
- Returns `EnvTransition`, where each field is a time-stacked JAX array of length `num_steps + 1` because the initial state is prepended.
- Fields: `state`, `obs`, `reward`, `terminated`, `truncated`, `info`.

## State Hovering Environment Inputs/Outputs
Source: `lotf/envs/hovering_state_env.py` (`HoveringStateEnv`).

Observation design (returned by `reset` and `_get_obs`):
- Position `p` (3).
- Rotation matrix `R` flattened to 9.
- Linear velocity `v` (3).
- Action history `last_actions` flattened (`num_last_actions * 4`).
- Total obs dim: `15 + 4 * num_last_actions` (confirmed by `observation_space`).

Action design:
- 4-dim vector `[thrust, body_rate_x, body_rate_y, body_rate_z]` with bounds from `action_space`.

State fields (stored in rollout transition):
- `time`, `step_idx`, `quadrotor_state` (position/rotation/vel/omega/acc),
- `last_actions`, `last_quadrotor_states`.

## Heading/Target Behavior
- The environment reward does **not** include a heading/orientation target.
- The only target used in reward is the **position goal** (`self.goal`), with penalties on velocity, angular velocity, acceleration, and action effort.
- `self.goal` is fixed at initialization:
  - Default: `[0.0, 0.0, hover_height]`.
  - Overridden by `hover_target` if provided (e.g., in `examples/state_hovering/2_eval_policy.ipynb`, `hover_target=[1.5, 0.0, 1.5]`).
- The plot function visualizes the **current body-frame heading** (using `R[:, 0]` arrows) but does not compare to a target heading.

## Notebook Usage Pattern
Source: `examples/state_hovering/2_eval_policy.ipynb`.

- Rollout is vectorized with `jax.vmap(rollout, in_axes=(None, 0, None, None))`.
- Inputs per rollout: `env`, a batch of `key`s, `policy`, and `dummy_residual_params`.
- Output is a batched `EnvTransition` (trajectory stack over both rollout index and time).

## Direct Answer
- Input/output of `state_hovering` rollout is governed by `env_base.rollout` and `HoveringStateEnv` observation/action design described above.
- The **heading target is not parameterized**; only the **position goal** is fixed (unless `hover_target` is explicitly set).
