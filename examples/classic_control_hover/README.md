# Classic Control Hover — Closed-Loop Chirp Experiment

Self-contained example package for visualising the simulator's closed-loop
thrust and angular-loop response under an SE(3) hover controller with
artificial chirp injection.

## What this experiment does

1. Stabilises a quadrotor at a fixed hover point using an **SE(3) controller**
   that outputs total thrust + body-rate commands.
2. Injects a **sequential single-channel log chirp** on each of the four plant
   inputs (thrust, p-rate, q-rate, r-rate) — one at a time.
3. Records the full state trajectory (position, velocity, quaternion, angular
   velocity, acceleration) plus all command and error signals.
4. Generates **per-segment figures** showing commanded vs measured response so
   you can directly assess thrust-loop lag and angular-loop bandwidth.

## Design principle

The entire experiment package is **backend-agnostic** except for a single
adaptor file.  The canonical data contract is:

| Layer | Owns | Does NOT know |
|-------|------|---------------|
| `schema.py` | ``StateSample``, ``ControlModel``, ``ControllerDiagnostics``, constants | Any backend |
| `controller.py` | SE(3) hover math | Any backend |
| `chirp.py` | Chirp schedule & injection | Any backend |
| `recorder.py` | Log schema & persistence | Any backend |
| `plotting.py` | Per-segment figures | Any backend |
| `sim_adapter.py` ⬅ | ``LotfAdapterConfig``, native↔canonical conversion | (the ONLY backend-aware file) |
| `run_chirp_experiment.py` | Experiment orchestration | Backend details (only through adaptor) |

## Running

```bash
# default (full setting, 140 s)
uv run python -m examples.classic_control_hover.run_chirp_experiment

# specific simulator setting
uv run python -m examples.classic_control_hover.run_chirp_experiment --setting full
uv run python -m examples.classic_control_hover.run_chirp_experiment --setting innerloop

# custom output directory
uv run python -m examples.classic_control_hover.run_chirp_experiment --output outputs/my_run
```

## Convenience entrypoint

```bash
# from project root
uv run python examples/classic_control_hover/run_chirp_experiment.py
```

## Outputs

| File | Description |
|------|-------------|
| `log.npz` | Full experiment trace (canonical fields + `ext_*` backend extras) |
| `metadata.json` | Experiment parameters + chirp segment definitions |
| `segment_thrust.png` | Thrust command vs estimated thrust, z / vz / az |
| `segment_p.png` | p-rate command vs actual p, roll angle |
| `segment_q.png` | q-rate command vs actual q, pitch angle |
| `segment_r.png` | r-rate command vs actual r, yaw angle |
| `overview.png` | 4-channel chirp timeline + position/attitude error norms |

## Canonical conventions

| Aspect | Convention |
|--------|-----------|
| World frame | +z up, right-handed |
| Body frame | **FLU** (Forward=x, Left=y, Up=z) |
| Gravity | `[0, 0, -9.81]` m/s² in world frame |
| Rotation repr. | Hamilton quaternion `[qw, qx, qy, qz]` (scalar-first) |
| Action order | `[thrust (N), p (rad/s), q (rad/s), r (rad/s)]` |
| Body-rate order | `[p, q, r]` (right-hand rule about body axes) |

All conventions are defined in `conventions.py` and referenced throughout the
package.

## Canonical data types

| Type | Contents |
|------|----------|
| ``StateSample`` | ``p_world_m``, ``v_world_mps``, ``q_world_from_body_wxyz``, ``omega_body_radps``, ``acc_world_mps2``, ``extras`` |
| ``ControlModel`` | ``mass_kg``, ``thrust_limits_N``, ``rate_limits_body_radps`` |
| ``ControllerDiagnostics`` | ``f_cmd_N``, ``omega_cmd_body_radps``, ``e_pos_world_m``, ``e_R_body``, ``F_des_world_N``, ``R_des_world_from_body`` |

Backend-specific telemetry (raw rotation matrix, motor speeds, angular
acceleration, estimated thrust) is carried in ``StateSample.extras`` and
stored in the log with an ``ext_`` prefix.

## Package files

```
examples/classic_control_hover/
  schema.py               # Canonical conventions + data contracts (frame, sign, unit, dataclasses)
  controller.py           # Self-contained SE(3) hover controller
  chirp.py                # Chirp schedule and 4-channel injection
  recorder.py             # Log init / append / save
  plotting.py             # Per-segment matplotlib figures
  sim_adapter.py          # ⬅ ONLY bridge to LOTF backend
  run_chirp_experiment.py # Entrypoint
```

**Only `sim_adapter.py` imports from `lotf.*`.**

## Chirp schedule (default)

| Start [s] | Duration [s] | Channel | Amplitude | Frequency [Hz] |
|-----------|-------------|---------|-----------|----------------|
| 0         | 5           | –       | (settle)  | –              |
| 5         | 30          | thrust  | 0.08 mg   | 0.2 → 6        |
| 35        | 5           | –       | (recover) | –              |
| 40        | 30          | p       | 0.6 rad/s | 0.2 → 10       |
| 70        | 5           | –       | (recover) | –              |
| 75        | 30          | q       | 0.6 rad/s | 0.2 → 10       |
| 105       | 5           | –       | (recover) | –              |
| 110       | 30          | r       | 0.3 rad/s | 0.2 → 6        |

## Customisation

Edit `_build_default_config()` in `run_chirp_experiment.py` or
`default_chirp_segments()` in `chirp.py` to change the schedule, gains, or
hover point.

## Dependencies

- `lotf` (the parent project — provides `Quadrotor` and residual checkpoints)
- `numpy`, `matplotlib`, `jax`
