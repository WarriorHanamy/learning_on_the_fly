# Model Analysis — Learning on the Fly (LOTF)

This document analyzes the three core models in the LOTF training pipeline:
forward rollout, residual dynamics, and backward nominal model.

---

## 1. Forward Rollout Model

The forward rollout is a **two-tier simulation stack**:

1. **Environment-level**: control delay simulation
2. **Physics-level**: quadrotor dynamics integration (two fidelity modes)

### 1.1 Control Delay Simulation

Both `HoveringStateEnv._step()` and `TrajTrackingStateEnv._step()` implement a rolling action buffer to simulate real-world control latency.

```python
num_last_actions = ceil(delay / dt) + 1   # e.g. 0.04 / 0.02 + 1 = 3

# Apply the oldest buffered action for fractional time
dt_1 = delay - (num_last_actions - 2) * dt
quadrotor.step(state, action_buffer[0], dt_1)

# Apply the next buffered action for remaining time
if dt_1 < dt:
    dt_2 = dt - dt_1
    quadrotor.step(state, action_buffer[1], dt_2)
```

- `delay == dt` → oldest action applies for the full timestep
- `delay == 0` → immediate action application (no buffer)

**Source**: `lotf/envs/hovering_state_env.py:177-221`, `lotf/envs/traj_tracking_state_env.py:210-250`

### 1.2 Physics Model Dispatch

`Quadrotor.step()` in `lotf/objects/quadrotor_obj.py:329-403` is decorated with
`@partial(jax.custom_jvp, nondiff_argnums=(3,))`. The forward (primal) computation bifurcates:

#### High-Fidelity Mode (`use_high_fidelity=True`)

Inner-loops at `dt_low_level = 0.001 s` (e.g. 20 substeps per 0.02 s frame).

```
For each substep:
  1. Low-Level Controller (_llc_betaflight):
     thrust → SBUS signal → normalized throttle
     Body-rate PD: τ = Kp·(ω_cmd − ω) + Kd_adj·dω
     4×4 mixer: [throttle, τx, τy, τz] → 4 motor throttles
     Throttle → DShot → motor rad/s target

  2. Full Dynamics (_full_dyn):
     --- Translation ---
     Fi = thrust_map_i · ω_i²           (domain-randomized ±15%)
     F_vec_body = [0, 0, ΣFi]
     acc = g + R·(F_vec_body / m)
           + rotor_aero_residual         (polynomial model)
           + res_acc_mean                (learned NN ensemble mean)

     RK4(p, v):
       dp/dt = v
       dv/dt = acc

     --- Rotation ---
     R_new = R @ exp(dt · ω^)            (exact Lie group)

     --- Motors ---
     dω_i/dt = (1/τ) · (ω_i_d − ω_i)    (1st-order lag, τ = 0.033 s)

     --- Angular ---
     dω/dt = J⁻¹[τ_aero − ω×Jω + τ_inertia]
     RK4(ω)
```

**Key characteristics**: full Betaflight emulation, motor lag, polynomial aero corrections, domain randomization, and optionally the learned NN residual.

**Source**: `lotf/objects/quadrotor_obj.py:339-355`, `:527-551`, `:442-525`

#### Low-Fidelity Mode (`use_high_fidelity=False`)

```
dv/dt = g + R · [0, 0, thrust/mass]  +  a_res   (optional NN residual)
dp/dt = v

RK4(p, v)
R_new = R @ exp(dt · ω^)              (exact Lie group)
```

No motor dynamics, no Betaflight controller, no polynomial aero, no angular dynamics
integration — `omega` is treated as a direct command.

**Source**: `lotf/objects/quadrotor_obj.py:358-368`, `:408-440`, `:615-654`

### 1.3 Polynomial Rotor Augmentation

The high-fidelity forward model includes a polynomial residual model for
aerodynamic force/torque corrections (`lotf/simulation/model_rotor.py:105-173`):

```
Features (per axis, body frame):
  fx: [vx, m_mean, ωy, vx·m_mean, vx|vx|, vx³]
  fy: [vy, m_mean, ωx, vy·m_mean, vy|vy|, vy³]
  fz: [1, v_hor, vz, m_mean, v_hor·vz, v_hor·m_mean, vz·m_mean, v_hor², v_hor·vz·m_mean, vz³]
  τx: [vy, m_mean, vy·m_mean]
  τy: [vx, m_mean, vx·m_mean]
  τz: [vx, vy]

where m_mean = mean(ω_i²),  v_hor = √(vx² + vy²)

Residual = scale · (features @ coefficients)
```

---

## 2. Residual Dynamics Model

The residual model learns to predict the **unmodeled acceleration** — the gap
between real-world physics and the nominal point-mass model.

### 2.1 Definition

```
residual = actual_acceleration − nominal_acceleration

nominal = g + R · [0, 0, thrust/mass]
```

From flight data, the residual is extracted post-hoc and stored as a supervised
learning target.

### 2.2 Network Architecture

**`ResidualDynamicsMLP`** (`lotf/modules/mlp.py:151-183`):

```
Input (19-dim):  [p(3), R_vec(9), v(3), thrust(1), ω_d(3)]
Architecture:    [19, 128, 128, 3] with ReLU
Init:            Variance scaling (scale=1.0, fan_avg, normal)
Output (3-dim):  [ax_res, ay_res, az_res] in world frame
```

### 2.3 Ensemble Approach

Default ensemble size: **3 models**, identical architecture, different random
initialization. Implemented via `jax.vmap` over model dimension.

| Stage      | Mechanism                                         |
|------------|---------------------------------------------------|
| Init       | `jax.vmap(init_fn, in_axes=(None, 0))` over seeds |
| Train      | `jax.vmap(train, in_axes=(0, None, ...))` over models |
| Predict    | `jax.vmap(predict_fn, in_axes=(0, None))` over models |

At inference time inside the simulation:
```python
preds = ensemble_predict(params, input)       # shape: (num_models, 3)
res_acc_mean = jnp.mean(preds, axis=0)        # ensemble mean → (3,)
```

**Source**: `lotf/utils/residual_dynamics.py:136-145`, `:12-23`

### 2.4 Training Objective

```
L = MSE(pred, y) + λ_reg · Σ ‖W_layer‖₂

‖W_layer‖₂ = max singular value (spectral norm)
```

Spectral regularization encourages smooth, Lipschitz-bounded predictions.
Optimized with Adam via `jax.lax.scan`.

**Source**: `lotf/utils/residual_dynamics.py:34-69`, `:92-133`

### 2.5 Integration into Forward Simulation

When `use_forward_residual=True`:

1. A 19-dim input vector is assembled from the current quadrotor state
2. The ensemble predicts `res_acc_mean`
3. This is added to the acceleration:
   - **High-fidelity**: `acc = g + R·F_vec/m + rotor_aero + res_acc_mean`
   - **Low-fidelity**: `dv/dt = g + R·[0,0,thrust/m] + a_res`

**Source**: `lotf/objects/quadrotor_obj.py:294-326`, `:469`, `:423`

---

## 3. Backward Nominal Model (BPTT)

### 3.1 The Mixed DiffSim Design

The core design: **high-fidelity forward, simplified backward**. This is
implemented via a **custom JVP** (Jacobian-Vector Product) on `Quadrotor._step()`.

```
@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _step(state, f_d, omega_d, dt):
    # primal: full high-fidelity (or low-fidelity with residual)
    ...
    return state_new, residual  # residual contains res_acc_mean

@_step.defjvp
def _step_jvp(dt, primals, tangents):
    # forward: still uses the full _step()
    state_new, _ = _step(state, f_d, omega_d, dt)

    # backward (tangent): uses ONLY simplified_dyn()
    _, tan_out = jax.jvp(
        simplified_dyn,
        (p, R, v, f_d/mass, omega_d, dt),
        (ṗ, Ṙ, v̇, ḟ_d/mass, ω̇_d, 0.0)
    )
    state_dot_new = state_dot.replace(p=p_tan, R=R_tan, v=v_tan)
    return state_new, state_dot_new
```

**Source**: `lotf/objects/quadrotor_obj.py:329-403`

### 3.2 Backward Model Equations

The gradient-relevant model used during the backward pass is `simplified_dyn()`:

```
dv/dt = g + R · [0, 0, thrust/mass]
dp/dt = v

Integrated via RK4(p, v)
Rotation via exact Lie group: R_new = R @ exp(dt · ω^)
```

This is analytically differentiable and numerically stable for long horizons.

### 3.3 What the Backward Pass Ignores

| Aspect                   | Forward                          | Backward (tangent)    |
|--------------------------|----------------------------------|-----------------------|
| Motor dynamics           | 1st-order lag (τ = 0.033 s)     | Ignored               |
| Betaflight controller    | PD + SBUS + DShot chain         | Ignored               |
| Polynomial aero model    | 10-term polynomial corrections   | Ignored               |
| Learned NN residual      | Ensemble mean added to `a`      | Ignored               |
| Domain randomization     | ±15% thrust_map noise            | Excluded (dr_key)     |
| Angular dynamics         | RK4(ω) with J⁻¹                 | Ignored (ω is direct) |

### 3.4 BPTT Gradient Flow

```
∂L/∂θ = ∂L/∂reward · ∂reward/∂state · ∂state/∂action · ∂action/∂θ
                             ↑ simplified_dyn JVP ↑       ↑ policy NN ↑
```

1. `loss_fn` calls `jax.value_and_grad` over an epoch rollout
2. The rollout runs `jax.lax.scan` over `num_steps_per_epoch` steps
3. Each step: `policy(obs) → action → env.step() → quadrotor.step()`
4. Gradients flow backward through time via the custom JVP
5. Optimizer (Adam with cosine decay) updates policy parameters

**Source**: `lotf/algos/bptt.py:67-187`

### 3.5 Decay Factor

A tunable `decay_factor` (default 1.0) can be applied to the tangent to mitigate
vanishing/exploding gradients on long horizons:

```python
state_dot_new = state_dot.replace(
    p=decay_factor * p_tan,
    R=decay_factor * R_tan,
    v=decay_factor * v_tan,
)
```

---

## 4. Complete Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                            │
│                                                                  │
│  Stage 1: Train Residual Model                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  CSV dataset: [state(19), action(6)] → residual_acc(3)     │  │
│  │       ↓                                                     │  │
│  │  Ensemble of 3 MLPs: [19, 128, 128, 3]                     │  │
│  │       ↓  MSE + spectral norm regularization               │  │
│  │  Checkpoint: residual_dynamics/residual_params             │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              ↓                                    │
│  Stage 2: Train Policy (BPTT)                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Policy MLP: [obs_dim, 512, 512, 4] + hovering_bias       │  │
│  │       ↓                                                     │  │
│  │  [FORWARD]   Quadrotor.step() → full dynamics + residual   │  │
│  │       ↓  reward = −dt · Σsmooth_l1(state_error, action)    │  │
│  │  [BACKWARD]  simplified_dyn() only (custom JVP)            │  │
│  │       ↓  gradient flows through simple point-mass model    │  │
│  │  Update policy params via Adam + cosine decay              │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

| Concept | File | Lines |
|---------|------|-------|
| `simplified_dyn()` (nominal) | `lotf/objects/quadrotor_obj.py` | 615–654 |
| `_simplified_res_dyn()` | `lotf/objects/quadrotor_obj.py` | 408–440 |
| `_full_dyn()` | `lotf/objects/quadrotor_obj.py` | 442–525 |
| `_llc_betaflight()` | `lotf/objects/quadrotor_obj.py` | 527–551 |
| `_step()` + custom JVP | `lotf/objects/quadrotor_obj.py` | 329–403 |
| Polynomial aero residuals | `lotf/simulation/model_rotor.py` | 105–173 |
| `ResidualDynamicsMLP` | `lotf/modules/mlp.py` | 151–183 |
| Ensemble predict | `lotf/utils/residual_dynamics.py` | 12–23 |
| Ensemble training | `lotf/utils/residual_dynamics.py` | 92–145 |
| BPTT training loop | `lotf/algos/bptt.py` | 67–187 |
| Control delay | `lotf/envs/hovering_state_env.py` | 177–221 |
| Reward computation | `lotf/envs/hovering_state_env.py` | 223–258 |
