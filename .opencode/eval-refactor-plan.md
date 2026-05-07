# Evaluation System & Forward Model Naming Refactor

**Status**: Planned | **Date**: 2026-05-07 | **Baseline commit**: b602fa7

---

## Intent

1. 建立一等公民 `evaluation` 入口（正式 CLI + 模块），不再靠 example 脚本手工评估
2. 统一 benchmark: `traj_tracking / FIG8`，simulator 固定为 `residual_acceleration + inner_loop_dynamics`
3. 全面重命名 simulator 的两个公开开关，消除误导命名
4. 抽出两个训练脚本、visualize、eval 之间共用的配置/env/policy builder

## Feasibility

**Feasible** — 所有受影响文件的语义边界已确认，checkpoint 不受命名影响，变更范围可控。

---

## Context

### Current Issues

| Issue                                                | Evidence                                                                                     |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| 无正式 evaluation CLI/模块                             | 评估靠 `examples/*/eval_policy.py` 独立脚本，无命令行入口                                        |
| 3 个 eval 脚本 90% 相同                                 | `examples/state_hovering/eval_policy.py` (107L) vs `examples/traj_tracking/eval_policy.py` (115L) — 仅 env 类和 checkpoint 名不同 |
| `SimDynConfig` 重复定义 2 次                            | `train_state_hovering.py:52-58` 和 `train_traj_tracking.py:55-61` — 字段完全相同               |
| 公开 flag 命名误导: `use_forward_residual`               | 实际行为是加 `res_acc_mean` 到加速度向量；"forward" 泄漏 JVP surrogate 实现细节                  |
| 公开 flag 命名误导: `use_high_fidelity`                  | 实际切换的是 inner-loop body-rate controller + motor dynamics，不是通用 high-fidelity              |
| 8 个 example 脚本 hardcode `sim_dyn_config` dict         | `examples/{state_hovering,traj_tracking}/*.py` — 每处 4 行 dict literal                         |
| visualize_policy.py 自建 sim_dyn_config                  | `visualize_policy.py:42-44` 从 YAML 重新解析                                                  |
| dummy_residual_params 路径硬编码 8+ 处                    | `lotf/scripts/` 和 `examples/` 各写各的                                                        |
| `create_env` / `create_policy` 在两个 train 脚本中重复    | hover 版 `train_state_hovering.py:184-271` vs track 版 `train_traj_tracking.py:184-270`        |

### Architecture Boundaries

```
User → CLI (__main__.py)
         ├── train hover   → train_state_hovering.py   → bptt.py
         ├── train track   → train_traj_tracking.py    → bptt.py
         ├── train residual → train_residual.py
         ├── play hover    → visualize_policy.py
         ├── play track    → visualize_policy.py
         └── [NEW] eval track → evaluate_traj_tracking.py → eval/runner.py

Quadrotor.step() [quadrotor_obj.py:273-408]
  └── two boolean flags forming 4 forward paths
       ├── enable_residual_acceleration: adds res_acc_mean to acceleration
       └── enable_inner_loop_dynamics: adds LLC + motor lag + full RK4 integration
```

---

## Task

### Parent Task

建立 traj_tracking FIG8 统一评估系统 + 全仓重命名 simulator backend 命名。

### Deliverables

1. `lotf/forward_model_config.py` — 共享 ForwardModelConfig dataclass
2. `lotf/eval/` — 评估模块 (runner + metrics)
3. `lotf/scripts/evaluate_traj_tracking.py` — 评估入口脚本
4. `configs/benchmark_traj_fig8.yaml` — 固定 benchmark 配置
5. `lotf/__main__.py` — 新增 `eval` subcommand
6. `pyproject.toml` — 新增 `eval` entry point
7. `lotf/objects/quadrotor_obj.py` — internal API 重命名
8. 所有 `.py` / `.yaml` 中 `use_high_fidelity` → `enable_inner_loop_dynamics`, `use_forward_residual` → `enable_residual_acceleration`
9. `examples/traj_tracking/eval_policy.py` — 退役 (tag deprecated)
10. `lotf/scripts/train_traj_tracking.py` — 抽出共享 env/policy builder

### Acceptance

- `uv run eval track --checkpoint <path>` 可运行，固定 FIG8 + full fidelity benchmark
- benchmark 输出 mean episodic return, collision rate, position RMSE, velocity RMSE
- 所有 YAML 配置用新字段名可正常加载
- `uv run train hover` / `uv run train track` 功能不变
- `uv run play hover` / `uv run play track` 功能不变
- 旧名 `use_high_fidelity` / `use_forward_residual` 在全仓无残留

---

## Child Tasks

### Child Task 1: Create `lotf/forward_model_config.py`

**Deliverable**: 单一 dataclass 文件，定义 `ForwardModelConfig`。

```python
# lotf/forward_model_config.py
@dataclass(frozen=True)
class ForwardModelConfig:
    enable_residual_acceleration: bool = False
    enable_inner_loop_dynamics: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "enable_residual_acceleration": self.enable_residual_acceleration,
            "enable_inner_loop_dynamics": self.enable_inner_loop_dynamics,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ForwardModelConfig:
        return cls(
            enable_residual_acceleration=d.get("enable_residual_acceleration", False),
            enable_inner_loop_dynamics=d.get("enable_inner_loop_dynamics", False),
        )
```

**Depends on**: 无
**Verification**: importable as `from lotf.forward_model_config import ForwardModelConfig`

---

### Child Task 2: Rename `SimDynConfig` → `ForwardModelConfig` in train scripts + all consumers

**Deliverable**: 全部 `.py` 和 `.yaml` 文件中旧字段名替换完毕。

**YAML 文件** (3 个):
- `configs/state_hovering.yaml:28-29`
- `configs/traj_tracking.yaml:22-23`  
- `configs/test_volume.yaml:28-29`

```
# Before
sim_dyn_config:
  use_high_fidelity: false
  use_forward_residual: false

# After
forward_model_config:
  enable_inner_loop_dynamics: false
  enable_residual_acceleration: false
```

**训练脚本** (`lotf/scripts/train_state_hovering.py`, `lotf/scripts/train_traj_tracking.py`):
- 删除本地 `SimDynConfig` dataclass 定义
- 导入 `from lotf.forward_model_config import ForwardModelConfig`
- 字段引用全部重命名
- sim_dyn_config 参数传参更新

**visualize_policy.py**:
- `visualize_policy.py:42-44` 的 dict 构造改用新字段名
- key 更新为 `"forward_model_config"` 和 `"enable_inner_loop_dynamics"` / `"enable_residual_acceleration"`

**example 脚本** (8 个):
- `examples/state_hovering/eval_policy.py:36-37`
- `examples/state_hovering/train_base_policy.py:41-42`
- `examples/state_hovering/finetune_policy_full.py:39-40`
- `examples/state_hovering/finetune_policy_lora.py:45-46`
- `examples/traj_tracking/eval_policy.py:36-37`
- `examples/traj_tracking/train_base_policy.py:41-42`
- `examples/traj_tracking/finetune_policy_full.py:39-40`
- `examples/traj_tracking/finetune_policy_lora.py:45-46`

**Depends on**: Child Task 1

---

### Child Task 3: Rename `Quadrotor` internal API + dispatch logic

**Deliverable**: `quadrotor_obj.py` 中所有 exposed 字段和内部方法重命名完毕。

**Prerequisite key changes**:

| From                        | To                                          | Line(s)  |
| --------------------------- | ------------------------------------------- | -------- |
| `self.use_high_fidelity`      | `self._enable_inner_loop_dynamics`            | 164, 341, 613 |
| `self.use_forward_residual`   | `self._enable_residual_acceleration`          | 165, 168, 296, 361, 614 |
| `self.compute_res_fn`         | `self._compute_residual_acceleration_fn`      | 169, 326 |
| `sim_dyn_config` (param name) | `forward_model_config` (param name)           | 120, 158-165 |
| dict key `"use_high_fidelity"`  | `"enable_inner_loop_dynamics"`                | 161, 164 |
| dict key `"use_forward_residual"` | `"enable_residual_acceleration"`           | 162, 165 |
| standalone `simplified_dyn()`   | `integrate_nominal_dynamics()`                | 617 |
| `_simplified_res_dyn()`         | `_integrate_nominal_dynamics_with_residual()` | 411 |
| `_full_dyn()`                   | `_integrate_inner_loop_dynamics()`            | 444 |
| `_llc_betaflight()`             | `_compute_inner_loop_motor_commands()`        | 529 |
| `print_config()` print strings  | update to new names                           | 612-614 |

**Note**: `integrate_nominal_dynamics()` 保持为 standalone function (被 custom JVP 调用)

**Depends on**: Child Task 1

---

### Child Task 4: Extract shared env/policy builder from `train_traj_tracking.py`

**Deliverable**: 共享 builder 函数，训练/visualize/eval 三个消费者统一使用。

**Extract**: `create_env()` + `create_policy()` from `train_traj_tracking.py`

**New location**: 建议直接复用 `train_traj_tracking.py` 里的函数作为 export（轻量方案），或抽到 `lotf/envs/traj_builder.py`。

**方案 A (轻量)**: 在 `train_traj_tracking.py` 顶部加 `# Shared builders for traj_tracking env/policy`，由 visualize 和 eval 直接 import。

**方案 B (重)**: 新建 `lotf/envs/traj_builder.py`。

**推荐方案 A** (最小改动)：
- `visualize_policy.py:40-110` 中 `load_policy_and_env()` 改用 `train_traj_tracking.create_env()` + `train_traj_tracking.create_policy()`
- eval 脚本也复用同一套

**同时**: 在 `train_state_hovering.py` 中也做同样兼容，让 hover 路径以后可复用。

**Depends on**: Child Task 2

---

### Child Task 5: Create `lotf/eval/` module

**Deliverable**: `lotf/eval/` 目录，含 `__init__.py`, `runner.py`。

**`lotf/eval/runner.py`** 核心接口:

```python
@dataclass
class BenchmarkMetrics:
    mean_episodic_return: float
    collision_rate: float
    mean_episode_length: float
    position_rmse: float
    velocity_rmse: float

def run_benchmark(
    env,                          # TrajTrackingStateEnv (wrapped)
    policy_fn: Callable,
    residual_params: FrozenDict,
    ref_traj: jnp.ndarray,       # reference trajectory for RMSE
    num_rollouts: int,
    seed: int,
) -> tuple[BenchmarkMetrics, EnvTransition]:
    """
    1. vmap(rollout) over num_rollouts
    2. compute reward statistics from transitions
    3. compute position/velocity RMSE against ref_traj
    4. return metrics + raw transitions (for optional plotting)
    """
```

**`lotf/eval/__init__.py`**:
```python
from lotf.eval.runner import BenchmarkMetrics, run_benchmark
```

**Depends on**: 无

---

### Child Task 6: Create `configs/benchmark_traj_fig8.yaml`

**Deliverable**: 固定 benchmark 配置，不暴露 simulator 开关给用户。

```yaml
# Benchmark configuration for trajectory tracking (FIG8)
# Simulator backend is FIXED — not configurable.

task: traj_tracking
ref_traj_name: fig8

env:
  sim_dt: 0.02
  delay: 0.04
  max_sim_time: 10.0
  skip_start: true
  # noise zeroed for deterministic evaluation
  yaw_scale: 0.0
  pitch_roll_scale: 0.0
  position_std: 0.0
  velocity_std: 0.0
  omega_std: 0.0

benchmark:
  num_rollouts: 20
  seed: 0
```

**Note**: `forward_model_config` 不出现在这个文件中 — eval 脚本内部硬编码 `enable_residual_acceleration=true, enable_inner_loop_dynamics=true`。

**Depends on**: 无

---

### Child Task 7: Create `lotf/scripts/evaluate_traj_tracking.py`

**Deliverable**: 评估入口脚本，可独立运行或被 `__main__.py` dispatch。

**核心流程**:
```
1. parse args: --checkpoint (required), --benchmark-config (default), --policy-config (for net structure), --residual-checkpoint (optional, default dummy)
2. load policy_config → extract network shape
3. load benchmark_config → build env with FIXED forward_model (hardcoded both=true)
4. restore policy params from --checkpoint → build TrainState
5. restore residual params → FrozenDict
6. call run_benchmark(...) → BenchmarkMetrics
7. print summary to stdout + optionally save summary.json
```

**CLI arguments**:
- `--checkpoint` (required): 策略 checkpoint 路径
- `--benchmark-config` (default: `configs/benchmark_traj_fig8.yaml`)
- `--policy-config` (default: `configs/traj_tracking.yaml`)
- `--residual-checkpoint` (optional, default: `checkpoints/residual_dynamics/dummy_params`)
- `--output` (optional): JSON 输出路径
- `--plot` (optional flag): 输出 plot.png

**Depends on**: Child Tasks 1, 2, 3, 4, 5, 6

---

### Child Task 8: Add `eval` entry point to CLI

**Deliverable**: `__main__.py` + `pyproject.toml` 更新。

**`lotf/__main__.py`**:
- 新增 `main_eval()` 函数 (类似 `main_play()`)
- 在 `create_parser()` 中新增 `eval` subcommand

**`pyproject.toml`**:
```toml
[project.scripts]
train = "lotf.__main__:main"
play = "lotf.__main__:main_play"
eval = "lotf.__main__:main_eval"      # NEW
```

**CLI 预期形态**:
```bash
uv run eval track --checkpoint checkpoints/policy/traj_tracking_params
uv run eval track --checkpoint <path> --output results.json --plot
```

**Depends on**: Child Task 7

---

### Child Task 9: Retire `examples/traj_tracking/eval_policy.py`

**Deliverable**: 旧 eval script 退役，留下一行指向新 CLI 的说明。

**Action**: 在文件顶部用注释声明 deprecated，指向 `uv run eval track`。或直接删除（既然已无 notebook 依赖）。

**Depends on**: Child Task 8

---

## Rename Map (Full Reference)

### Public API

| Old                                   | New                                     |
| ------------------------------------- | --------------------------------------- |
| `use_high_fidelity` (config field)      | `enable_inner_loop_dynamics`             |
| `use_forward_residual` (config field)   | `enable_residual_acceleration`           |
| `sim_dyn_config` (block name)           | `forward_model_config`                   |
| `SimDynConfig` (dataclass)              | `ForwardModelConfig`                     |
| `sim_dyn_config_dict` (local variable)  | `forward_model_cfg_dict`                 |

### Quadrotor Internal

| Old                                | New                                          |
| ---------------------------------- | -------------------------------------------- |
| `self.use_high_fidelity`             | `self._enable_inner_loop_dynamics`             |
| `self.use_forward_residual`          | `self._enable_residual_acceleration`           |
| `self.compute_res_fn`                | `self._compute_residual_acceleration_fn`       |
| `simplified_dyn()` (standalone)      | `integrate_nominal_dynamics()`                 |
| `_simplified_res_dyn()`              | `_integrate_nominal_dynamics_with_residual()`  |
| `_full_dyn()`                        | `_integrate_inner_loop_dynamics()`             |
| `_llc_betaflight()`                  | `_compute_inner_loop_motor_commands()`         |

### Affected Files (Complete)

| File                                            | Change Type                |
| ----------------------------------------------- | -------------------------- |
| `lotf/forward_model_config.py`                    | **NEW**                    |
| `lotf/eval/__init__.py`                           | **NEW**                    |
| `lotf/eval/runner.py`                             | **NEW**                    |
| `lotf/scripts/evaluate_traj_tracking.py`          | **NEW**                    |
| `configs/benchmark_traj_fig8.yaml`                | **NEW**                    |
| `lotf/objects/quadrotor_obj.py`                   | internal rename + dispatch |
| `lotf/scripts/train_state_hovering.py`            | rename + import shared     |
| `lotf/scripts/train_traj_tracking.py`             | rename + import shared + extract builders |
| `lotf/scripts/visualize_policy.py`                | rename + reuse builders    |
| `lotf/__main__.py`                                 | add eval command           |
| `pyproject.toml`                                  | add eval entry point       |
| `configs/state_hovering.yaml`                     | rename keys                |
| `configs/traj_tracking.yaml`                      | rename keys                |
| `configs/test_volume.yaml`                        | rename keys                |
| `examples/state_hovering/train_base_policy.py`    | rename dict keys           |
| `examples/state_hovering/eval_policy.py`          | rename dict keys           |
| `examples/state_hovering/finetune_policy_full.py` | rename dict keys           |
| `examples/state_hovering/finetune_policy_lora.py` | rename dict keys           |
| `examples/traj_tracking/train_base_policy.py`     | rename dict keys           |
| `examples/traj_tracking/eval_policy.py`           | **deprecated / removed**   |
| `examples/traj_tracking/finetune_policy_full.py`  | rename dict keys           |
| `examples/traj_tracking/finetune_policy_lora.py`  | rename dict keys           |

---

## Verification

- [ ] `ForwardModelConfig` importable from `lotf.forward_model_config`
- [ ] `configs/state_hovering.yaml` loads with new field names (test: `from_yaml`)
- [ ] `configs/traj_tracking.yaml` loads with new field names
- [ ] `uv run train hover --config configs/state_hovering.yaml` starts successfully
- [ ] `uv run train track --config configs/traj_tracking.yaml` starts successfully
- [ ] `uv run play hover` launches viewer (smoke test)
- [ ] `uv run play track` launches viewer (smoke test)
- [ ] `uv run eval track --checkpoint checkpoints/policy/traj_tracking_params` runs to completion
- [ ] eval output contains: mean_episodic_return, collision_rate, position_rmse, velocity_rmse
- [ ] `grep -r "use_high_fidelity" lotf/ configs/ examples/` returns zero results (no residual)
- [ ] `grep -r "use_forward_residual" lotf/ configs/ examples/` returns zero results
- [ ] `grep -r "sim_dyn_config" lotf/ configs/` returns zero results
- [ ] `grep -r "simplified_dyn" lotf/` returns zero results
- [ ] `grep -r "_full_dyn" lotf/` returns zero results
- [ ] `grep -r "_llc_betaflight" lotf/` returns zero results
- [ ] `pytest` (if tests exist) or manual import check passes

## Constraints

- 不修改 checkpoint 存储格式 (Orbax OCDBT)
- 不修改 `algos/bptt.py` 逻辑
- 不修改 `envs/env_base.py` + `wrappers.py` 接口
- 不做 backward compatibility (hard break on naming)
- 第一版只做 `traj_tracking / FIG8` benchmark
