# Codebase Documentation

> Auto-generated repository documentation for Learning on the Fly (LOTF)

## Overview

Learning on the Fly (LOTF) 是一个基于 JAX 的可微仿真库，专注于四旋翼飞行器的敏捷飞行策略学习。该项目采用两阶段训练方法：首先从真实硬件数据中学习残差动力学，然后在包含学习残差的可微分仿真环境中训练策略。

**核心特性**：
- 可微物理仿真，支持自动微分
- 残差动力学学习，弥合 sim-to-real 差距
- BPTT（Backpropagation Through Time）策略优化
- 支持状态悬停、轨迹跟踪和视觉悬停任务
- JAX JIT 编译优化，支持 GPU 加速

**技术栈**：JAX, Flax, Optax, Orbax, PyTorch（概率建模部分）

---

## Repository Structure

```
learning_on_the_fly/
├── lotf/                    # 核心库代码
│   ├── algos/              # 算法层
│   │   └── bptt.py         # 通过时间反向传播 (BPTT)
│   ├── envs/               # 环境层
│   │   ├── env_base.py              # 环境基类
│   │   ├── hovering_state_env.py    # 状态悬停环境
│   │   ├── hovering_features_env.py # 特征悬停（视觉）环境
│   │   ├── traj_tracking_state_env.py # 轨迹跟踪环境
│   │   └── wrappers.py              # 环境包装器
│   ├── modules/            # 神经网络模块层
│   │   └── mlp.py          # MLP、LoRA MLP、残差动力学 MLP
│   ├── objects/            # 物理对象层
│   │   ├── quadrotor_obj.py         # 四旋翼对象
│   │   ├── world_box_obj.py         # 世界边界框
│   │   └── reference_traj_obj.py    # 参考轨迹（CIRCLE、FIG8、STAR）
│   ├── sensors/            # 传感器层
│   │   └── double_sphere_camera.py  # 双球面相机模型
│   ├── simulation/         # 仿真层
│   │   └── model_rotor.py            # 高保真四旋翼动力学模型
│   ├── utils/              # 工具层
│   │   ├── math.py      # 数学工具
│   │   ├── spaces.py    # 空间定义
│   │   ├── pytrees.py   # PyTree 工具
│   │   ├── random.py    # 随机数生成
│   │   ├── lora.py      # LoRA 实现
│   │   └── residual_dynamics.py  # 残差动力学工具
│   ├── configs/           # 配置加载
│   │   ├── configs.py   # 配置数据类
│   │   └── loader.py    # YAML 加载器
│   ├── scripts/           # 训练脚本
│   │   ├── train_state_hovering.py
│   │   ├── train_traj_tracking.py
│   │   └── train_residual.py
│   ├── __init__.py        # 包初始化
│   └── __main__.py        # CLI 统一入口点
│
├── examples/              # Jupyter notebook 示例
│   ├── residual_dynamics/ # 残差动力学（1个笔记本）
│   ├── state_hovering/    # 状态悬停（4个笔记本）
│   ├── traj_tracking/     # 轨迹跟踪（4个笔记本）
│   └── vision_hovering/   # 视觉悬停（5个笔记本）
│
├── tests/                 # 单元测试
│   ├── test_main.py
│   ├── configs/
│   │   └── test_config_loader.py
│   └── scripts/
│       ├── test_train_residual.py
│       ├── test_train_state_hovering.py
│       └── test_train_traj_tracking.py
│
├── configs/               # YAML 配置文件
│   ├── state_hovering.yaml
│   ├── traj_tracking.yaml
│   ├── vision_hovering.yaml
│   └── residual_dynamics.yaml
│
├── docs/                  # 场景文档（TDD 方法）
│   ├── scenario/          # 轨迹跟踪场景文档
│   └── scenario-train-residual/  # 残差动力学训练文档
│
├── checkpoints/           # 预训练模型检查点
│   ├── policy/
│   └── residual_dynamics/
│
├── assets/                # 可视化资源
│
├── AGENTS.md              # 仓库开发指南
├── README.md              # 项目说明文档
├── USAGE.md               # CLI 使用指南
├── LICENSE                # MIT 许可证
├── pyproject.toml         # 项目元数据和构建配置
├── setup.py               # setuptools 配置
├── environment.yml        # Conda 环境配置
└── uv.lock                # uv 依赖锁定文件
```

### 架构模式

**分层模块化单体架构**：
1. **算法层** (`algos/`): BPTT 梯度优化
2. **环境层** (`envs/`): 任务特定的模拟环境
3. **模块层** (`modules/`): 可复用的神经网络组件
4. **对象层** (`objects/`): 物理仿真对象
5. **传感器层** (`sensors/`): 感知模块
6. **仿真层** (`simulation/`): 物理动力学模型
7. **工具层** (`utils/`): 通用工具函数

### 文件命名约定

- **Python 模块**: `snake_case.py`（如 `hovering_state_env.py`）
- **类名**: `PascalCase`（如 `QuadrotorState`、`ResidualDynamicsConfig`）
- **函数名**: `snake_case`（如 `create_ensemble`、`load_dataset`）
- **Jupyter Notebooks**: 数字前缀 + 描述性名称（如 `1_train_base_policy.ipynb`）
- **配置文件**: `snake_case.yaml`（如 `state_hovering.yaml`）
- **测试文件**: `test_<module>.py`（如 `test_train_state_hovering.py`）

---

## Getting Started

### 环境要求

| 组件     | 规格                         |
| -------- | ---------------------------- |
| 操作系统 | Ubuntu 22.04 LTS             |
| Python   | 3.10（推荐）                 |
| CUDA     | 12.x                         |
| GPU      | NVIDIA GPU with CUDA support |
| 内存     | 建议 16GB+                   |

### 安装步骤

#### 方式一：使用 uv（推荐，快速开发）

```bash
# 1. 克隆仓库
git clone https://github.com/WarriorHanamy/learning_on_the_fly.git
cd learning_on_the_fly

# 2. 安装 uv（如果未安装）
pip install uv

# 3. 安装基础依赖
uv sync

# 4. 安装 GPU 支持
uv sync --extra cuda12

# 5. 可编辑安装
uv pip install --use-pep517 -e .

# 6. 运行测试
pytest
```

#### 方式二：使用 Conda（完整 ROS 环境）

```bash
# 1. 创建 Conda 环境
conda env create -f environment.yml
conda activate lotf

# 2. 可编辑安装
pip install --use-pep517 -e .

# 3. 运行测试
pytest
```

**注意**：`environment.yml` 使用 Python 3.9.18，而 `pyproject.toml` 要求 Python 3.10+，建议统一为 3.10。

### 快速开始

#### 1. 训练残差动力学模型

```bash
./bin/python_exec -m lotf residual \
  --config configs/residual_dynamics.yaml \
  --dataset examples/residual_dynamics/example_dataset.csv \
  --output checkpoints/residual_dynamics/my_model
```

#### 2. 训练状态悬停策略

```bash
./bin/python_exec -m lotf hover \
  --config configs/state_hovering.yaml \
  --output checkpoints/policy/my_hovering_policy
```

#### 3. 训练轨迹跟踪策略

```bash
./bin/python_exec -m lotf track \
  --config configs/traj_tracking.yaml \
  --checkpoint checkpoints/policy/my_tracking_policy
```

### Jupyter Notebook 示例

```bash
# 启动 Jupyter Lab
jupyter lab examples/state_hovering

# 按顺序执行笔记本：
# 1_train_base_policy.ipynb      - 训练基础策略
# 2_eval_base_policy.ipynb       - 评估基础策略
# 3_finetune_with_residual.ipynb  - 使用残差动力学微调
# 4_finetune_with_lora.ipynb     - 使用 LoRA 微调
```

---

## Architecture

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户接口层                            │
│  CLI (lotf command) | Python API | Jupyter Notebooks      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       配置管理层                             │
│  YAML 配置文件 | 数据类配置 (StateHoveringConfig)            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        训练脚本层                             │
│  train_state_hovering.py | train_traj_tracking.py            │
│  train_residual.py                                        │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│     算法层 (algos/)      │  │    环境层 (envs/)       │
│  BPTT 训练循环            │  │  HoveringStateEnv        │
│  损失函数计算             │  │  TrajTrackingStateEnv    │
│  梯度反向传播             │  │  HoveringFeaturesEnv     │
└──────────────────────────┘  └──────────────────────────┘
                │                       │
                │                       │
                ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      核心组件层                              │
│  神经网络模块 (modules/)  │  物理对象 (objects/)             │
│  - MLP                     │  - Quadrotor                    │
│  - ResidualDynamicsMLP     │  - ReferenceTraj               │
│  - LoraMLP                 │  - WorldBox                    │
└─────────────────────────────────────────────────────────────┘
                │                       │
                ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      基础设施层                              │
│  仿真 (simulation/)      │  传感器 (sensors/)                │
│  - model_rotor.py        │  - DoubleSphereCamera            │
│                          │  工具 (utils/)                   │
│                          │  - math, spaces, pytrees        │
│                          │  - random, lora                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      JAX 基础设施                             │
│  JIT 编译 | 自动微分 | GPU 加速 | PyTree 管理                 │
└─────────────────────────────────────────────────────────────┘
```

### 核心设计模式

1. **JAX 优先**：所有操作使用 JAX 进行 JIT 编译和自动微分
2. **PyTree 状态**：不可变状态容器，支持梯度传播
3. **向量化**：使用 `vmap` 和 `scan` 实现批量操作
4. **配置驱动**：YAML 配置文件实现可重复实验
5. **包装器模式**：模块化环境包装器（日志、向量化、归一化）
6. **集成学习**：多个残差模型用于不确定性量化
7. **两阶段训练**：
   - 阶段 1：从真实数据学习残差动力学
   - 阶段 2：在学习残差的可微分仿真中训练策略

### 数据流

```
CSV 数据集 → pandas 加载 → JAX 数组 → 向量化初始化 → 集成训练 → Orbax 检查点
                                                            │
                                                            ▼
Jupyter Notebook / CLI → 配置加载 → 环境创建 → BPTT 训练循环 → 策略检查点
```

---

## Data Layer

### 核心数据结构

#### QuadrotorState（四旋翼完整状态）
```python
@jdc.pytree_dataclass
class QuadrotorState:
    p: jax.Array              # 位置 [x, y, z] (世界坐标系)
    R: jax.Array              # 旋转矩阵 body→world (3×3)
    v: jax.Array              # 速度 [vx, vy, vz] (世界坐标系)
    omega: jax.Array          # 角速度 [wx, wy, wz] (机身坐标系)
    domega: jax.Array         # 角加速度 (机身坐标系)
    motor_omega: jax.Array    # 电机转速 [ω1, ω2, ω3, ω4]
    acc: jax.Array            # 线性加速度 (世界坐标系)
    res_acc_mean: jax.Array   # 预测的残差加速度
    dr_key: chex.PRNGKey      # 域随机化随机密钥
```

#### EnvTransition（环境转换容器）
```python
class EnvTransition(NamedTuple):
    state: TEnvState          # 下一状态（或重置状态）
    obs: jnp.ndarray          # 观测向量
    reward: jnp.ndarray       # 标量奖励
    terminated: jnp.ndarray   # 剧集终止标志
    truncated: jnp.ndarray    # 剧集截断标志
    info: Dict[str, Any]      # 额外元数据
```

#### ReferenceTraj（参考轨迹）
```python
@jdc.pytree_dataclass
class ReferenceTraj:
    ref_traj: jnp.array        # 路点数组 [num_points, 30]
    num_waypoints: int         # 路点数量
    pos_bounds: jnp.array      # [min_pos, max_pos]
    vel_bounds: jnp.array      # [min_vel, max_vel]

# 轨迹列（共30列）：
# - TIME[0:1], POS[1:4], QUAT[4:8], VEL[8:11],
# - OMEGA[11:14], ACC[14:17], COMMANDS[20:24], etc.
```

### 数据存储模式

#### 检查点结构（Orbax OCDBT 格式）
```
checkpoints/
├── policy/
│   ├── state_hovering_params/
│   ├── traj_tracking_params/
│   └── vision_hovering_params/
└── residual_dynamics/
    ├── dummy_params/
    │   ├── _CHECKPOINT_METADATA  # JSON: 时间戳、处理器信息
    │   ├── _METADATA            # JSON: PyTree 结构、数组形状
    │   ├── _sharding            # JSON: 设备分片信息
    │   ├── d/                   # 实际数组数据
    │   └── array_metadatas/
    └── example_params/
```

#### 数据集格式（CSV）
- **列数**: 24列（19个输入 + 3个输出）
- **输入**（19维）：位置(3) + 旋转矩阵(9) + 速度(3) + 推力(1) + 角速度(3)
- **输出**（3维）：残差加速度 [ax, ay, az]

### 数据流模式

#### 残差动力学训练
```
CSV 数据集 → pandas 加载 → JAX 数组 → 向量化初始化 → 集成训练 → Orbax 检查点
```
- 支持集成学习（多个模型）
- 使用 `jax.vmap` 并行化
- 损失函数：MSE + 谱正则化

#### 策略训练（BPTT）
```
初始化状态 → 环境步进扫描 → 累积奖励 → 梯度反向传播 → 参数更新
```
- 使用 `jax.lax.scan` 实现轨迹展开
- 通过时间反向传播
- 并行环境：`num_envs` 个环境同时训练

#### 仿真数据流
```
动作输入 → 延迟缓冲区 → 低层控制器 → 电机动力学 → 物理积分 → 状态更新
```
- 支持控制延迟模拟
- 双精度仿真：`high_fidelity`（RK4 积分）vs `low_fidelity`（简化动力学）

---

## Core Logic

### 主要服务和职责

#### 算法服务 (lotf/algos/)

**BPTT 训练 (bptt.py)**
- 主要职责：使用 BPTT 实现策略学习的主训练循环
- 核心函数：
  - `train()`: 使用 JIT 编编排导完整的训练过程
  - `epoch_fn()`: 执行单次 rollout、梯度计算和参数更新
  - `rollout()`: 在轨迹上模拟环境动力学
  - `loss_fn()`: 计算基于奖励的损失用于梯度下降
- 状态管理：
  - `RunnerState`: 容器，包含 train_state、env_state、观测、RNG 密钥和 epoch 索引
  - `TrajectoryState`: 保存 rollouts 期间收集的转换数据（奖励）
- 回调函数：按可配置间隔打印进度和梯度

#### 环境服务 (lotf/envs/)

**环境基类 (env_base.py)**
- 主要职责：所有 JAX 兼容环境的抽象基类
- 核心职责：
  - 提供 `step()` 方法，在终止/截断时自动重置
  - 定义 `reset()` 和 `_step()` 接口用于环境特定逻辑
  - 实现 `rollout()` 工具函数用于固定长度策略执行
  - 通过 `EnvTransition` namedtuple 管理状态转换

**具体环境**：
1. **HoveringStateEnv**: 训练四旋翼保持固定悬停位置
   - 包含控制延迟模拟（动作历史缓冲区）
   - 基于位置、速度和控制努力计算奖励

2. **TrajTrackingStateEnv**: 训练四旋翼跟踪预定义参考轨迹
   - 从 CSV 文件加载参考路径（circle、figure-8、star）
   - 比较状态与参考路点计算跟踪奖励

3. **HoveringFeaturesEnv**: 基于特征的变体，用于视觉控制

### 关键算法和计算

#### 物理仿真（两种保真度模式）

**低保真模式**（`simplified_dyn()`）:
```python
# 核心方程：
dvdt = gravity + R @ [0, 0, thrust/mass] + residual_acceleration
dpdt = velocity
# 积分：位置/速度使用 RK4
# 旋转：精确矩阵指数
```

**保真模式**（`_full_dyn()`）:
- **电机动力学**: 具有时间常数 `tau` 的一阶滞后
- **低层控制**: Betaflight 风格的体角速度 PD 控制器
- **转子增强**: 气动残差的多项式模型
- **完整状态**: 包括电机转速、角速度、角加速度
- **积分**: 所有状态变量的 RK4

#### 控制延迟模拟
- **机制**: 具有分数时间步的滚动动作缓冲区
- **实现**:
  ```python
  dt_1 = delay - (num_last_actions - 2) * dt
  dt_2 = dt - dt_1
  # 应用 action_1 持续 dt_1，然后应用 action_2 持续 dt_2
  ```

#### 残差动力学学习

**多项式残差模型**（`model_rotor.py`）:
- **力残差**: 速度、电机转速和交互项的多项式
- **力矩残差**: 体轴力矩的多项式
- **特征**: 线性、平方、立方速度项、电机均值耦合项、交叉速度交互项

**神经网络残差模型**（`residual_dynamics.py`）:
- **架构**: MLP [19 输入, 128, 128, 3 输出]
- **输入**: [位置(3), 旋转矩阵(9), 速度(3), 推力(1), 体角速度(3)]
- **输出**: 残差加速度 [ax, ay, az]
- **训练**: MSE 损失 + 谱正则化

#### 奖励计算

**悬停奖励**（`hovering_state_env.py`）:
```python
pos_cost = smooth_l1(sharpness * (position - goal))
vel_cost = 0.1 * smooth_l1(velocity)
omega_cost = 0.1 * smooth_l1(angular_velocity)
acc_cost = 0.1 * smooth_l1(acceleration)
action_cost = action_penalty_weight * smooth_l1(action - hover_action)
total_cost = pos_cost + vel_cost + omega_cost + acc_cost + action_cost
reward = -dt * total_cost
```

**轨迹跟踪奖励**（`traj_tracking_state_env.py`）:
```python
pos_target = reference_trajectory[target_idx].position
vel_target = reference_trajectory[target_idx].velocity
pos_cost = smooth_l1(position - pos_target)
vel_cost = smooth_l1(velocity - vel_target)
tracking_cost = pos_cost + vel_cost
reward = -dt * (tracking_cost + action_cost)
```

### 业务规则和验证

#### 物理约束
- **推力限制**: 每个电机 `thrust_min` 到 `thrust_max`
- **体角速度限制**: `omega_max` [10.0, 10.0, 4.0] rad/s
- **电机转速范围**: `motor_omega_min` (150) 到 `motor_omega_max` (2800) rad/s
- **延迟非负**: 断言 `delay >= 0.0`
- **时间步对齐**: 高保真要求 `dt` 是 `dt_low_level` 的倍数

#### 状态验证
- **位置边界**: 通过 `WorldBox.contains(position)` 检查
- **碰撞检测**: 当超出边界时返回 `terminated`
- **观测归一化**: 确保归一化前没有无限边界
- **轨迹索引裁剪**: 防止越界参考访问

#### 训练配置
- **学习率调度**: epochs 上余弦衰减
- **梯度监控**: 记录最大梯度绝对值
- **集成训练**: 支持多个残差模型用于不确定性估计
- **LoRA 适配**: 使用低秩参数更新微调策略

### 状态管理模式

#### PyTree 状态
- 所有状态使用 `@jdc.pytree_dataclass` 以实现 JAX 兼容性
- 启用通过状态转换的自动梯度传播
- 支持用于条件状态更新的 `tree_select()`

#### 转换模式
```python
EnvTransition = NamedTuple(
    state: TEnvState,           # 下一环境状态（或重置状态）
    obs: jnp.ndarray,           # 观测向量
    reward: jnp.ndarray,        # 标量奖励
    terminated: jnp.ndarray,    # 如果达到目标或碰撞则为 True
    truncated: jnp.ndarray,     # 如果超过最大步数则为 True
    info: Dict[str, Any]        # 额外元数据
)
```

#### 自动重置逻辑
```python
done = jnp.logical_or(terminated, truncated)
state = tree_select(done, reset_state, step_state)
obs = tree_select(done, reset_obs, step_obs)
```

#### 向量化环境
- `VecEnv` 使用 `jax.vmap` 包装基础环境以进行并行执行
- 每个环境实例独立的 RNG 密钥
- 在 `num_envs` 维度上批量状态管理

### 残差动力学学习逻辑

#### 数据收集阶段
1. **飞行真实四旋翼**: 从硬件收集状态-动作-残差三元组
2. **计算残差**: `residual = actual_acceleration - nominal_acceleration`
3. **存储数据集**: CSV 格式，[state, action] 输入和残差输出

#### 模型架构
```python
ResidualDynamicsMLP(
    feature_list=[19, 128, 128, 3],  # 输入、隐藏1、隐藏2、输出
    nonlinearity=nn.relu,
    initial_scale=1.0
)
```

#### 训练流程
```python
# 1. 加载数据集
X, y = load_dataset(path, input_dim=19)  # X: 状态+动作, y: 残差

# 2. 创建集成（默认3个模型）
model_params, train_states = create_vec_funcs().init_fn(learning_rate, seeds)

# 3. 使用 MSE + 谱正则化训练
train_states = train_fn(
    train_states, X, y,
    lambda_reg=0.001,
    num_epochs=100
)

# 4. 保存检查点
save_checkpoint("checkpoints/residual_dynamics/residual_params", train_states.params)
```

---

## API Reference

### CLI 接口

#### 入口点
**文件**: `lotf/__main__.py`

#### 全局命令
```bash
./bin/python_exec -m lotf --help              # 显示所有命令
./bin/python_exec -m lotf --version           # 显示包版本
./bin/python_exec -m lotf --list-configs      # 列出可用的 YAML 配置
```

#### 子命令

**1. 残差动力学训练**
```bash
./bin/python_exec -m lotf residual \
  --config configs/residual_dynamics.yaml \
  --dataset path/to/dataset.csv \
  --output checkpoints/residual_dynamics/residual_params
```

**2. 状态悬停训练**
```bash
./bin/python_exec -m lotf hover \
  --config configs/state_hovering.yaml \
  --output checkpoints/policy/state_hovering_params
```

**3. 轨迹跟踪训练**
```bash
./bin/python_exec -m lotf track \
  --config configs/traj_tracking.yaml \
  --checkpoint checkpoints/policy/traj_tracking_params \
  --trajectory-output outputs/trajectory.csv
```

### Python API

#### 环境接口

```python
from lotf.envs.env_base import Env, EnvState, EnvTransition, rollout

# 创建环境
env = HoveringStateEnv(
    max_steps_in_episode=10000,
    dt=0.02,
    delay=0.02,
    hover_target=[0.0, 0.0, 1.0]
)

# 重置环境
key = jax.random.key(0)
state, obs = env.reset(key)

# 执行步骤
transition = env.step(state, action, res_model_params, key)

# 执行 rollout
transitions = rollout(env, key, policy, res_model_params, state)
```

#### 算法接口（BPTT 训练）

```python
from lotf.algos import bptt

result = bptt.train(
    env: Env,                        # 环境实例
    env_state: EnvState,             # 初始环境状态
    obs: jax.Array,                 # 初始观测
    train_state: TrainState,         # Flax TrainState（包含策略）
    num_epochs: int,                # 训练 epochs 数
    num_steps_per_epoch: int,       # 每个 epoch 的 rollout 长度
    num_envs: int,                  # 并行环境数
    res_model_params: FrozenDict,    # 残差动力学参数
    key: chex.PRNGKey,              # RNG 密钥
)

# 返回:
#   {"runner_state": RunnerState, "metrics": losses}
```

#### 残差动力学接口

```python
from lotf.utils.residual_dynamics import (
    create_vec_funcs,
    get_residual_dyn_model_apply_fn
)

# 向量化函数用于集成训练
init_fn, train_fn, apply_fn = create_vec_funcs()

# 初始化集成
model_params, train_states = init_fn(
    learning_rate=0.01,
    seeds=jnp.arange(num_models)  # 每个模型不同的种子
)

# 训练集成
train_states = train_fn(
    train_states,
    X,                    # 输入特征 (N, 19)
    y,                    # 目标残差 (N, 3)
    lambda_reg=0.001,
    num_epochs=100,
)

# 获取向量化应用函数（用于集成预测）
parallel_apply_fn = get_residual_dyn_model_apply_fn()
residuals = parallel_apply_fn(model_params, x)
```

#### 四旋翼物理接口

```python
from lotf.objects import Quadrotor, QuadrotorState

# 创建四旋翼实例
quad = Quadrotor(
    drone_name="example_quad",
    mass=0.75,
    inertia=jnp.array([0.002410, 0.001800, 0.003759]),
    motor_omega_min=150.0,
    motor_omega_max=2800.0,
    motor_tau=0.033,
    omega_max=jnp.array([10.0, 10.0, 4.0]),
    dt_low_level=0.001,
    sim_dyn_config={
        "use_high_fidelity": False,
        "use_forward_residual": False,
    }
)

# 步进四旋翼动力学
next_quad_state = quad.step(
    state=QuadrotorState(...),
    thrust=9.81 * mass,
    omega=jnp.array([0.0, 0.0, 0.0]),
    res_model_params=FrozenDict({...}),
    dt=0.02
)
```

#### 策略网络接口

```python
from lotf.modules import MLP, LoraMLP, ResidualDynamicsMLP

# 标准 MLP 策略
policy = MLP(
    feature_list=[obs_dim, 512, 512, action_dim],
    nonlinearity=nn.relu,
    initial_scale=0.01,
)

# 初始化参数
key = jax.random.key(0)
params = policy.initialize(key)

# 前向传播
action = policy.apply(params, observation)

# 使用 Flax TrainState
from flax.training.train_state import TrainState
import optax

tx = optax.adam(learning_rate=0.005)
train_state = TrainState.create(
    apply_fn=policy.apply,
    params=params,
    tx=tx
)

# 更新参数
train_state = train_state.apply_gradients(grads=grads)
```

#### 检查点接口

```python
from orbax.checkpoint import PyTreeCheckpointer

# 保存检查点
ckptr = PyTreeCheckpointer()
ckptr.save("checkpoints/policy/my_policy", train_state.params)

# 加载检查点
ckptr = PyTreeCheckpointer()

# 加载策略
policy_params = ckptr.restore("checkpoints/policy/state_hovering_params")

# 加载残差动力学集成
residual_params = ckptr.restore("checkpoints/residual_dynamics/example_params")
```

---

## Testing

### 测试框架

- **pytest** (8.4.1): 核心测试框架
- **pytest-cov** (6.2.1): 代码覆盖率测量
- **pytest-repeat** (0.9.4): 重复运行测试
- **pytest-rerunfailures** (15.1): 测试失败自动重试

### 测试文件组织

```
tests/
├── test_main.py                    # CLI 入口点测试
├── configs/
│   └── test_config_loader.py      # 配置加载器测试
└── scripts/
    ├── test_train_residual.py      # 残差动力学训练脚本测试
    ├── test_train_state_hovering.py # 状态悬停训练脚本测试
    └── test_train_traj_tracking.py  # 轨迹跟踪训练脚本测试
```

### 测试模式

#### 单元测试
- **配置测试**: 数据类字段、YAML 加载、配置合并
- **CLI 测试**: 参数解析、子命令分发、版本/帮助标志
- **脚本测试**: 各训练脚本的组件（环境创建、策略创建、参数解析）

#### 集成测试
使用 `@pytest.mark.integration` 标记，但当前因 GPU 依赖跳过：
```python
@pytest.mark.integration
@pytest.mark.skip(reason="Integration test requires GPU environment to load dummy checkpoint")
class TestIntegration:
    """完整训练流水线的集成测试。"""
```

### Mocking 策略

- **`unittest.mock.MagicMock`**: 创建模拟对象
- **`unittest.mock.patch`**: 替换对象和函数
- **`sys.argv` patching**: 模拟命令行参数

### 运行测试

```bash
# 运行所有测试
pytest

# 运行带覆盖率的测试
pytest --cov=lotf --cov-report=html

# 跳过集成测试
pytest -m "not integration"

# 运行特定测试文件
pytest tests/test_main.py
```

### 覆盖率状态

| 模块                         | 测试文件                     | 覆盖组件                             |
| ---------------------------- | ---------------------------- | ------------------------------------ |
| `__main__`                     | `test_main.py`                 | CLI 解析、版本、配置列表、子命令分发 |
| `configs/loader`               | `test_config_loader.py`        | YAML 加载、验证、合并                |
| `scripts/train_residual`       | `test_train_residual.py`       | 数据集加载、集成创建、参数解析       |
| `scripts/train_state_hovering` | `test_train_state_hovering.py` | 配置、环境、策略、CLI、检查点        |
| `scripts/train_traj_tracking`  | `test_train_traj_tracking.py`  | 配置、环境、策略、CLI、轨迹导出      |

---

## Deployment

### 环境配置

#### Conda 环境（完整 ROS 环境）

```bash
# 创建 Conda 环境
conda env create -f environment.yml
conda activate lotf

# 可编辑安装
pip install --use-pep517 -e .
```

#### uv 包管理器（快速开发）

```bash
# 安装基础依赖
uv sync

# 安装 GPU 支持
uv sync --extra cuda12

# 可编辑安装
uv pip install --use-pep517 -e .
```

### 部署工作流示例

```bash
# 1. 环境准备
conda env create -f environment.yml
conda activate lotf
uv sync

# 2. 训练残差动力学模型
./bin/python_exec -m lotf residual \
  --config configs/residual_dynamics.yaml \
  --dataset examples/residual_dynamics/example_dataset.csv \
  --output checkpoints/residual_dynamics/my_model

# 3. 训练悬停策略
./bin/python_exec -m lotf hover \
  --config configs/state_hovering.yaml \
  --output checkpoints/policy/my_hovering_policy

# 4. 训练轨迹跟踪
./bin/python_exec -m lotf track \
  --config configs/traj_tracking.yaml \
  --checkpoint checkpoints/policy/my_tracking_policy
```

### 监控和日志

#### 日志策略

1. **打印语句（标准输出）**：
   ```python
   print(f"Loading configuration from: {args.config}")
   print(f"Environment info:")
   print(f"  action_dim: {action_dim}")
   ```

2. **JAX Debug 回调**：
   ```python
   def progress_callback_host(episode_loss):
       episode, loss = episode_loss
       print(f"Episode: {episode}, Loss: {loss:.2f}")

   def progress_callback(episode, loss):
       jax.lax.cond(
           pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
           true_fun=lambda eps_lss: jax.debug.callback(progress_callback_host, eps_lss),
           false_fun=lambda eps_lss: None,
           operand=(episode, loss),
       )
   ```

3. **环境包装器日志**：
   - `LogWrapper`: 记录剧集奖励和长度
   - `info` 字典包含：`returned_episode_returns`、`returned_episode_lengths`、`timestep`、`returned_episode`

#### 监控指标

- **训练指标**: 损失值、梯度范数、训练时间
- **环境指标**: 剧集返回值、剧集长度、时间步数
- **残差动力学指标**: MSE 损失、总损失、谱范数

### CI/CD 流水线

**当前状态**: 未配置

**建议**: 添加 GitHub Actions 工作流
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          conda env create -f environment.yml
          conda activate lotf
          pip install --use-pep517 -e .
      - name: Run tests
        run: pytest
```

---

## Dependencies

### 核心运行时依赖

#### JAX 生态系统
| 包名                  | 版本约束   | 说明                       |
| --------------------- | ---------- | -------------------------- |
| jax                   | >=0.4.30   | 自动微分框架，核心计算引擎 |
| jaxlib                | >=0.4.30   | JAX 的底层 XLA 编译接口    |
| flax                  | >=0.8.5    | JAX 上的神经网络库         |
| optax                 | >=0.2.4    | 优化器库                   |
| orbax-checkpoint      | >=0.6.4    | 模型检查点管理工具         |
| chex                  | >=0.1.90   | 测试和调试工具             |
| jax-dataclasses       | >=1.6.2    | 数据类增强，用于 PyTree    |

#### 科学计算与可视化
| 包名       | 版本约束   | 说明           |
| ---------- | ---------- | -------------- |
| numpy      | >=1.26.4   | 数值计算基础库 |
| scipy      | >=1.13.1   | 科学计算高级函数 |
| pandas     | 未指定     | 数据分析       |
| matplotlib | 未指定     | 绘图可视化     |
| seaborn    | 未指定     | 统计数据可视化 |
| pyyaml     | 未指定     | YAML 配置解析  |
| tqdm       | 未指定     | 进度条显示     |

#### CUDA / GPU 加速依赖
```
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cudnn-cu12==9.12.0.46
nvidia-cublas-cu12==12.9.1.4
nvidia-cufft-cu12==11.4.1.4
nvidia-cusolver-cu12==11.7.5.82
nvidia-cusparse-cu12==12.5.10.65
nvidia-nccl-cu12==2.27.7
nvidia-nvjitlink-cu12==12.9.86
jax-cuda12-pjrt==0.4.30
jax-cuda12-plugin==0.4.30
```

#### ROS Noetic 依赖
从 `robostack-staging` 通道安装大量 ROS 包（约 180+ 个包）：
- **核心 ROS**: `ros-noetic-catkin`, `ros-noetic-rospack`, `ros-noetic-rospy`, `ros-noetic-roscpp`
- **机器人相关**: `ros-noetic-robot-state-publisher`, `ros-noetic-urdf`, `ros-noetic-control-msgs`
- **视觉与传感器**: `ros-noetic-cv-bridge`, `ros-noetic-sensor-msgs`, `ros-noetic-image-transport`
- **可视化工具**: `ros-noetic-rviz`, `ros-noetic-interactive-markers`
- **工具链**: `catkin_tools`, `colcon-*` 系列

#### 物理仿真与图形依赖
| 包名     | 版本    | 说明                |
| -------- | ------- | ------------------- |
| ogre     | 1.10.12 | 3D 渲染引擎        |
| assimp   | 5.2.5   | 3D 模型导入库      |
| opencv   | 4.6.0   | 计算机视觉库       |
| boost    | 1.78.0  | C++ 高性能库       |
| eigen    | 3.4.0   | 线性代数库         |
| orocos-kdl | 1.5.1  | 运动学动力学库     |

### 开发依赖

#### 测试框架
| 包名                 | 版本约束 | 说明             |
| -------------------- | -------- | ---------------- |
| pytest               | >=8.0    | Python 测试框架  |
| pytest-cov           | >=6.0    | 代码覆盖率插件   |
| pytest-repeat        | >=0.9    | 重复执行测试插件 |
| pytest-rerunfailures | >=15.0   | 失败重试插件     |

#### 开发工具
- **ruff**: Python 代码格式化和 linting（line-length: 100）
- **coverage**: 代码覆盖率工具
- **jupyterlab**: Jupyter Lab IDE

### 依赖层级总结

```
lotf (0.1.0)
├── ML Core
│   ├── jax>=0.4.30
│   ├── jaxlib>=0.4.30
│   ├── flax>=0.8.5
│   └── optax>=0.2.4
├── Scientific Computing
│   ├── numpy>=1.26.4
│   ├── scipy>=1.13.1
│   └── pandas, matplotlib, seaborn
├── CUDA Support (optional)
│   └── jax[cuda12]>=0.4.30
└── ROS Ecosystem
    ├── ros-noetic-* (~180 packages)
    ├── catkin_tools
    └── colcon-* suite
```

### 注意事项

1. **Python 版本不一致**: `environment.yml` 使用 3.9.18，`pyproject.toml` 要求 3.10+，建议统一为 3.10
2. **ROS 与 Python 3.x**: 通过 robostack-staging 提供，与官方 ROS Noetic 不同
3. **CUDA 支持**: 需要 CUDA 12 兼容的 GPU
4. **Conda 频道**: 同时使用 conda-forge 和 robostack-staging，可能存在包冲突风险

---

## Domain Glossary

### 核心业务领域术语

#### Differentiable Simulation（可微仿真）
支持梯度反向传播的物理仿真，允许通过仿真轨迹优化控制策略。与传统黑箱仿真不同，可微仿真提供了损失函数相对于控制参数的解析梯度。

#### Residual Dynamics（残差动力学）
使用神经网络学习物理模型中未建模的部分。残差 = 真实加速度 - 名义加速度。学习残差可以弥补 sim-to-real 差距，提高策略在真实硬件上的性能。

#### Quadrotor（四旋翼）
具有四个旋翼的无人航空器（UAV），采用 X 型配置。每个旋翼产生推力和力矩，通过差动推力实现姿态控制。本项目使用四旋翼作为主要研究对象。

#### Sim-to-Real Gap（仿真到现实差距）
在仿真中训练的策略部署到真实硬件时性能下降的现象。主要原因包括：
- 未建模的物理效应（空气阻力、电机延迟、振动）
- 传感器噪声和偏差
- 环境差异（风、光照、纹理）

#### Backpropagation Through Time (BPTT)（通过时间反向传播）
通过展开时间序列并反向传播梯度来优化策略的方法。BPTT 可以直接优化长期累积奖励，适用于轨迹优化和策略学习。

#### Ensemble Learning（集成学习）
训练多个相同结构的模型（集成）以预测残差动力学。集成平均可以减少方差，提供不确定性估计，提高鲁棒性。

#### Low-Rank Adaptation (LoRA)（低秩适应）
一种参数高效的微调方法，通过添加低秩矩阵来适应新任务。LoRA 只训练少量参数，避免灾难性遗忘，同时保持原模型知识。

### 领域实体和关系

#### 物理实体层次
```
Quadrotor（四旋翼）
├── Motors（电机）
│   ├── Motor Dynamics（电机动力学）
│   └── Thrust/Torque Generation（推力/力矩生成）
├── Rotors（旋翼）
│   └── Aerodynamics（空气动力学）
├── Low-Level Controller（低层控制器）
│   └── Body-Rate PD Controller（体角速度 PD 控制器）
└── Sensors（传感器）
    ├── IMU（惯性测量单元）
    └── Camera（相机）
```

#### 环境层次
```
Env（环境基类）
├── HoveringStateEnv（状态悬停环境）
├── TrajTrackingStateEnv（轨迹跟踪环境）
└── HoveringFeaturesEnv（特征悬停环境）
    └── DoubleSphereCamera（双球面相机）
```

#### 神经网络模块层次
```
MLP（多层感知机）
├── Policy MLP（策略网络）
├── ResidualDynamicsMLP（残差动力学网络）
└── LoraMLP（LoRA 适配网络）
```

#### 参考轨迹实体
```
ReferenceTraj（参考轨迹）
├── Circle（圆形）
├── Figure-8（8字形）
└── Star（星形）
```

### 行业特定模式

#### 控制理论
- **Betaflight 风格控制**: 开源飞控软件使用的控制架构
- **RK4 积分**: 四阶龙格-库塔法，用于数值积分
- **控制延迟处理**: 使用动作缓冲区模拟真实系统延迟

#### 机器人学
- **分配矩阵**: 将体轴推力/力矩映射到电机推力的矩阵
- **域随机化**: 在仿真中随机化物理参数以提高泛化能力
- **Smooth L1 损失**: 结合 L1 和 L2 优点的损失函数

#### 机器学习
- **自定义 JVP**: 为不稳定动力学定义自定义正向/反向梯度
- **JAX JIT 编译**: 使用 XLA 编译器加速计算
- **PyTree 注册**: 使自定义数据结构与 JAX 兼容

#### 计算机视觉
- **双球相机模型**: 更精确的相机投影模型，减少畸变

### 关键抽象和隐喻

#### "Learning on the Fly" 隐喻
飞行员在飞行中实时调整控制的隐喻，表示策略可以适应不断变化的动力学。

#### 混合 DiffSim（可微仿真）
正向传播使用高保真物理（电机动力学、残差），反向传播使用简化动力学（解析形式）以提高稳定性。

#### 残差作为"误差校正"
残差动力学就像 GPS+IMU 融合，残差模型校正名义模型的误差，提高预测精度。

#### 集成作为"专家委员会"
集成学习就像多个专家投票，减少个体模型的偏见和方差。

#### LoRA 作为"适配器模式"
LoRA 就像电源适配器，将预训练模型"插到"新任务上，而不改变原模型。

### 项目特定术语缩写

| 缩写   | 全称                               | 说明                       |
| ------ | ---------------------------------- | -------------------------- |
| BPTT   | Backpropagation Through Time       | 通过时间反向传播           |
| LoRA   | Low-Rank Adaptation                | 低秩适应                   |
| JAX    | Just Another eXpression            | 自动微分框架               |
| JIT    | Just-In-Time Compilation           | 即时编译                   |
| JVP    | Jacobian-Vector Product            | 雅可比-向量乘积           |
| VJP    | Vector-Jacobian Product            | 向量-雅可比乘积           |
| vmap   | Vectorized Map                     | 向量化映射                 |
| scan   | Scan operation                     | 扫描操作（类似 foldl）    |
| MSE    | Mean Squared Error                 | 均方误差                   |
| RK4    | Runge-Kutta 4th Order              | 四阶龙格-库塔法            |
| PD     | Proportional-Derivative Controller | 比例-微分控制器            |
| IMU    | Inertial Measurement Unit          | 惯性测量单元               |
| UAV    | Unmanned Aerial Vehicle            | 无人航空器                 |
| CI/CD  | Continuous Integration/Deployment | 持续集成/部署             |
| ROS    | Robot Operating System             | 机器人操作系统             |

---

## Documentation Index

### 主要文档

| 文档       | 路径                 | 说明                           |
| ---------- | -------------------- | ------------------------------ |
| README     | README.md            | 项目概述、安装、快速开始       |
| USAGE      | USAGE.md             | 详细的 CLI 使用指南            |
| AGENTS     | AGENTS.md            | 仓库开发指南                   |
| CODEBASE   | CODEBASE.md          | 本文档（代码库文档）           |

### 子模块 README

| 模块 README | 路径                        | 说明               |
| ----------- | --------------------------- | ------------------ |
| 算法目录   | lotf/algos/README.md         | BPTT 算法         |
| 环境目录   | lotf/envs/README.md         | 悬停和轨迹跟踪环境 |
| 神经网络   | lotf/modules/README.md       | MLP、LoRA、残差   |
| 对象目录   | lotf/objects/README.md      | 四旋翼、参考轨迹   |
| 传感器目录 | lotf/sensors/README.md      | 双球相机模型      |
| 仿真目录   | lotf/simulation/README.md    | 旋翼动力学        |

### 场景文档（TDD 方法）

| 场景文档                        | 路径                                 | 说明                   |
| ------------------------------- | ------------------------------------ | ---------------------- |
| CLI 模块执行                    | docs/scenario/cli-module-executable.md | CLI 可执行性测试       |
| CLI 参数解析                    | docs/scenario/cli-argument-parsing.md  | 参数解析行为           |
| 创建轨迹跟踪环境                | docs/scenario/create-env-traj-tracking.md | 环境创建              |
| 参考轨迹加载                    | docs/scenario/reference-trajectory-loading.md | 轨迹文件加载          |
| 训练循环                        | docs/scenario/training-loop.md          | 训练过程              |
| 轨迹导出                        | docs/scenario/trajectory-export.md      | CSV 导出              |
| 创建集成模型                    | docs/scenario-train-residual/create-ensemble.md | 集成创建              |
| 加载数据集                      | docs/scenario-train-residual/load-dataset.md | 数据集加载            |

### TDD 总结文档

| TDD 总结            | 路径                           | 说明                     |
| ------------------- | ------------------------------ | ------------------------ |
| Step 1-7            | tdd-summary/step-*.md          | 轨迹跟踪训练脚本开发过程 |

### Jupyter Notebook 示例

| 示例                      | 路径                                      | 说明           |
| ------------------------- | ----------------------------------------- | -------------- |
| 残差动力学                | examples/residual_dynamics/               | 1 个笔记本     |
| 状态悬停                  | examples/state_hovering/                  | 4 个笔记本     |
| 轨迹跟踪                  | examples/traj_tracking/                   | 4 个笔记本     |
| 视觉悬停                  | examples/vision_hovering/                 | 5 个笔记本     |

### 配置文件

| 配置文件                    | 路径                           | 说明                   |
| --------------------------- | ------------------------------ | ---------------------- |
| 状态悬停配置                | configs/state_hovering.yaml    | 带详细注释             |
| 轨迹跟踪配置                | configs/traj_tracking.yaml     | 轨迹跟踪训练           |
| 视觉悬停配置                | configs/vision_hovering.yaml   | 视觉悬停训练           |
| 残差动力学配置              | configs/residual_dynamics.yaml | 残差动力学集成训练     |

---

## Citation

如果您使用此代码，请引用：

```bibtex
@inproceedings{pan2024learning,
  title={Learning on the Fly: Differentiable Quadrotor Simulation},
  author={Pan, Michael and ...},
  booktitle={...},
  year={2024}
}
```

---

## License

MIT License - 详见 LICENSE 文件

---

## Contributing

请参阅 AGENTS.md 了解：
- 项目结构和模块组织
- 构建、测试和开发命令
- 编码风格和命名约定
- 测试指南
- 提交和 PR 指南

---

## Support

如有问题或建议，请：
1. 提交 GitHub Issue
2. 联系作者：michael.pan31415@gmail.com
3. 查阅 AGENTS.md 和 USAGE.md

---

*最后更新：2026-02-25*
