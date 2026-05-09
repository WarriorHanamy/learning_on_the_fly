"""Shared trajectory tracking configuration and builder functions.

Used by ``train_traj_tracking``, ``visualize_policy``, and ``evaluate_policy``
so that environment/policy construction is defined in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax
import optax
import yaml
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer

from lotf import LOTF_ROOT, resolve_path
from lotf.forward_model_config import ForwardModelConfig
from lotf.modules import MLP
from lotf.objects import Quadrotor
from lotf.schemes import build_scheme
from lotf.schemes.configs import (
    QuadrotorParams,
    SimplestConfig,
    ResAccConfig,
    ApproxConfig,
    ApproxResAccConfig,
    InnerLoopConfig,
    FullConfig,
)

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PolicyNetConfig:
    """Policy network architecture configuration."""

    hidden_layers: list[int] = field(default_factory=lambda: [512, 512])
    initial_scale: float = 0.01


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    initial_lr: float = 0.001
    scheduler: str = "cosine_decay"


@dataclass
class TrajTrackingConfig:
    """Complete configuration for trajectory tracking training/evaluation.

    Attributes:
        seed: Random seed for reproducibility.
        num_envs: Number of parallel environments (training only).
        max_epochs: Maximum number of training epochs (training only).
        sim_dt: Simulation time step [s].
        max_sim_time: Maximum simulation time per episode [s].
        delay: Action delay [s].
        ref_traj_name: Reference trajectory name (CIRCLE, FIG8, STAR).
        skip_start: Skip initial speedup portion of trajectory.
        forward_model_config: Forward model fidelity flags.
        yaw_scale: Yaw randomization scale.
        pitch_roll_scale: Pitch/roll randomization scale.
        position_std: Position noise standard deviation.
        velocity_std: Velocity noise standard deviation.
        omega_std: Angular velocity noise standard deviation.
        policy_net: Policy network configuration.
        optimizer: Optimizer configuration.
    """

    seed: int = 0
    num_envs: int = 300
    max_epochs: int = 300
    sim_dt: float = 0.02
    max_sim_time: float = 5.0
    delay: float = 0.04
    ref_traj_name: str = "fig8"
    skip_start: bool = True
    forward_model_config: ForwardModelConfig = field(default_factory=ForwardModelConfig)
    scheme_name: str = "simplest"
    scheme_config: dict = field(default_factory=dict)
    yaw_scale: float = 0.1
    pitch_roll_scale: float = 0.1
    position_std: float = 0.1
    velocity_std: float = 0.1
    omega_std: float = 0.1
    policy_net: PolicyNetConfig = field(default_factory=PolicyNetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrajTrackingConfig:
        path = Path(path)
        if not path.is_absolute():
            path = LOTF_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:
            raise ValueError(f"Configuration file is empty: {path}")

        fwd_dict = raw_config.get("forward_model_config", {})
        forward_model_config = ForwardModelConfig.from_dict(fwd_dict)

        # scheme-based config (new approach)
        scheme_name = raw_config.get("scheme", "simplest")
        scheme_config = raw_config.get("scheme_config", {})

        policy_net_dict = raw_config.get("policy_net", {})
        policy_net = PolicyNetConfig(
            hidden_layers=policy_net_dict.get("hidden_layers", [512, 512]),
            initial_scale=policy_net_dict.get("initial_scale", 0.01),
        )

        optimizer_dict = raw_config.get("optimizer", {})
        optimizer = OptimizerConfig(
            initial_lr=optimizer_dict.get("initial_lr", 0.001),
            scheduler=optimizer_dict.get("scheduler", "cosine_decay"),
        )

        return cls(
            seed=raw_config.get("seed", 0),
            num_envs=raw_config.get("num_envs", 300),
            max_epochs=raw_config.get("max_epochs", 300),
            sim_dt=raw_config.get("sim_dt", 0.02),
            max_sim_time=raw_config.get("max_sim_time", 5.0),
            delay=raw_config.get("delay", 0.04),
            ref_traj_name=raw_config.get("ref_traj_name", "FIG8"),
            skip_start=raw_config.get("skip_start", True),
            forward_model_config=forward_model_config,
            scheme_name=scheme_name,
            scheme_config=scheme_config,
            yaw_scale=raw_config.get("yaw_scale", 0.1),
            pitch_roll_scale=raw_config.get("pitch_roll_scale", 0.1),
            position_std=raw_config.get("position_std", 0.1),
            velocity_std=raw_config.get("velocity_std", 0.1),
            omega_std=raw_config.get("omega_std", 0.1),
            policy_net=policy_net,
            optimizer=optimizer,
        )


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def build_quadrotor(config: TrajTrackingConfig) -> Quadrotor:
    """Create a Quadrotor instance using scheme-based architecture.

    Reads scheme name from config, loads QuadrotorParams from YAML,
    and constructs the bridge.
    """
    # load physical params
    import os

    lotf_obj_dir = os.path.join(LOTF_ROOT, "lotf", "objects")
    param_path = os.path.join(lotf_obj_dir, "quadrotor_files", "example_quad.yaml")
    with open(param_path) as f:
        param_dict = yaml.safe_load(f)
    params = QuadrotorParams(
        mass=param_dict["mass"],
        tbm_fr=tuple(param_dict["tbm_fr"]),
        tbm_bl=tuple(param_dict["tbm_bl"]),
        tbm_br=tuple(param_dict["tbm_br"]),
        tbm_fl=tuple(param_dict["tbm_fl"]),
        inertia=tuple(param_dict["inertia"]),
        motor_omega_min=param_dict.get("motor_omega_min", 150.0),
        motor_omega_max=param_dict.get("motor_omega_max", 2800.0),
        motor_tau=param_dict.get("motor_tau", 0.033),
        motor_inertia=param_dict.get("motor_inertia", 5.64e-6),
        omega_max=tuple(param_dict.get("omega_max", [10.0, 10.0, 4.0])),
        thrust_map=tuple(param_dict.get("thrust_map", [1.562522e-6, 0.0, 0.0])),
        kappa=param_dict.get("kappa", 0.022),
        thrust_min=param_dict.get("thrust_min", 0.0),
        thrust_max=param_dict.get("thrust_max", 8.5),
        rotors_config=param_dict.get("rotors_config", "cross"),
    )

    # determine scheme name from config
    scheme_name = config.scheme_name
    if scheme_name == "simplest" and config.forward_model_config != ForwardModelConfig():
        # backward compat: infer from old ForwardModelConfig
        from lotf.forward_model_config import infer_setting_name

        try:
            scheme_name = infer_setting_name(config.forward_model_config)
        except ValueError:
            scheme_name = "simplest"

    # ensure approx_path is populated from forward_model_config if available
    sc = dict(config.scheme_config)
    fwd = config.forward_model_config
    if fwd.inner_loop_approx_path and "chirp_path" not in sc:
        sc["chirp_path"] = fwd.inner_loop_approx_path

    # build scheme config from name
    if scheme_name == "simplest":
        scheme_cfg = SimplestConfig()
    elif scheme_name == "resacc":
        scheme_cfg = ResAccConfig()
    elif scheme_name == "approx":
        approx_path = sc.get("chirp_path", "audit/default_inner_loop_approx.json")
        scheme_cfg = ApproxConfig(chirp_path=approx_path)
    elif scheme_name == "approx_resacc":
        approx_path = sc.get("chirp_path", "audit/default_inner_loop_approx.json")
        scheme_cfg = ApproxResAccConfig(chirp_path=approx_path)
    elif scheme_name in ("inner_loop", "innerloop"):
        scheme_cfg = InnerLoopConfig(
            dt_low_level=sc.get("dt_low_level", 0.001),
            rotor_augmentation_path=sc.get("rotor_augmentation_path"),
            kp=tuple(sc.get("kp", [40.0, 40.0, 30.0])),
            kd=tuple(sc.get("kd", [20.0, 20.0, 0.0])),
        )
        scheme_name = "inner_loop"
    elif scheme_name == "full":
        scheme_cfg = FullConfig(
            dt_low_level=sc.get("dt_low_level", 0.001),
            rotor_augmentation_path=sc.get("rotor_augmentation_path"),
            kp=tuple(sc.get("kp", [40.0, 40.0, 30.0])),
            kd=tuple(sc.get("kd", [20.0, 20.0, 0.0])),
        )
    else:
        raise ValueError(f"Unknown scheme name: {scheme_name}")

    # load residual params if needed
    res_params = None
    if scheme_name in ("resacc", "approx", "approx_resacc", "full"):
        from orbax.checkpoint import PyTreeCheckpointer

        ckpt_path = sc.get("residual_checkpoint", "checkpoints/residual_dynamics/dummy_params")
        ckpt_path = resolve_path(ckpt_path)
        ckptr = PyTreeCheckpointer()
        res_params = ckptr.restore(str(ckpt_path))

    scheme = build_scheme(scheme_cfg, params, res_params)
    return Quadrotor(scheme, params)


def build_traj_tracking_env(
    config: TrajTrackingConfig,
    *,
    with_log_wrapper: bool = True,
    with_vec_wrapper: bool = True,
):
    """Create and wrap a TrajTrackingStateEnv.

    Args:
        config: Trajectory tracking configuration.
        with_log_wrapper: Apply LogWrapper for episode stats tracking (training).
        with_vec_wrapper: Apply VecEnv for parallel batch (training).

    Returns:
        Wrapped environment.
    """
    from lotf.envs import TrajTrackingStateEnv
    from lotf.envs.wrappers import LogWrapper, MinMaxObservationWrapper, VecEnv

    quad_obj = build_quadrotor(config)

    env = TrajTrackingStateEnv(
        max_steps_in_episode=int(config.max_sim_time / config.sim_dt),
        dt=config.sim_dt,
        delay=config.delay,
        yaw_scale=config.yaw_scale,
        pitch_roll_scale=config.pitch_roll_scale,
        position_std=config.position_std,
        velocity_std=config.velocity_std,
        omega_std=config.omega_std,
        quad_obj=quad_obj,
        ref_traj_name=config.ref_traj_name,
        skip_start=config.skip_start,
    )

    env = MinMaxObservationWrapper(env)

    if with_log_wrapper:
        env = LogWrapper(env)
    if with_vec_wrapper:
        env = VecEnv(env)

    return env


def build_policy_train_state(
    config: TrajTrackingConfig,
    env,
    key: jax.Array,
) -> TrainState:
    """Build a Flax TrainState with an MLP policy and optimizer.

    Args:
        config: Trajectory tracking configuration.
        env: Wrapped environment (used to get obs/action dims and hovering_action).
        key: JAX PRNG key for parameter initialisation.

    Returns:
        TrainState ready for training.
    """
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    layer_sizes = [obs_dim] + config.policy_net.hidden_layers + [action_dim]

    policy_net = MLP(
        layer_sizes,
        initial_scale=config.policy_net.initial_scale,
        action_bias=getattr(env, "hovering_action", None),
    )
    policy_params = policy_net.initialize(key)

    scheduler = optax.cosine_decay_schedule(config.optimizer.initial_lr, config.max_epochs)
    tx = optax.adam(scheduler)

    return TrainState.create(apply_fn=policy_net.apply, params=policy_params, tx=tx)


def load_policy_fn(
    checkpoint_path: str | Path,
    config: TrajTrackingConfig,
    env,
):
    """Load a trained policy checkpoint and return a callable policy function.

    Args:
        checkpoint_path: Path to the policy checkpoint directory.
        config: Trajectory tracking config (used for network architecture).
        env: Wrapped environment (used for obs/action dims and hovering_action).

    Returns:
        Callable ``(obs, key) -> action``.
    """
    ckptr = PyTreeCheckpointer()
    ckpt_path = resolve_path(checkpoint_path)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    layer_sizes = [obs_dim] + config.policy_net.hidden_layers + [action_dim]

    policy_net = MLP(
        layer_sizes,
        initial_scale=config.policy_net.initial_scale,
        action_bias=getattr(env, "hovering_action", None),
    )

    policy_params = ckptr.restore(str(ckpt_path))

    train_state = TrainState.create(
        apply_fn=policy_net.apply,
        params=policy_params,
        tx=optax.adam(1e-3),
    )

    def _policy_fn(obs, key):
        return train_state.apply_fn(train_state.params, obs)

    return _policy_fn


def load_residual_params(
    checkpoint_path: str | Path,
) -> FrozenDict:
    """Load residual dynamics ensemble parameters from an Orbax checkpoint.

    Args:
        checkpoint_path: Path to the residual dynamics checkpoint directory.

    Returns:
        FrozenDict of ensemble parameters.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    path = resolve_path(checkpoint_path)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Residual checkpoint not found: {path}\n"
            "Train a residual model first (uv run train residual) or pass "
            "--residual-checkpoint explicitly."
        )
    ckptr = PyTreeCheckpointer()
    return ckptr.restore(str(path))
