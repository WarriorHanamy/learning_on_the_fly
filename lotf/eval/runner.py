"""Trajectory tracking benchmark runner.

Evaluates a trained policy against a fixed simulator configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from lotf.envs.env_base import EnvTransition, rollout
from lotf.objects.reference_traj_obj import TrajColumns


@dataclass
class BenchmarkMetrics:
    """Numerical metrics from a trajectory tracking benchmark run."""

    mean_episodic_return: float
    collision_rate: float
    mean_episode_length: float
    position_rmse: float
    velocity_rmse: float

    def summary(self) -> str:
        return (
            f"mean_episodic_return : {self.mean_episodic_return:.4f}\n"
            f"collision_rate        : {self.collision_rate:.3f}\n"
            f"mean_episode_length   : {self.mean_episode_length:.1f}\n"
            f"position_rmse         : {self.position_rmse:.4f} [m]\n"
            f"velocity_rmse         : {self.velocity_rmse:.4f} [m/s]"
        )


@dataclass
class BenchmarkPolicySpec:
    """Policy input for a benchmark suite run."""

    label: str
    checkpoint_path: str
    scheme_name: str
    policy_fn: object


@dataclass
class BenchmarkRunResult:
    """Result for one policy in a benchmark suite."""

    label: str
    checkpoint_path: str
    scheme_name: str
    metrics: BenchmarkMetrics
    transitions: EnvTransition


def _compute_episode_lengths(terminated: jnp.ndarray, truncated: jnp.ndarray) -> jnp.ndarray:
    """Compute effective episode length for each rollout before termination/truncation."""
    done = jnp.logical_or(terminated, truncated)
    return jnp.argmax(done, axis=1)


def _compute_episode_returns(
    rewards: jnp.ndarray,
    terminated: jnp.ndarray,
    truncated: jnp.ndarray,
) -> jnp.ndarray:
    """Sum rewards up to and including the termination/truncation step."""
    done = jnp.logical_or(terminated, truncated)
    masks = jnp.logical_not(jnp.cumsum(done, axis=1))
    masks = jnp.concatenate([jnp.ones((done.shape[0], 1), dtype=bool), masks[:, :-1]], axis=1)
    return (rewards * masks.astype(rewards.dtype)).sum(axis=1)


def _compute_rmse(
    positions: np.ndarray,  # (num_rollouts, T, 3)
    velocities: np.ndarray,  # (num_rollouts, T, 3)
    init_ref_idx: np.ndarray,  # (num_rollouts, T)
    ref_traj: np.ndarray,  # (num_ref_points, 30)
    terminated: np.ndarray,
    truncated: np.ndarray,
) -> tuple[float, float]:
    """Compute position and velocity RMSE against the reference trajectory.

    Only timesteps before termination/truncation are included.
    """
    num_rollouts, T, _ = positions.shape
    done = np.logical_or(terminated, truncated)

    total_pos_sq = 0.0
    total_vel_sq = 0.0
    total_steps = 0

    for i in range(num_rollouts):
        ep_len = int(np.argmax(done[i])) + 1
        if ep_len <= 0:
            ep_len = T
        steps = min(ep_len, T)
        rollout_ref_start = int(init_ref_idx[i, 0])

        for t in range(steps):
            ref_idx = rollout_ref_start + t
            ref_idx = min(ref_idx, ref_traj.shape[0] - 1)
            pos_ref = ref_traj[ref_idx, TrajColumns.POS.slice]
            vel_ref = ref_traj[ref_idx, TrajColumns.VEL.slice]
            pos_err = positions[i, t] - pos_ref
            vel_err = velocities[i, t] - vel_ref
            total_pos_sq += np.sum(pos_err**2)
            total_vel_sq += np.sum(vel_err**2)
            total_steps += 1

    if total_steps == 0:
        return float("inf"), float("inf")

    pos_rmse = float(np.sqrt(total_pos_sq / total_steps))
    vel_rmse = float(np.sqrt(total_vel_sq / total_steps))
    return pos_rmse, vel_rmse


def run_benchmark(
    env,
    policy_fn,
    residual_params: FrozenDict,
    ref_traj: jnp.ndarray,
    num_rollouts: int = 20,
    seed: int = 0,
) -> tuple[BenchmarkMetrics, EnvTransition]:
    """Run a batch of rollouts and compute benchmark metrics.

    Args:
        env: Wrapped ``TrajTrackingStateEnv`` (MinMaxObservationWrapper only).
        policy_fn: Callable ``(obs, key) -> action``.
        residual_params: Residual dynamics ensemble parameters.
        ref_traj: Reference trajectory array (num_waypoints x 30).
        num_rollouts: Number of rollouts to run.
        seed: Random seed for rollout keys.

    Returns:
        ``(metrics, transitions)`` tuple.
    """
    key = jax.random.key(seed)
    rollout_keys = jax.random.split(key, num_rollouts)
    return run_benchmark_with_keys(env, policy_fn, residual_params, ref_traj, rollout_keys)


def run_benchmark_with_keys(
    env,
    policy_fn,
    residual_params: FrozenDict,
    ref_traj: jnp.ndarray,
    rollout_keys: jax.Array,
) -> tuple[BenchmarkMetrics, EnvTransition]:
    """Run benchmark with caller-provided rollout keys for fair comparisons."""
    parallel_rollout = jax.vmap(rollout, in_axes=(None, 0, None, None))
    transitions = parallel_rollout(env, rollout_keys, policy_fn, residual_params)

    # extract trajectory data
    rewards = np.array(transitions.reward)  # (num_rollouts, T)
    terminated = np.array(transitions.terminated)
    truncated = np.array(transitions.truncated)
    positions = np.array(transitions.state.quadrotor_state.p)
    velocities = np.array(transitions.state.quadrotor_state.v)
    init_ref_idx = np.array(transitions.state.init_ref_traj_idx)
    ref_traj = np.array(ref_traj)

    # compute metrics
    episode_returns = _compute_episode_returns(rewards, terminated, truncated)
    episode_lengths = _compute_episode_lengths(terminated, truncated)
    collision_rate = float(np.mean(terminated.any(axis=1)))

    pos_rmse, vel_rmse = _compute_rmse(
        positions, velocities, init_ref_idx, ref_traj, terminated, truncated
    )

    metrics = BenchmarkMetrics(
        mean_episodic_return=float(np.mean(episode_returns)),
        collision_rate=collision_rate,
        mean_episode_length=float(np.mean(episode_lengths.astype(np.float32))),
        position_rmse=pos_rmse,
        velocity_rmse=vel_rmse,
    )

    return metrics, transitions


def run_benchmark_suite(
    env,
    policy_specs: list[BenchmarkPolicySpec],
    residual_params: FrozenDict,
    ref_traj: jnp.ndarray,
    num_rollouts: int = 20,
    seed: int = 0,
) -> list[BenchmarkRunResult]:
    """Run multiple policies on the same benchmark seed and initial conditions."""
    key = jax.random.key(seed)
    rollout_keys = jax.random.split(key, num_rollouts)

    results = []
    for spec in policy_specs:
        metrics, transitions = run_benchmark_with_keys(
            env,
            spec.policy_fn,
            residual_params,
            ref_traj,
            rollout_keys,
        )
        results.append(
            BenchmarkRunResult(
                label=spec.label,
                checkpoint_path=spec.checkpoint_path,
                scheme_name=spec.scheme_name,
                metrics=metrics,
                transitions=transitions,
            )
        )
    return results
