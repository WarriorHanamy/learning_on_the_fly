from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState

from lotf.envs.env_base import Env, EnvState


class TrajectoryState(PyTreeNode):
    """Holds the transition data collected during a rollout."""

    reward: jnp.array
    obs: jnp.array


def progress_callback_host(episode_loss):
    """Prints training progress from the host process."""
    episode, loss = episode_loss
    print(f"Episode: {episode}, Loss: {loss:.2f}")


NUM_EPOCHS_PER_CALLBACK = 10


def progress_callback(episode, loss):
    """Triggers a host-side debug callback for loss logging."""
    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda eps_lss: jax.debug.callback(progress_callback_host, eps_lss),
        false_fun=lambda eps_lss: None,
        operand=(episode, loss),
    )


def grad_callback_host(episode_grad):
    """Prints gradient statistics from the host process."""
    episode, grad = episode_grad
    print(f"Episode: {episode}, Grad max: {grad:.4f}")


def grad_callback(episode, grad_norm):
    """Triggers a host-side debug callback for gradient logging."""
    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda eps_lss: jax.debug.callback(grad_callback_host, eps_lss),
        false_fun=lambda eps_lss: None,
        operand=(episode, grad_norm),
    )


class RunnerState(NamedTuple):
    """Represents the complete state of the training loop."""

    actor_train_state: TrainState
    env_state: EnvState
    last_obs: jax.Array
    key: chex.PRNGKey
    epoch_idx: int


def _make_step_fn(num_envs, res_model_params, env):
    """Creates a single-step transition function used in rollouts.

    Args:
        num_envs: number of parallel environments.
        res_model_params: frozen parameters for the residual dynamics model.
        env: the environment instance.

    Returns:
        A step_fn callable compatible with jax.lax.scan.
    """

    def step_fn(runner_state, actor_params, _unused):
        actor_ts, env_state, last_obs, key, epoch_idx = runner_state
        action = actor_ts.apply_fn(actor_params, last_obs)
        key, key_ = jax.random.split(key)
        key_step = jax.random.split(key_, num_envs)
        env_state, obs, reward, _terminated, _truncated, info = env.step(
            env_state, action, res_model_params, key_step
        )
        runner_state = RunnerState(actor_ts, env_state, obs, key, epoch_idx)
        return runner_state, TrajectoryState(reward=reward, obs=last_obs)

    return step_fn


def _full_trajectory_epoch_fn(epoch_state, _unused, num_steps_per_epoch, num_envs, step_fn):
    """Original epoch function: one gradient over the full trajectory."""

    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(actor_params, runner_state):

        def rollout(rs):
            def _step(rs_inner, u):
                return step_fn(rs_inner, actor_params, u)

            final_rs, traj = jax.lax.scan(_step, rs, None, num_steps_per_epoch)
            return final_rs, traj

        final_rs, traj = rollout(runner_state)
        loss = -traj.reward.sum() / num_envs
        return loss, final_rs

    actor_ts = epoch_state.actor_train_state
    (loss, epoch_state), grad = loss_fn(actor_ts.params, epoch_state)
    actor_ts = actor_ts.apply_gradients(grads=grad)

    leaves = jax.tree_util.tree_leaves(grad)
    grad_vec = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
    grad_max = jnp.max(jnp.abs(grad_vec))

    progress_callback(epoch_state.epoch_idx, loss)
    grad_callback(epoch_state.epoch_idx, grad_max)

    epoch_state = epoch_state._replace(
        actor_train_state=actor_ts, epoch_idx=epoch_state.epoch_idx + 1
    )
    return epoch_state, loss


def train(
    env: Env,
    env_state: EnvState,
    obs: jax.Array,
    actor_train_state: TrainState,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
    res_model_params: FrozenDict,
    key: chex.PRNGKey,
    window_size: int = 0,
):
    """Executes a full-trajectory BPTT training loop.

    Args:
        env: the environment instance.
        env_state: the initial state of the environment.
        obs: the initial observation.
        actor_train_state: the flax train state containing actor params and optimizer.
        num_epochs: total number of training iterations.
        num_steps_per_epoch: rollout length per epoch.
        num_envs: number of parallel environments.
        res_model_params: fixed parameters for the residual dynamics model.
        key: rng key for stochastic operations.
        window_size: *unused* (routing hint, kept for compatibility).

    Returns:
        a dictionary containing the final runner state and training metrics.
    """
    runner_state = RunnerState(actor_train_state, env_state, obs, key, epoch_idx=0)
    step_fn = _make_step_fn(num_envs, res_model_params, env)

    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def _epoch_step(
        env,
        num_steps_per_epoch,
        num_envs,
        window_size,
        _gamma,
        res_model_params: FrozenDict,
        runner_state: RunnerState,
    ):
        return _full_trajectory_epoch_fn(
            runner_state,
            None,
            num_steps_per_epoch,
            num_envs,
            step_fn,
        )

    losses = []
    for _ in range(num_epochs):
        try:
            runner_state, loss = _epoch_step(
                env,
                num_steps_per_epoch,
                num_envs,
                0,
                0.0,
                res_model_params,
                runner_state,
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            break
        losses.append(loss)

    return {"runner_state": runner_state, "metrics": jnp.array(losses)}
