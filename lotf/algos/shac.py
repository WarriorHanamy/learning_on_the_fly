"""Short-Horizon Actor-Critic (SHAC) training for differentiable simulation.

Reference: Xu et al., "Accelerated Policy Learning with Parallel Differentiable
Simulation", ICLR 2023 (https://arxiv.org/abs/2204.07137).
"""

from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState

from lotf.envs.env_base import Env, EnvState

NUM_EPOCHS_PER_CALLBACK = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class TrajectoryData(PyTreeNode):
    """Data collected during actor rollout, used for critic training.

    Shapes assume (T, N, …) where T = window_size, N = num_envs.
    """

    obs_buf: jnp.array
    rew_buf: jnp.array
    done_mask: jnp.array
    next_values: jnp.array


class RunnerState(NamedTuple):
    actor_train_state: TrainState
    critic_train_state: TrainState
    target_critic_params: FrozenDict
    env_state: EnvState
    last_obs: jax.Array
    key: chex.PRNGKey
    epoch_idx: int


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _progress_callback(episode, loss):
    def _host(ep_loss):
        ep, lv = ep_loss
        print(f"Episode: {ep}, Loss: {lv:.2f}")

    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda x: jax.debug.callback(_host, x),
        false_fun=lambda x: None,
        operand=(episode, loss),
    )


def _grad_callback(episode, grad_norm):
    def _host(ep_grad):
        ep, g = ep_grad
        print(f"Episode: {ep}, Grad max: {g:.4f}")

    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda x: jax.debug.callback(_host, x),
        false_fun=lambda x: None,
        operand=(episode, grad_norm),
    )


# ---------------------------------------------------------------------------
# Environment step helper
# ---------------------------------------------------------------------------


def _make_env_step_fn(env, res_model_params, num_envs):
    def _env_step(env_state, obs, action, key):
        key, key_step = jax.random.split(key)
        key_step_batch = jax.random.split(key_step, num_envs)
        env_state, next_obs, reward, terminated, truncated, info = env.step(
            env_state, action, res_model_params, key_step_batch
        )
        done = jnp.logical_or(terminated, truncated)
        obs_before_reset = info["obs_before_reset"]
        return env_state, next_obs, reward, done, obs_before_reset, key

    return _env_step


# ---------------------------------------------------------------------------
# Actor rollout
# ---------------------------------------------------------------------------


def _actor_rollout_and_loss(
    actor_params,
    target_critic_params,
    actor_apply_fn,
    critic_apply_fn,
    env_step,
    runner_state,
    window_size,
    num_envs,
    gamma,
):
    """Differentiable rollout returning (loss, traj_data, carry)."""

    obs_dim = runner_state.last_obs.shape[-1]

    carry_init = (
        runner_state.env_state,
        runner_state.last_obs,
        jnp.zeros(num_envs),
        jnp.ones(num_envs),
        runner_state.key,
        jnp.float32(0.0),
        jnp.zeros((window_size, num_envs, obs_dim)),
        jnp.zeros((window_size, num_envs)),
        jnp.zeros((window_size, num_envs), dtype=jnp.bool_),
        jnp.zeros((window_size + 1, num_envs)),
    )

    def _scan_body(carry, t):
        (
            env_state,
            last_obs,
            rew_acc,
            gamma_per_env,
            key,
            loss_accum,
            obs_buf,
            rew_buf,
            done_mask,
            next_values,
        ) = carry

        action = actor_apply_fn(actor_params, last_obs)
        env_state, next_obs, reward, done, obs_before_reset, key = env_step(
            env_state, last_obs, action, key
        )

        next_value = critic_apply_fn(target_critic_params, next_obs)
        term_value_done = critic_apply_fn(target_critic_params, obs_before_reset)
        next_value = jnp.where(done, term_value_done, next_value)

        rew_acc = rew_acc + gamma_per_env * reward

        loss_contrib = jnp.where(done, -rew_acc - gamma * gamma_per_env * next_value, 0.0)
        loss_accum = loss_accum + loss_contrib.sum()

        gamma_per_env = gamma_per_env * gamma
        gamma_per_env = jnp.where(done, 1.0, gamma_per_env)
        rew_acc = jnp.where(done, 0.0, rew_acc)

        obs_buf = obs_buf.at[t].set(last_obs)
        rew_buf = rew_buf.at[t].set(reward)
        done_mask = done_mask.at[t].set(done)
        next_values = next_values.at[t + 1].set(next_value)

        new_carry = (
            env_state,
            next_obs,
            rew_acc,
            gamma_per_env,
            key,
            loss_accum,
            obs_buf,
            rew_buf,
            done_mask,
            next_values,
        )
        return new_carry, None

    carry_final, _ = jax.lax.scan(_scan_body, carry_init, jnp.arange(window_size))
    (
        env_state_f,
        obs_f,
        rew_acc_f,
        gamma_per_env_f,
        key_f,
        loss_accum,
        obs_buf,
        rew_buf,
        done_mask,
        next_values,
    ) = carry_final

    final_value = critic_apply_fn(target_critic_params, obs_f)
    loss_accum = loss_accum + (-rew_acc_f - gamma * gamma_per_env_f * final_value).sum()

    actor_loss = loss_accum / (window_size * num_envs)

    traj_data = TrajectoryData(
        obs_buf=obs_buf,
        rew_buf=rew_buf,
        done_mask=done_mask,
        next_values=next_values,
    )

    new_runner_state = runner_state._replace(
        env_state=env_state_f, last_obs=obs_f, key=key_f, epoch_idx=runner_state.epoch_idx + 1
    )

    return actor_loss, (new_runner_state, traj_data)


# ---------------------------------------------------------------------------
# Critic targets
# ---------------------------------------------------------------------------


def _td_lambda_targets(rew_buf, next_values, done_mask, gamma, lam):
    """TD(λ) targets via backward scan (replicating NVIDIA SHAC reference).

    Args:
        rew_buf:     (T, N)
        next_values: (T+1, N) — V(s_{i+1}) at next_values[i+1]
        done_mask:   (T, N)
        gamma, lam:  scalars

    Returns:
        targets: (T, N)
    """
    init = (jnp.zeros_like(rew_buf[0]), jnp.zeros_like(rew_buf[0]), jnp.ones_like(rew_buf[0]))

    def _scan_fn(carry, inputs):
        A, B, lam_env = carry
        rew, nv, dm = inputs

        lam_env = lam_env * lam * (1.0 - dm) + dm
        A_new = (1.0 - dm) * (lam * gamma * A + gamma * nv + (1.0 - lam_env) / (1.0 - lam) * rew)
        B_new = gamma * (nv * dm + B * (1.0 - dm)) + rew
        target = (1.0 - lam) * A_new + lam_env * B_new

        return (A_new, B_new, lam_env), target

    _, targets = jax.lax.scan(
        _scan_fn,
        init,
        (rew_buf, next_values[1:], done_mask),
        reverse=True,
    )
    return targets


# ---------------------------------------------------------------------------
# Critic training
# ---------------------------------------------------------------------------


def _critic_train_iteration(critic_ts, obs_flat, targets_flat, batch_size, key):
    """One critic training iteration over mini-batches."""

    total = obs_flat.shape[0]
    key, key_perm = jax.random.split(key)
    perm = jax.random.permutation(key_perm, total)

    def _batch_body(carry, start):
        critic_ts, key = carry
        idx = jax.lax.dynamic_slice_in_dim(perm, start, batch_size)
        batch_obs = obs_flat[idx]
        batch_targets = jax.lax.stop_gradient(targets_flat[idx])

        @jax.value_and_grad
        def _loss(params):
            preds = critic_ts.apply_fn(params, batch_obs)
            return ((preds - batch_targets) ** 2).mean()

        loss, grad = _loss(critic_ts.params)
        critic_ts = critic_ts.apply_gradients(grads=grad)
        return (critic_ts, key), loss

    starts = jnp.arange(0, total, batch_size)
    (critic_ts, _), batch_losses = jax.lax.scan(_batch_body, (critic_ts, key), starts)
    return critic_ts, batch_losses.mean()


# ---------------------------------------------------------------------------
# Main loop helpers
# ---------------------------------------------------------------------------


def _reset_envs(env, key, num_envs):
    key_reset = jax.random.split(key, num_envs)
    return env.reset(key_reset, None)


# ---------------------------------------------------------------------------
# Public train entry-point
# ---------------------------------------------------------------------------


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
    window_size: int = 32,
    gamma: float = 0.99,
    lam: float = 0.95,
    critic_train_state: TrainState = None,
    critic_method: str = "td-lambda",
    critic_iterations: int = 16,
    target_critic_alpha: float = 0.4,
    critic_batch_size: int | None = None,
):
    """SHAC training loop.

    Each epoch resets environments, performs one differentiable actor rollout
    of length ``window_size`` with terminal-value bootstrap from the target
    critic, then trains the critic for ``critic_iterations`` on the collected
    data, and finally updates the target critic via Polyak averaging.

    Args:
        env:                  environment instance.
        env_state:            *ignored* (reset each epoch).
        obs:                  *ignored* (reset each epoch).
        actor_train_state:    Flax TrainState for the actor (MLP policy).
        num_epochs:           number of SHAC iterations.
        num_steps_per_epoch:  *ignored* (``window_size`` is used).
        num_envs:             number of parallel environments.
        res_model_params:     frozen residual-dynamics parameters.
        key:                  PRNG key.
        window_size:          short-horizon rollout length (= steps_num).
        gamma:                discount factor.
        lam:                  TD(λ) parameter.
        critic_train_state:   Flax TrainState for the critic.
        critic_method:        ``"td-lambda"`` or ``"one-step"``.
        critic_iterations:    number of critic gradient steps per epoch.
        target_critic_alpha:  Polyak averaging coefficient.
        critic_batch_size:    mini-batch size for critic (default: N*T//4).

    Returns:
        {"runner_state": RunnerState, "metrics": losses array}
    """
    assert critic_train_state is not None, "critic_train_state is required for SHAC"
    assert num_steps_per_epoch % window_size == 0, (
        f"num_steps_per_epoch ({num_steps_per_epoch}) must be divisible by "
        f"window_size ({window_size})"
    )

    if critic_batch_size is None:
        critic_batch_size = num_envs * window_size // 4
        critic_batch_size = max(1, critic_batch_size)

    env_step = _make_env_step_fn(env, res_model_params, num_envs)

    target_critic_params = jax.tree_util.tree_map(lambda x: x, critic_train_state.params)

    runner_state = RunnerState(
        actor_train_state=actor_train_state,
        critic_train_state=critic_train_state,
        target_critic_params=target_critic_params,
        env_state=env_state,
        last_obs=obs,
        key=key,
        epoch_idx=0,
    )

    # value_and_grad wrapper — only actor_params differentiate
    _actor_vg = jax.value_and_grad(_actor_rollout_and_loss, has_aux=True)

    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def _epoch_step(env, num_steps_per_epoch, num_envs, window_size, gamma, runner_state):
        # ── 1. Reset environments ──
        key = runner_state.key
        key, key_reset = jax.random.split(key)
        env_state_reset, obs_reset = _reset_envs(env, key_reset, num_envs)
        runner_state = runner_state._replace(env_state=env_state_reset, last_obs=obs_reset, key=key)

        # ── 2. Actor rollout + gradient step ──
        (actor_loss, (runner_state, traj_data)), actor_grad = _actor_vg(
            runner_state.actor_train_state.params,
            runner_state.target_critic_params,
            runner_state.actor_train_state.apply_fn,
            runner_state.critic_train_state.apply_fn,
            env_step,
            runner_state,
            window_size,
            num_envs,
            gamma,
        )
        actor_ts = runner_state.actor_train_state.apply_gradients(grads=actor_grad)
        runner_state = runner_state._replace(actor_train_state=actor_ts)

        grad_vec = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(actor_grad)])
        grad_max = jnp.max(jnp.abs(grad_vec))

        # ── 3. Critic training ──
        obs_flat = traj_data.obs_buf.reshape(-1, traj_data.obs_buf.shape[-1])

        if critic_method == "one-step":
            targets = traj_data.rew_buf + gamma * traj_data.next_values[1:]
        elif critic_method == "td-lambda":
            targets = _td_lambda_targets(
                traj_data.rew_buf, traj_data.next_values, traj_data.done_mask, gamma, lam
            )
        else:
            raise ValueError(f"Unknown critic_method: {critic_method}")

        targets_flat = targets.reshape(-1)

        def _critic_scan_body(critic_ts_key, _):
            critic_ts, key_c = critic_ts_key
            key_c, key_iter = jax.random.split(key_c)
            critic_ts, critic_loss = _critic_train_iteration(
                critic_ts, obs_flat, targets_flat, critic_batch_size, key_iter
            )
            return (critic_ts, key_c), critic_loss

        (critic_ts, key_c), critic_losses = jax.lax.scan(
            _critic_scan_body,
            (runner_state.critic_train_state, runner_state.key),
            None,
            length=critic_iterations,
        )
        runner_state = runner_state._replace(critic_train_state=critic_ts, key=key_c)

        # ── 4. Polyak update target critic ──
        def _polyak(tgt, src):
            return target_critic_alpha * tgt + (1.0 - target_critic_alpha) * src

        target_critic_params = jax.tree_util.tree_map(
            _polyak, runner_state.target_critic_params, critic_ts.params
        )
        runner_state = runner_state._replace(target_critic_params=target_critic_params)

        # ── 5. Callbacks ──
        _progress_callback(runner_state.epoch_idx, actor_loss)
        _grad_callback(runner_state.epoch_idx, grad_max)

        return runner_state, actor_loss

    losses = []
    for _ in range(num_epochs):
        try:
            runner_state, loss = _epoch_step(
                env, num_steps_per_epoch, num_envs, window_size, gamma, runner_state
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            break
        losses.append(loss)

    return {"runner_state": runner_state, "metrics": jnp.array(losses)}
