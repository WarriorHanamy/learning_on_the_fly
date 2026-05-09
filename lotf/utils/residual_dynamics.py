from typing import Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from lotf.modules import ResidualDynamicsMLP


def get_residual_dyn_model_apply_fn() -> Callable:
    """Returns a vectorized apply function for the residual dynamics MLP.

    Handles both ensemble-batched params (trained via create_vec_funcs)
    and single-model params (from a basic checkpoint).
    """

    def apply_fn(params, x):
        model = ResidualDynamicsMLP([19, 128, 128, 3], initial_scale=1.0)
        return model.apply(params, x)

    parallel_apply_fn = jax.vmap(apply_fn, in_axes=(0, None))

    def _apply(params, x):
        leaves = jax.tree_util.tree_leaves(params)
        # Ensemble params have extra leading batch dim: max ndim >= 3
        # Single-model params: max ndim == 2 (kernels are 2D)
        max_ndim = max(leaf.ndim for leaf in leaves)
        if max_ndim <= 2:
            return apply_fn(params, x)[jnp.newaxis, :]
        return parallel_apply_fn(params, x)

    return _apply


@jax.jit
def mse_loss(state: TrainState, x: jax.Array, y: jax.Array) -> jax.Array:
    """Computes the mean squared error loss"""
    preds = state.apply_fn(state.params, x)
    return jnp.mean((preds - y) ** 2)


@jax.jit
def full_loss(state: TrainState, x: jax.Array, y: jax.Array, lambda_reg: float) -> jax.Array:
    """Computes total loss including spectral regularization"""
    preds = state.apply_fn(state.params, x)
    mse = jnp.mean((preds - y) ** 2)
    spec_norm = compute_spectral_norm(state.params)
    return mse + lambda_reg * spec_norm


@jax.jit
def train_step(state: TrainState, x: jax.Array, y: jax.Array, lambda_reg: float) -> TrainState:
    """Performs a single gradient descent update step"""

    def loss_fn(params):
        preds = state.apply_fn(params, x)
        mse = jnp.mean((preds - y) ** 2)
        spec_norm = compute_spectral_norm(params)
        return mse + lambda_reg * spec_norm

    # compute gradients with respect to params
    grads = jax.grad(loss_fn)(state.params)

    # update training state using optax optimizer
    return state.apply_gradients(grads=grads)


@jax.jit
def compute_spectral_norm(params: dict) -> jax.Array:
    """
    Approximates regularization by summing the l2-norm of kernel weights.
    """
    reg = 0.0
    for layer in params["params"].values():
        if "kernel" in layer:
            W = layer["kernel"]
            # compute spectral norm (largest singular value)
            reg += jnp.linalg.norm(W, ord=2)
    return reg


def predict_fn(params: dict, x: jax.Array) -> jax.Array:
    """Forward pass function for making predictions"""
    model = ResidualDynamicsMLP([19, 128, 128, 3], initial_scale=1.0)
    return model.apply(params, x)


def init_fn(learning_rate: float, seed: int) -> Tuple[dict, TrainState]:
    """Initializes model parameters and flax trainstate"""
    model = ResidualDynamicsMLP([19, 128, 128, 3], initial_scale=1.0)
    model_params = model.initialize(jax.random.PRNGKey(seed))

    # define optimizer
    tx = optax.adam(learning_rate)

    train_state = TrainState.create(apply_fn=model.apply, params=model_params, tx=tx)
    return model_params, train_state


@partial(jax.jit, static_argnames=("lambda_reg", "num_epochs", "eval_every"))
def train(
    train_state: TrainState,
    X: jax.Array,
    y: jax.Array,
    lambda_reg: float,
    num_epochs: int,
    eval_every: int,
) -> TrainState:
    """
    JIT-compiled training loop using jax.lax.scan for performance.
    """

    def scan_fn(carry, epoch):
        current_state = carry

        # execute optimization step
        new_state = train_step(current_state, X, y, lambda_reg)

        def do_log(state_to_log):
            """Helper for conditional side-effect logging"""
            train_mse_loss = mse_loss(state_to_log, X, y)
            train_total_loss = full_loss(state_to_log, X, y, lambda_reg)

            jax.debug.print(
                "Epoch {e}/{t} | Train MSE: {mse} | Total Loss: {loss}",
                e=epoch,
                t=num_epochs,
                mse=train_mse_loss,
                loss=train_total_loss,
            )
            return None

        # trigger logging only at specified intervals
        jax.lax.cond(epoch % eval_every == 0, do_log, lambda _: None, new_state)

        return new_state, None

    # create epoch range for scan
    epochs = jnp.arange(num_epochs + 1)
    final_state, _ = jax.lax.scan(scan_fn, train_state, xs=epochs, length=num_epochs + 1)

    return final_state


def create_vec_funcs():
    """
    Creates vectorized versions of init, train, and predict functions.
    useful for ensemble training or hyperparameter sweeps.
    """
    parallel_init_fn = jax.vmap(init_fn, in_axes=(None, 0))
    parallel_train_fn = jax.vmap(train, in_axes=(0, None, None, None, None, None))
    parallel_predict_fn = jax.vmap(predict_fn, in_axes=(0, None))

    return parallel_init_fn, parallel_train_fn, parallel_predict_fn
