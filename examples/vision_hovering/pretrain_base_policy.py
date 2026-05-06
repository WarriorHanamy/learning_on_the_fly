#!/usr/bin/env python3
"""Converted from 1_pretrain_base_policy.ipynb."""
import matplotlib

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
matplotlib.use("TkAgg")  # Use interactive backend when available

import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer
from tqdm import tqdm

from lotf import LOTF_PATH
from lotf.envs import HoveringFeaturesEnv, rollout
from lotf.modules import MLP
from lotf.objects import Quadrotor
from lotf.utils.math import normalize

"""Training a Feature-Based Hovering Policy With BPTT"""

# ======================================================================
#  1. Seed
# ======================================================================

seed = 0
key = jax.random.key(seed)
key_init, key_bptt = jax.random.split(key, 2)

# ======================================================================
#  2. Define Simulation Dynamics Config and Training Params
# ======================================================================

# simulation dynamics config
sim_dyn_config = {
    "use_high_fidelity": False,          # whether to use high-fidelity dynamics in forward simulation
    "use_forward_residual": False,       # whether to use residual dynamics in forward simulation
}

# ======================================================================
#  3. Create Quadrotor Object and Simulation Environment
# ======================================================================

# simulation parameters
sim_dt = 0.02
max_sim_time = 3.0

# quadrotor object
quad_obj = Quadrotor.from_name("example_quad", sim_dyn_config)

# simulation environment
env = HoveringFeaturesEnv(
    max_steps_in_episode=int(max_sim_time / sim_dt),
    dt=sim_dt,
    delay=0.04,
    yaw_scale=1.0,
    pitch_roll_scale=0.3,
    velocity_std=2.,
    omega_std=2.,
    quad_obj=quad_obj,
    reward_sharpness=5.0,
    action_penalty_weight=0.5,
    num_last_quad_states=15,
    skip_frames=3,
    hover_target=[1.5, 0.0, 1.5],
)

# get dimensions
action_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

print("====== env info ======")
print(f"action_dim: {action_dim}")
print(f"obs_dim: {obs_dim}")
print(f"target hover goal: {env.goal}")

# ======================================================================
#  4. Create Policy Network, Optimizer, and Train State (For Data Collection)
# ======================================================================

# policy network and init parameters for data collection
policy_net = MLP(
    [obs_dim, 512, 512, action_dim],
    initial_scale=1.0,
    action_bias=env.hovering_action,
)
policy_params = policy_net.initialize(key_init)

# dummy optimizer and train state for data collection
tx = optax.adam(0)
train_state = TrainState.create(
    apply_fn=policy_net.apply, params=policy_params, tx=tx
)

# ======================================================================
#  5. Load Dummy Residual Dynamics Network Parameters
# ======================================================================

# NOTE: Since we are training a base policy, we do not actually use the residual dynamics for forward sim or backprop
# However, we simply load a dummy residual dynamics model to satisfy the simulation environment requirements

path = LOTF_PATH + "/../checkpoints/residual_dynamics/dummy_params"
ckptr = PyTreeCheckpointer()
dummy_residual_params = ckptr.restore(path)

# ======================================================================
#  6. Collect Data
# ======================================================================

def collect_data(env, policy, num_rollouts, key):
    parallel_rollout = jax.vmap(
        partial(rollout, real_step=True, num_steps=1000),
        in_axes=(None, 0, None, None),
    )
    rollout_keys = jax.random.split(key, num_rollouts)
    transitions = parallel_rollout(env, rollout_keys, policy, dummy_residual_params)
    return transitions

def policy_collection(obs, key):
    return train_state.apply_fn(train_state.params, obs)

### collect rollout data
time_rollout = time.time()
transitions = collect_data(env, policy_collection, 100, jax.random.key(3))
time_rollout = time.time() - time_rollout
print(f"Rollout time: {time_rollout}")

### create dataset
# inputs: observations
observations = transitions.obs
observations = jnp.reshape(observations, (-1, observations.shape[-1]))

# targets: quadrotor state
p = transitions.state.quadrotor_state.p
# normalize the position
p = normalize(p, env.world_box.min, env.world_box.max)
R = transitions.state.quadrotor_state.R
v = transitions.state.quadrotor_state.v
v = normalize(v, env.v_min, env.v_max)
# flatten the last axis of R
R = jnp.reshape(R, (*R.shape[:-2], -1))
# concatenate the states
targets = jnp.concatenate([p, R, v], axis=-1)
targets = jnp.reshape(targets, (-1, targets.shape[-1]))

# ======================================================================
#  7. Pretrain Policy Network on State Prediction Task
# ======================================================================

@jax.jit
def train_step(state: TrainState, obs, targets):
    def loss_fn(params):
        preds = state.apply_fn(params, obs)
        loss = jnp.mean(jnp.abs(preds - targets))
        return loss

    grads = jax.grad(loss_fn)(state.params)  # Compute gradients
    new_state = state.apply_gradients(grads=grads)  # Update parameters
    return new_state

def train_loop(state, observations, targets, epochs=100, batch_size=32):
    dataset_size = observations.shape[0]
    for epoch in tqdm(range(epochs)):
        # Shuffle the data at the start of each epoch
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), dataset_size)
        obs_shuffled = observations[perm]
        targets_shuffled = targets[perm]

        # Iterate over the dataset in batches
        for i in range(0, dataset_size, batch_size):
            batch_obs = obs_shuffled[i : i + batch_size]
            batch_targets = targets_shuffled[i : i + batch_size]

            # Perform a training step
            state = train_step(state, batch_obs, batch_targets)

    return state

### create predictor MLP
predictor = MLP([obs_dim, 512, 512, targets.shape[-1]], initial_scale=.1)
predictor_params = predictor.initialize(jax.random.PRNGKey(0))

tx_predictor = optax.adam(1e-3)
train_state_predictor = TrainState.create(
    apply_fn=predictor.apply, params=predictor_params, tx=tx_predictor
)
epochs = 500
batch_size = 1024

# train the state predictor
train_state_predictor_new = train_loop(train_state_predictor, observations,
                                    targets,
                                    epochs, batch_size)

# ======================================================================
#  8. Save the Pretrained Policy Parameters
# ======================================================================

policy_name = "my_vision_hovering_pre_params"

# copy the trained policy parameters
policy_params['params']['Dense_0'] = train_state_predictor_new.params['params']['Dense_0']
policy_params['params']['Dense_1'] = train_state_predictor_new.params['params']['Dense_1']

path = LOTF_PATH + "/../checkpoints/policy/" + policy_name
ckptr = PyTreeCheckpointer()
ckptr.save(path, policy_params)
print("Policy saved successfully!")

