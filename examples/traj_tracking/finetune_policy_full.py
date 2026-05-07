#!/usr/bin/env python3
"""Converted from 3_finetune_policy_full.ipynb."""

import matplotlib

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
matplotlib.use("TkAgg")  # Use interactive backend when available

import time

import jax
import optax
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer

from lotf import LOTF_PATH
from lotf.algos import bptt
from lotf.envs import TrajTrackingStateEnv
from lotf.envs.wrappers import LogWrapper, MinMaxObservationWrapper, VecEnv
from lotf.modules import MLP
from lotf.objects import Quadrotor, RefTrajNames

"""Finetuning a Trained Trajectory Tracking Policy With BPTT"""

# ======================================================================
#  1. Seed
# ======================================================================

seed = 0
key = jax.random.key(seed)
key_init, key_bptt = jax.random.split(key, 2)

# ======================================================================
#  2. Define Simulation Dynamics Config and Training Params
# ======================================================================

# forward model config
forward_model_config = {
    "enable_inner_loop_dynamics": False,
    "enable_residual_acceleration": False,
}

# training parameters
num_envs = 200
max_epochs = 200

# ======================================================================
#  3. Create Quadrotor Object and Simulation Environment
# ======================================================================

# simulation parameters
sim_dt = 0.02
max_sim_time = 5.0

# reference trajectory
ref_traj_name = RefTrajNames.FIG8

# quadrotor object
quad_obj = Quadrotor.from_name("example_quad", forward_model_config)

# simulation environment
env = TrajTrackingStateEnv(
    max_steps_in_episode=int(max_sim_time / sim_dt),
    dt=sim_dt,
    delay=0.04,
    yaw_scale=0.1,
    pitch_roll_scale=0.1,
    position_std=0.05,
    velocity_std=0.05,
    omega_std=0.05,
    quad_obj=quad_obj,
    ref_traj_name=ref_traj_name,
)

# apply min-max observation wrapper
env = MinMaxObservationWrapper(env)

# get dimensions
action_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

# apply additional wrappers
env = LogWrapper(env)
env = VecEnv(env)

print("====== env info ======")
print(f"action_dim: {action_dim}")
print(f"obs_dim: {obs_dim}")

# ======================================================================
#  4. Load Policy Parameters, Create Optimizer and Train State
# ======================================================================

policy_name = "traj_tracking_params"

# policy network and init parameters
policy_net = MLP(
    [obs_dim, 512, 512, action_dim],
    initial_scale=0.01,
    action_bias=env.hovering_action,
)
path = LOTF_PATH + "/../checkpoints/policy/" + policy_name
ckptr = PyTreeCheckpointer()
policy_params = ckptr.restore(path)

# optimizer
scheduler = optax.cosine_decay_schedule(1e-3, max_epochs)
tx = optax.adam(scheduler)

# train state object
train_state = TrainState.create(apply_fn=policy_net.apply, params=policy_params, tx=tx)

# ======================================================================
#  5. Load Residual Dynamics Network Parameters
# ======================================================================

residual_dynamics_name = "example_params"

path = LOTF_PATH + "/../checkpoints/residual_dynamics/" + residual_dynamics_name
ckptr = PyTreeCheckpointer()
dummy_residual_params = ckptr.restore(path)

# ======================================================================
#  6. Train
# ======================================================================

# intialize environments
key_bptt, key_ = jax.random.split(key_bptt)
key_reset = jax.random.split(key_, num_envs)
init_env_state, init_obs = env.reset(key_reset, None)

# training loop
time_start = time.time()
res_dict = bptt.train(
    env,
    init_env_state,
    init_obs,
    train_state,
    num_epochs=max_epochs,
    num_steps_per_epoch=env.max_steps_in_episode,
    num_envs=num_envs,
    res_model_params=dummy_residual_params,
    key=key_bptt,
)
time_train_compile = time.time() - time_start
print(f"Compile + Training time: {time_train_compile}")
