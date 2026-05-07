#!/usr/bin/env python3
"""Converted from 1_train_base_policy.ipynb."""

import matplotlib

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
matplotlib.use("TkAgg")  # Use interactive backend when available

import time

import jax
import matplotlib.pyplot as plt
import optax
import seaborn as sns
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer

from lotf import LOTF_PATH
from lotf.algos import bptt
from lotf.envs import TrajTrackingStateEnv
from lotf.envs.wrappers import LogWrapper, MinMaxObservationWrapper, VecEnv
from lotf.modules import MLP
from lotf.objects import Quadrotor, RefTrajNames

"""Training a Trajectory Tracking Policy With BPTT"""

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
num_envs = 300
max_epochs = 300

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
    position_std=0.1,
    velocity_std=0.1,
    omega_std=0.1,
    quad_obj=quad_obj,
    ref_traj_name=ref_traj_name,
    skip_start=True,
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
#  4. Create Policy Network, Optimizer, and Train State
# ======================================================================

# policy network and init parameters
policy_net = MLP(
    [obs_dim, 512, 512, action_dim],
    initial_scale=0.01,
    action_bias=env.hovering_action,
)
policy_params = policy_net.initialize(key_init)

# optimizer
scheduler = optax.cosine_decay_schedule(1e-3, max_epochs)
tx = optax.adam(scheduler)

# train state object
train_state = TrainState.create(apply_fn=policy_net.apply, params=policy_params, tx=tx)

# ======================================================================
#  5. Load Dummy Residual Dynamics Network Parameters
# ======================================================================

# NOTE: Since we are training a base policy, we do not actually use the residual dynamics for forward sim or backprop
# However, we simply load a dummy residual dynamics model to satisfy the simulation environment requirements

path = LOTF_PATH + "/../checkpoints/residual_dynamics/dummy_params"
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

# ======================================================================
#  7. Plot Rewards
# ======================================================================

losses = res_dict["metrics"]
returns = -losses

sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 5))
plt.plot(returns, color="#2c3e50", linewidth=2)
plt.title(
    f"Final Reward: {returns[-1]:.2f} | Total Runtime: {time_train_compile:.2f}s",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Reward", fontsize=12)
plt.tight_layout()
sns.despine()
plt.savefig("/tmp/plot.png")
plt.show()

# ======================================================================
#  8. (Optional) Save the Policy Checkpoint
# ======================================================================

policy_name = "my_traj_tracking_params"

path = LOTF_PATH + "/../checkpoints/policy/" + policy_name
ckptr = PyTreeCheckpointer()
trained_policy_params = res_dict["runner_state"].train_state.params
ckptr.save(path, trained_policy_params)
print("Policy saved successfully!")
