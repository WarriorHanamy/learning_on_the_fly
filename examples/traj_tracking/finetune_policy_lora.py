#!/usr/bin/env python3
"""Converted from 3_finetune_policy_lora.ipynb."""
import matplotlib

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
matplotlib.use("TkAgg")  # Use interactive backend when available

import time

import jax
import optax
from flax.core import freeze, unfreeze
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer

from lotf import LOTF_PATH
from lotf.algos import bptt
from lotf.envs import TrajTrackingStateEnv
from lotf.envs.wrappers import LogWrapper, MinMaxObservationWrapper, VecEnv
from lotf.modules import MLP, LoraMLP
from lotf.objects import Quadrotor, RefTrajNames
from lotf.utils.lora import (
    lora_only_mask,
    partition_params,
    recursive_merge,
)

"""(LoRA) Finetuning a Trained State-Based Hovering Policy With BPTT"""

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
quad_obj = Quadrotor.from_name("example_quad", sim_dyn_config)

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
#  4. Load Base Policy Parameters, Create Optimizer and Train State
# ======================================================================

policy_name = "traj_tracking_params"

# policy network and init parameters
base_policy_net = MLP(
    [obs_dim, 512, 512, action_dim],
    initial_scale=0.01,
    action_bias=env.hovering_action,
)
path = LOTF_PATH + "/../checkpoints/policy/" + policy_name
ckptr = PyTreeCheckpointer()
base_policy_params = ckptr.restore(path)

# ======================================================================
#  5. Define LoRA Policy Network, Create Optimizer and Train State
# ======================================================================

lora_ranks = [1, 1, 1]
lora_alpha = 1.0

# LoRA policy network
policy_net = LoraMLP(base_mlp=base_policy_net, lora_ranks=lora_ranks, lora_alpha=lora_alpha)
policy_params = policy_net.initialize_with_base(key_init, base_policy_params)

mask = lora_only_mask(policy_params)
frozen_params, trainable_params = partition_params(policy_params, mask)
def apply_combined(params, x):
    full_params = freeze(recursive_merge(unfreeze(frozen_params), unfreeze(params)))
    return policy_net.apply(full_params, x)

# optimizer and train state
tx = optax.adam(learning_rate=1e-3)
train_state = TrainState.create(
    apply_fn=apply_combined, params=trainable_params, tx=tx
)

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

