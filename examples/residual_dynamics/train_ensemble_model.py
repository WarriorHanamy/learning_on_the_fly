#!/usr/bin/env python3
"""Converted from train_ensemble_model.ipynb."""

import matplotlib

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
matplotlib.use("TkAgg")  # Use interactive backend when available

from time import time

import jax.numpy as jnp
import pandas as pd
from orbax.checkpoint import PyTreeCheckpointer

from lotf import LOTF_PATH
from lotf.utils.residual_dynamics import create_vec_funcs

"""Training an Ensemble of Residual Dynamics Networks.

Residual dynamics compensates for unmodeled aerodynamic effects (sim-to-real gap).
This script trains the model OFFLINE from collected flight data.

CSV data layout (22 columns total, no header):

    INPUT (19 columns) — quadrotor state at a given timestep:
        [0:3]   p  [m]          world-frame position (x, y, z)
        [3:12]  R  [dimless]    rotation matrix body→world, flattened row-major (9 values)
        [12:15] v  [m/s]        world-frame velocity (vx, vy, vz)
        [15]    f_d [m/s²]      thrust force divided by mass (scalar)
        [16:19] ω_d [rad/s]     desired body angular velocity (ωx, ωy, ωz)

    OUTPUT (3 columns) — residual acceleration to add to nominal dynamics:
        [19:22] a_res [m/s²]    residual acceleration in world frame (ax, ay, az)

Usage in simulation (online inference):
    Every timestep, the trained ensemble predicts a_res from the current (p, R, v, f_d, ω_d).
    The mean across ensemble members is added to the nominal acceleration:
        acc = gravity + R·thrust/mass + rotor_residual + a_res_mean
"""
# ======================================================================
#  1. Set Number of Models, Create Vectorized Functions
# ======================================================================

num_models = 3

# NOTE: these vectorized functions automatically broadcasts over arbitrary number of ensemble models
init_fn, train_fn, predict_fn = create_vec_funcs()

# ======================================================================
#  2. Load Dataset
# ======================================================================

# Offline-collected flight data: 19 input features → 3 residual acceleration targets.
# Data is gathered from real hardware or high-fidelity simulation by comparing
# measured acceleration against nominal model prediction.
dataset_name = "example_dataset.csv"

file_path = LOTF_PATH + "/../examples/residual_dynamics/" + dataset_name
df = pd.read_csv(file_path, header=None)
dataset = df.to_numpy()
print(f"Loaded dataset shape: {dataset.shape}")

input_dim = 19
output_dim = 3

# ======================================================================
#  3. Define Training Hyperparams
# ======================================================================

weight_init_scale = 1.0  # scale of weight initialization
learning_rate = 1e-2  # optimizer learning rate
lambda_reg = 1e-3  # weight norm regularization coefficient
num_epochs = 100
batch_size = 256
eval_every = 10

# ======================================================================
#  4. Initialize and Train
# ======================================================================

# initialize model params and train states
model_params, train_states = init_fn(learning_rate, jnp.arange(num_models, dtype=jnp.int32))

# prepare dataset
X, y = dataset[:, :input_dim], dataset[:, input_dim:]
X, y = jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)

tic = time()
train_states = train_fn(train_states, X, y, lambda_reg, num_epochs, eval_every)
print(f"Residual model training took {time() - tic:.2f} seconds")

# ======================================================================
#  5. Save Model Params
# ======================================================================

model_name = "my_residual_dynamics_params"

model_path = LOTF_PATH + "/../checkpoints/residual_dynamics/" + model_name
ckptr = PyTreeCheckpointer()
residual_params = train_states.params
ckptr.save(model_path, residual_params)
print("Saved model params!")
