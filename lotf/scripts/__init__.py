"""Training scripts for LOTF differentiable simulation.

This module provides executable training scripts for various control tasks:
- State-based hovering policy training
- Feature-based hovering (vision-based)
- Trajectory tracking

Scripts are designed to be run via:
    uv run python -m lotf.scripts.train_state_hovering --config configs/state_hovering.yaml
"""

from .train_state_hovering import create_env, create_policy, main

__all__ = [
    "create_env",
    "create_policy",
    "main",
]
