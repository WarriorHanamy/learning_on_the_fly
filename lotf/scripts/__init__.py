"""Training scripts for LOTF differentiable simulation.

This module provides executable training scripts for various control tasks:
- State-based hovering policy training
- Feature-based hovering (vision-based)
- Trajectory tracking

Scripts are designed to be run via:
    uv run python -m lotf.scripts.train_state_hovering --config configs/state_hovering.yaml
    uv run python -m lotf.scripts.train_traj_tracking --config configs/traj_tracking.yaml
"""

from .train_state_hovering import create_env as create_hovering_env
from .train_state_hovering import create_policy as create_hovering_policy
from .train_state_hovering import main as main_state_hovering

__all__ = [
    "create_hovering_env",
    "create_hovering_policy",
    "main_state_hovering",
]
