"""Configuration dataclasses for training, environment, and simulation parameters.

All configs are frozen (immutable) to ensure reproducibility and prevent accidental
modifications during runtime.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters configuration.

    Attributes:
        seed: Random seed for reproducibility.
        num_envs: Number of parallel environments.
        max_epochs: Maximum number of training epochs.
        learning_rate: Learning rate for optimizer.
    """

    seed: int
    num_envs: int
    max_epochs: int
    learning_rate: float


@dataclass(frozen=True)
class EnvConfig:
    """Environment configuration parameters.

    Attributes:
        dt: Time step duration in seconds.
        delay: Action delay in seconds.
        max_steps_in_episode: Maximum steps per episode before truncation.
    """

    dt: float
    delay: float
    max_steps_in_episode: int


@dataclass(frozen=True)
class SimConfig:
    """Simulation configuration parameters.

    Base simulation config that can be extended for specific simulators.

    Attributes:
        timestep: Physics simulation timestep in seconds.
        gravity: Gravitational acceleration (default: 9.81 m/s^2).
        render: Whether to enable rendering.
    """

    timestep: float
    gravity: float = 9.81
    render: bool = False
