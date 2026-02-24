"""Configuration module for LOTF (Learning on the Fly).

This module provides:
- Typed frozen dataclasses for configuration (TrainingConfig, EnvConfig, SimConfig)
- YAML configuration loader with validation
- Configuration merge utility for runtime overrides

Example:
    >>> from lotf.configs import load_config, TrainingConfig
    >>> config = load_config("config/training.yaml", TrainingConfig)
    >>> print(config.seed)
    42

    >>> # With runtime overrides
    >>> config = load_config("config/training.yaml", TrainingConfig, overrides={"seed": 123})
"""

from lotf.configs.configs import EnvConfig, SimConfig, TrainingConfig
from lotf.configs.loader import load_config, merge_config

__all__ = [
    "TrainingConfig",
    "EnvConfig",
    "SimConfig",
    "load_config",
    "merge_config",
]
