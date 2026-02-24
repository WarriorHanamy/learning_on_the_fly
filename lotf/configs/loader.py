"""YAML configuration loader with validation and override support.

Provides utilities for loading configuration from YAML files into typed
frozen dataclasses, with support for runtime overrides.
"""

from dataclasses import MISSING, Field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from lotf.configs.configs import EnvConfig, SimConfig, TrainingConfig

T = TypeVar("T")


def _validate_config_dict(config_dict: dict[str, Any], config_class: type) -> dict[str, Any]:
    """Validate that all required fields are present in the config dict.

    Args:
        config_dict: Dictionary loaded from YAML.
        config_class: Target dataclass type.

    Returns:
        Validated config dict with only valid fields.

    Raises:
        ValueError: If required fields are missing.
    """
    if not is_dataclass(config_class):
        raise ValueError(f"{config_class.__name__} is not a dataclass")

    config_fields: tuple[Field, ...] = fields(config_class)
    field_names = {f.name for f in config_fields}
    required_fields = [
        f.name for f in config_fields if f.default is MISSING and f.default_factory is MISSING
    ]

    # Check for missing required fields
    missing_fields = [f for f in required_fields if f not in config_dict]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    # Filter to only valid fields (ignore extras)
    return {k: v for k, v in config_dict.items() if k in field_names}


def load_config(
    config_path: str | Path,
    config_class: type[T],
    *,
    overrides: dict[str, Any] | None = None,
) -> T:
    """Load configuration from a YAML file into a frozen dataclass.

    Args:
        config_path: Path to the YAML configuration file.
        config_class: Target dataclass type (e.g., TrainingConfig, EnvConfig).
        overrides: Optional dictionary of values to override after loading.

    Returns:
        A frozen dataclass instance with the loaded configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If YAML parsing fails or required fields are missing.
        IsADirectoryError: If the path points to a directory.
    """
    path = Path(config_path)

    # Validate path - handle empty string explicitly
    if not config_path or str(config_path).strip() == "":
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if path.is_dir():
        raise IsADirectoryError(f"Path is a directory, not a file: {config_path}")

    # Parse YAML
    try:
        content = path.read_text()
        config_dict = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file: {e}") from e

    # Handle empty file
    if config_dict is None:
        raise ValueError("Configuration file is empty")

    # Ensure config is a dict
    if not isinstance(config_dict, dict):
        raise ValueError(f"Configuration must be a YAML mapping, got {type(config_dict).__name__}")

    # Apply overrides if provided
    if overrides:
        config_dict = {**config_dict, **overrides}

    # Validate and filter fields
    validated_config = _validate_config_dict(config_dict, config_class)

    # Create dataclass instance
    return config_class(**validated_config)


def merge_config(base: T, overrides: dict[str, Any]) -> T:
    """Merge override values into a base config, returning a new instance.

    Since configs are frozen, this creates a new dataclass instance with
    the overridden values while preserving unspecified fields.

    Args:
        base: Base frozen dataclass config instance.
        overrides: Dictionary of values to override.

    Returns:
        A new frozen dataclass instance with merged values.

    Raises:
        ValueError: If override contains unknown fields.
        TypeError: If base is not a dataclass instance.
    """
    if not is_dataclass(base):
        raise TypeError(f"Expected dataclass instance, got {type(base).__name__}")

    config_class = type(base)
    config_fields = {f.name for f in fields(config_class)}

    # Check for unknown fields in overrides
    unknown_fields = set(overrides.keys()) - config_fields
    if unknown_fields:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown_fields))}")

    # Build merged config dict
    base_dict = {f.name: getattr(base, f.name) for f in fields(config_class)}
    merged_dict = {**base_dict, **overrides}

    return config_class(**merged_dict)


__all__ = [
    "TrainingConfig",
    "EnvConfig",
    "SimConfig",
    "load_config",
    "merge_config",
]
