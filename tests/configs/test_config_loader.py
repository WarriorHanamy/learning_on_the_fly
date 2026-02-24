"""Unit tests for YAML config loader module.

Tests cover:
- load_config function with valid/invalid inputs
- Dataclass config definitions (TrainingConfig, EnvConfig, SimConfig)
- merge_config utility for combining configs with overrides
"""

import tempfile
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import yaml

# These imports will fail initially - that's expected in TDD RED phase
from lotf.configs import (
    EnvConfig,
    SimConfig,
    TrainingConfig,
    load_config,
    merge_config,
)


class TestTrainingConfigFields:
    """Test TrainingConfig dataclass has required fields."""

    def test_training_config_has_seed_field(self):
        """TrainingConfig must have 'seed' field."""
        field_names = [f.name for f in fields(TrainingConfig)]
        assert "seed" in field_names, "TrainingConfig missing 'seed' field"

    def test_training_config_has_num_envs_field(self):
        """TrainingConfig must have 'num_envs' field."""
        field_names = [f.name for f in fields(TrainingConfig)]
        assert "num_envs" in field_names, "TrainingConfig missing 'num_envs' field"

    def test_training_config_has_max_epochs_field(self):
        """TrainingConfig must have 'max_epochs' field."""
        field_names = [f.name for f in fields(TrainingConfig)]
        assert "max_epochs" in field_names, "TrainingConfig missing 'max_epochs' field"

    def test_training_config_has_learning_rate_field(self):
        """TrainingConfig must have 'learning_rate' field."""
        field_names = [f.name for f in fields(TrainingConfig)]
        assert "learning_rate" in field_names, "TrainingConfig missing 'learning_rate' field"


class TestEnvConfigFields:
    """Test EnvConfig dataclass has required fields."""

    def test_env_config_has_dt_field(self):
        """EnvConfig must have 'dt' field."""
        field_names = [f.name for f in fields(EnvConfig)]
        assert "dt" in field_names, "EnvConfig missing 'dt' field"

    def test_env_config_has_delay_field(self):
        """EnvConfig must have 'delay' field."""
        field_names = [f.name for f in fields(EnvConfig)]
        assert "delay" in field_names, "EnvConfig missing 'delay' field"

    def test_env_config_has_max_steps_in_episode_field(self):
        """EnvConfig must have 'max_steps_in_episode' field."""
        field_names = [f.name for f in fields(EnvConfig)]
        assert "max_steps_in_episode" in field_names, (
            "EnvConfig missing 'max_steps_in_episode' field"
        )


class TestSimConfigExists:
    """Test SimConfig dataclass exists and is properly defined."""

    def test_sim_config_is_dataclass(self):
        """SimConfig must be a dataclass."""
        assert is_dataclass(SimConfig), "SimConfig must be a dataclass"

    def test_sim_config_is_frozen(self):
        """SimConfig must be frozen (immutable)."""
        # Check if the dataclass is frozen by inspecting its __dataclass_fields__
        import dataclasses

        # Get the frozen attribute from the dataclass definition
        # We check by trying to set an attribute after creation
        # But for frozen dataclass, we can't do that, so we check the decorator params
        # A simpler way: frozen dataclasses have __dataclass_params__.frozen == True
        params = getattr(SimConfig, "__dataclass_params__", None)
        if params is not None:
            assert params.frozen, "SimConfig must be a frozen dataclass"


class TestConfigsAreFrozen:
    """Test all configs are frozen (immutable)."""

    def test_training_config_is_frozen(self):
        """TrainingConfig must be frozen."""
        params = getattr(TrainingConfig, "__dataclass_params__", None)
        if params is not None:
            assert params.frozen, "TrainingConfig must be a frozen dataclass"

    def test_env_config_is_frozen(self):
        """EnvConfig must be frozen."""
        params = getattr(EnvConfig, "__dataclass_params__", None)
        if params is not None:
            assert params.frozen, "EnvConfig must be a frozen dataclass"


class TestLoadConfigValidYAML:
    """Test load_config with valid YAML files."""

    def test_load_training_config_from_valid_yaml(self, tmp_path: Path):
        """load_config returns TrainingConfig instance from valid YAML."""
        config_file = tmp_path / "training.yaml"
        config_data = {
            "seed": 42,
            "num_envs": 16,
            "max_epochs": 100,
            "learning_rate": 0.001,
        }
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file, TrainingConfig)

        assert isinstance(config, TrainingConfig)
        assert config.seed == 42
        assert config.num_envs == 16
        assert config.max_epochs == 100
        assert config.learning_rate == 0.001

    def test_load_env_config_from_valid_yaml(self, tmp_path: Path):
        """load_config returns EnvConfig instance from valid YAML."""
        config_file = tmp_path / "env.yaml"
        config_data = {
            "dt": 0.01,
            "delay": 0.005,
            "max_steps_in_episode": 1000,
        }
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file, EnvConfig)

        assert isinstance(config, EnvConfig)
        assert config.dt == 0.01
        assert config.delay == 0.005
        assert config.max_steps_in_episode == 1000

    def test_load_config_with_path_object(self, tmp_path: Path):
        """load_config accepts pathlib.Path object."""
        config_file = tmp_path / "training.yaml"
        config_data = {
            "seed": 1,
            "num_envs": 1,
            "max_epochs": 1,
            "learning_rate": 0.1,
        }
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file, TrainingConfig)

        assert isinstance(config, TrainingConfig)

    def test_load_config_with_string_path(self, tmp_path: Path):
        """load_config accepts string path."""
        config_file = tmp_path / "training.yaml"
        config_data = {
            "seed": 1,
            "num_envs": 1,
            "max_epochs": 1,
            "learning_rate": 0.1,
        }
        config_file.write_text(yaml.dump(config_data))

        config = load_config(str(config_file), TrainingConfig)

        assert isinstance(config, TrainingConfig)


class TestLoadConfigInvalidPath:
    """Test load_config raises FileNotFoundError for invalid paths."""

    def test_non_existent_path_raises_filenotfounderror(self):
        """load_config raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config("/non/existent/path/config.yaml", TrainingConfig)

        # Check that the path is mentioned in the error message
        assert "/non/existent/path/config.yaml" in str(exc_info.value)

    def test_empty_string_path_raises_filenotfounderror(self):
        """load_config raises FileNotFoundError for empty string path."""
        with pytest.raises(FileNotFoundError):
            load_config("", TrainingConfig)

    def test_directory_path_raises_appropriate_error(self, tmp_path: Path):
        """load_config raises appropriate error for directory path."""
        with pytest.raises((FileNotFoundError, IsADirectoryError, ValueError)):
            load_config(tmp_path, TrainingConfig)


class TestLoadConfigMalformedYAML:
    """Test load_config raises ValueError for malformed YAML content."""

    def test_invalid_yaml_syntax_raises_valueerror(self, tmp_path: Path):
        """load_config raises ValueError for invalid YAML syntax."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [unclosed")

        with pytest.raises(ValueError) as exc_info:
            load_config(config_file, TrainingConfig)

        # Error message should mention YAML parsing
        assert "yaml" in str(exc_info.value).lower() or "parse" in str(exc_info.value).lower()

    def test_empty_file_raises_appropriate_error(self, tmp_path: Path):
        """load_config raises appropriate error for empty file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError):
            load_config(config_file, TrainingConfig)

    def test_non_dict_yaml_raises_appropriate_error(self, tmp_path: Path):
        """load_config raises error when YAML content is not a dict."""
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2")

        with pytest.raises(ValueError):
            load_config(config_file, TrainingConfig)


class TestLoadConfigMissingFields:
    """Test load_config raises ValueError for missing required fields."""

    def test_missing_single_field_raises_valueerror(self, tmp_path: Path):
        """load_config raises ValueError with field name when field is missing."""
        config_file = tmp_path / "incomplete.yaml"
        config_data = {
            "seed": 42,
            "num_envs": 16,
            # missing max_epochs and learning_rate
        }
        config_file.write_text(yaml.dump(config_data))

        with pytest.raises(ValueError) as exc_info:
            load_config(config_file, TrainingConfig)

        # Error should mention the missing field
        error_msg = str(exc_info.value).lower()
        assert "max_epochs" in error_msg or "learning_rate" in error_msg or "missing" in error_msg

    def test_missing_multiple_fields_raises_valueerror(self, tmp_path: Path):
        """load_config raises ValueError when multiple fields are missing."""
        config_file = tmp_path / "very_incomplete.yaml"
        config_data = {
            "seed": 42,
            # missing most fields
        }
        config_file.write_text(yaml.dump(config_data))

        with pytest.raises(ValueError):
            load_config(config_file, TrainingConfig)

    def test_extra_fields_are_ignored(self, tmp_path: Path):
        """Extra fields in YAML are ignored (not an error)."""
        config_file = tmp_path / "extra.yaml"
        config_data = {
            "seed": 42,
            "num_envs": 16,
            "max_epochs": 100,
            "learning_rate": 0.001,
            "extra_field": "should be ignored",
            "another_extra": 123,
        }
        config_file.write_text(yaml.dump(config_data))

        # Should not raise - extra fields are ignored
        config = load_config(config_file, TrainingConfig)
        assert config.seed == 42


class TestMergeConfig:
    """Test merge_config utility for combining configs with overrides."""

    def test_merge_config_overrides_scalar_values(self):
        """merge_config correctly overrides scalar values from dict."""
        base = TrainingConfig(seed=42, num_envs=16, max_epochs=100, learning_rate=0.001)
        overrides = {"seed": 123, "learning_rate": 0.01}

        merged = merge_config(base, overrides)

        assert merged.seed == 123  # overridden
        assert merged.learning_rate == 0.01  # overridden
        assert merged.num_envs == 16  # preserved
        assert merged.max_epochs == 100  # preserved

    def test_merge_config_preserves_unspecified_values(self):
        """merge_config preserves values not in override dict."""
        base = EnvConfig(dt=0.01, delay=0.005, max_steps_in_episode=1000)
        overrides = {"dt": 0.02}

        merged = merge_config(base, overrides)

        assert merged.dt == 0.02  # overridden
        assert merged.delay == 0.005  # preserved
        assert merged.max_steps_in_episode == 1000  # preserved

    def test_merge_config_returns_new_instance(self):
        """merge_config returns a new frozen dataclass instance."""
        base = TrainingConfig(seed=42, num_envs=16, max_epochs=100, learning_rate=0.001)
        overrides = {"seed": 123}

        merged = merge_config(base, overrides)

        # Should be a new instance
        assert merged is not base
        # Base should be unchanged (frozen anyway)
        assert base.seed == 42
        assert merged.seed == 123

    def test_merge_config_with_empty_overrides(self):
        """merge_config with empty dict returns equivalent config."""
        base = EnvConfig(dt=0.01, delay=0.005, max_steps_in_episode=1000)
        overrides = {}

        merged = merge_config(base, overrides)

        assert merged.dt == base.dt
        assert merged.delay == base.delay
        assert merged.max_steps_in_episode == base.max_steps_in_episode

    def test_merge_config_with_unknown_field_raises_error(self):
        """merge_config raises error for unknown fields in override dict."""
        base = TrainingConfig(seed=42, num_envs=16, max_epochs=100, learning_rate=0.001)
        overrides = {"unknown_field": 123}

        with pytest.raises((ValueError, TypeError, KeyError)):
            merge_config(base, overrides)


class TestLoadConfigWithOverrides:
    """Test load_config with optional overrides parameter."""

    def test_load_config_with_overrides_dict(self, tmp_path: Path):
        """load_config applies overrides dict to loaded config."""
        config_file = tmp_path / "training.yaml"
        config_data = {
            "seed": 42,
            "num_envs": 16,
            "max_epochs": 100,
            "learning_rate": 0.001,
        }
        config_file.write_text(yaml.dump(config_data))

        overrides = {"seed": 999, "num_envs": 32}
        config = load_config(config_file, TrainingConfig, overrides=overrides)

        assert config.seed == 999  # overridden
        assert config.num_envs == 32  # overridden
        assert config.max_epochs == 100  # from file
        assert config.learning_rate == 0.001  # from file
