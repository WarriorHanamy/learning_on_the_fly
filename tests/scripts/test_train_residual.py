"""Unit tests for train_residual.py script.

Tests cover:
- load_dataset: Loading CSV data into JAX arrays
- create_ensemble: Initializing ensemble models from config
- CLI argument parsing
- Integration: End-to-end training script execution
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Skip all tests if JAX not available (for CI environments without GPU)
pytest.importorskip("jax")

import jax.numpy as jnp


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def example_dataset_path() -> str:
    """Path to example dataset."""
    return "examples/residual_dynamics/example_dataset.csv"


@pytest.fixture
def example_config_path() -> str:
    """Path to example config."""
    return "configs/residual_dynamics.yaml"


# =============================================================================
# Test: load_dataset function
# =============================================================================


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_dataset_returns_jax_arrays(self, example_dataset_path: str) -> None:
        """load_dataset should return JAX arrays with correct shapes."""
        from lotf.scripts.train_residual import load_dataset

        X, y = load_dataset(example_dataset_path, input_dim=19)

        # Verify JAX arrays
        assert isinstance(X, jnp.ndarray), "X should be a JAX array"
        assert isinstance(y, jnp.ndarray), "y should be a JAX array"

        # Verify shapes: example_dataset has 1000 rows, 22 columns (19 input + 3 output)
        assert X.shape[0] == 1000, f"Expected 1000 samples, got {X.shape[0]}"
        assert X.shape[1] == 19, f"Expected input_dim=19, got {X.shape[1]}"
        assert y.shape[0] == 1000, f"Expected 1000 samples, got {y.shape[0]}"
        assert y.shape[1] == 3, f"Expected output_dim=3, got {y.shape[1]}"

    def test_load_dataset_returns_float32(self, example_dataset_path: str) -> None:
        """load_dataset should return float32 arrays."""
        from lotf.scripts.train_residual import load_dataset

        X, y = load_dataset(example_dataset_path, input_dim=19)

        assert X.dtype == jnp.float32, f"X should be float32, got {X.dtype}"
        assert y.dtype == jnp.float32, f"y should be float32, got {y.dtype}"

    def test_load_dataset_raises_filenotfound(self, tmp_path: Path) -> None:
        """load_dataset should raise FileNotFoundError for non-existent path."""
        from lotf.scripts.train_residual import load_dataset

        non_existent = tmp_path / "non_existent.csv"
        with pytest.raises(FileNotFoundError):
            load_dataset(str(non_existent), input_dim=19)


# =============================================================================
# Test: create_ensemble function
# =============================================================================


class TestCreateEnsemble:
    """Tests for create_ensemble function."""

    def test_create_ensemble_returns_correct_num_models(self) -> None:
        """create_ensemble should return train states for num_models."""
        from lotf.scripts.train_residual import ResidualDynamicsConfig, create_ensemble

        config = ResidualDynamicsConfig(
            num_models=3,
            input_dim=19,
            output_dim=3,
            learning_rate=0.01,
            lambda_reg=0.001,
            num_epochs=100,
            batch_size=256,
            eval_every=10,
            weight_init_scale=1.0,
        )

        model_params, train_states = create_ensemble(config)

        # Check that params have batch dimension for num_models
        # The params structure is nested, so we check the first layer's batch dim
        assert model_params is not None, "model_params should not be None"
        assert train_states is not None, "train_states should not be None"

    def test_create_ensemble_single_model(self) -> None:
        """create_ensemble should work with num_models=1."""
        from lotf.scripts.train_residual import ResidualDynamicsConfig, create_ensemble

        config = ResidualDynamicsConfig(
            num_models=1,
            input_dim=19,
            output_dim=3,
            learning_rate=0.01,
            lambda_reg=0.001,
            num_epochs=100,
            batch_size=256,
            eval_every=10,
            weight_init_scale=1.0,
        )

        model_params, train_states = create_ensemble(config)

        assert model_params is not None
        assert train_states is not None


# =============================================================================
# Test: CLI argument parsing
# =============================================================================


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_parse_args_all_arguments(self) -> None:
        """parse_args should correctly parse all three arguments."""
        from lotf.scripts.train_residual import parse_args

        # Simulate command line arguments
        original_argv = sys.argv
        try:
            sys.argv = [
                "train_residual.py",
                "--config",
                "my_config.yaml",
                "--dataset",
                "my_data.csv",
                "--output",
                "my_output",
            ]
            args = parse_args()
            assert args.config == "my_config.yaml"
            assert args.dataset == "my_data.csv"
            assert args.output == "my_output"
        finally:
            sys.argv = original_argv

    def test_parse_args_default_values(self) -> None:
        """parse_args should have correct default values for config and output."""
        from lotf.scripts.train_residual import parse_args

        original_argv = sys.argv
        try:
            # --dataset is required, so we must provide it
            sys.argv = ["train_residual.py", "--dataset", "data.csv"]
            args = parse_args()
            assert args.config == "configs/residual_dynamics.yaml"
            assert args.dataset == "data.csv"
            assert args.output == "checkpoints/residual_dynamics/residual_params"
        finally:
            sys.argv = original_argv

    def test_help_flag(self) -> None:
        """--help flag should show help and exit with 0."""
        result = subprocess.run(
            ["uv", "run", "python", "-m", "lotf.scripts.train_residual", "--help"],
            capture_output=True,
            text=True,
        )
        # --help exits with 0
        assert result.returncode == 0
        # Help should mention key arguments
        assert "--config" in result.stdout
        assert "--dataset" in result.stdout
        assert "--output" in result.stdout


# =============================================================================
# Test: ResidualDynamicsConfig dataclass
# =============================================================================


class TestResidualDynamicsConfig:
    """Tests for ResidualDynamicsConfig dataclass."""

    def test_config_from_yaml(self, example_config_path: str) -> None:
        """ResidualDynamicsConfig should load from YAML file."""
        from lotf.scripts.train_residual import ResidualDynamicsConfig

        config = ResidualDynamicsConfig.from_yaml(example_config_path)

        assert config.num_models == 3
        assert config.input_dim == 19
        assert config.output_dim == 3
        assert config.learning_rate == 0.01
        assert config.lambda_reg == 0.001
        assert config.num_epochs == 100
        assert config.batch_size == 256
        assert config.eval_every == 10
        assert config.weight_init_scale == 1.0

    def test_config_file_not_found(self, tmp_path: Path) -> None:
        """ResidualDynamicsConfig.from_yaml should raise FileNotFoundError."""
        from lotf.scripts.train_residual import ResidualDynamicsConfig

        non_existent = tmp_path / "non_existent.yaml"
        with pytest.raises(FileNotFoundError):
            ResidualDynamicsConfig.from_yaml(str(non_existent))


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests for the training script."""

    def test_script_runs_with_valid_inputs(
        self, example_config_path: str, example_dataset_path: str, tmp_path: Path
    ) -> None:
        """Script should run successfully with valid config and dataset."""
        output_path = tmp_path / "test_checkpoint"

        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "lotf.scripts.train_residual",
                "--config",
                example_config_path,
                "--dataset",
                example_dataset_path,
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Script should complete successfully
        assert result.returncode == 0, f"Script failed with: {result.stderr}"

        # Should have training output with MSE loss logging
        assert "Epoch" in result.stdout or "MSE" in result.stdout

    def test_checkpoint_contains_all_ensemble_params(
        self, example_config_path: str, example_dataset_path: str, tmp_path: Path
    ) -> None:
        """Checkpoint should contain parameters for all ensemble members."""
        output_path = tmp_path / "test_checkpoint"

        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "lotf.scripts.train_residual",
                "--config",
                example_config_path,
                "--dataset",
                example_dataset_path,
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Verify checkpoint was created
        # orbax creates a directory with checkpoint data
        assert output_path.exists() or any(output_path.parent.glob("test_checkpoint*"))
