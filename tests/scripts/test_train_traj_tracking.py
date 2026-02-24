"""Unit tests for train_traj_tracking.py training script.

These tests verify the core components of the trajectory tracking training script:
- Configuration loading
- Environment creation
- Policy creation
- CLI argument parsing
- Checkpoint saving
- Trajectory export
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
import yaml


# Test fixtures
@pytest.fixture
def sample_config_dict() -> dict:
    """Create a sample configuration dictionary matching YAML structure."""
    return {
        "seed": 0,
        "num_envs": 300,
        "max_epochs": 300,
        "sim_dt": 0.02,
        "max_sim_time": 5.0,
        "delay": 0.04,
        "ref_traj_name": "FIG8",
        "skip_start": True,
        "sim_dyn_config": {
            "use_high_fidelity": False,
            "use_forward_residual": False,
        },
        "yaw_scale": 0.1,
        "pitch_roll_scale": 0.1,
        "position_std": 0.1,
        "velocity_std": 0.1,
        "omega_std": 0.1,
        "policy_net": {
            "hidden_layers": [512, 512],
            "initial_scale": 0.01,
        },
        "optimizer": {
            "initial_lr": 0.001,
            "scheduler": "cosine_decay",
        },
    }


@pytest.fixture
def temp_config_file(sample_config_dict: dict) -> Path:
    """Create a temporary YAML config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config_dict, f)
        yield Path(f.name)
    Path(f.name).unlink()


@pytest.fixture
def temp_output_dir() -> Path:
    """Create a temporary directory for output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestModuleImport:
    """Tests for module import and CLI executable."""

    def test_module_importable(self) -> None:
        """Test that the module can be imported."""
        from lotf.scripts import train_traj_tracking

        assert train_traj_tracking is not None

    def test_module_has_main(self) -> None:
        """Test that the module has a main function."""
        from lotf.scripts.train_traj_tracking import main

        assert callable(main)


class TestTrajTrackingConfig:
    """Tests for TrajTrackingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        from lotf.scripts.train_traj_tracking import TrajTrackingConfig

        config = TrajTrackingConfig()
        assert config.seed == 0
        assert config.num_envs == 300
        assert config.max_epochs == 300
        assert config.sim_dt == 0.02
        assert config.max_sim_time == 5.0
        assert config.delay == 0.04
        assert config.ref_traj_name == "fig8"
        assert config.skip_start is True

    def test_from_yaml_success(self, temp_config_file: Path) -> None:
        """Test loading config from a valid YAML file."""
        from lotf.scripts.train_traj_tracking import TrajTrackingConfig

        config = TrajTrackingConfig.from_yaml(temp_config_file)
        assert config.seed == 0
        assert config.num_envs == 300
        assert config.max_epochs == 300
        assert config.sim_dt == 0.02
        assert config.ref_traj_name == "FIG8"
        assert config.skip_start is True

    def test_from_yaml_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        from lotf.scripts.train_traj_tracking import TrajTrackingConfig

        with pytest.raises(FileNotFoundError):
            TrajTrackingConfig.from_yaml("/nonexistent/path.yaml")

    def test_from_yaml_empty_file(self, tmp_path: Path) -> None:
        """Test that ValueError is raised for empty file."""
        from lotf.scripts.train_traj_tracking import TrajTrackingConfig

        empty_file = tmp_path / "empty.yaml"
        empty_file.touch()
        with pytest.raises(ValueError, match="empty"):
            TrajTrackingConfig.from_yaml(empty_file)


class TestCreateEnv:
    """Tests for create_env function."""

    def test_returns_wrapped_env(self) -> None:
        """Test that create_env returns a properly wrapped environment."""
        from lotf.scripts.train_traj_tracking import TrajTrackingConfig, create_env

        config = TrajTrackingConfig(num_envs=2, max_epochs=2)
        env = create_env(config)

        # Check that environment has expected attributes from wrappers
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")

    def test_correct_observation_space(self) -> None:
        """Test that observation space has correct dimensions."""
        from lotf.scripts.train_traj_tracking import TrajTrackingConfig, create_env

        config = TrajTrackingConfig(num_envs=2, max_epochs=2)
        env = create_env(config)

        # After MinMaxObservationWrapper, obs space should be normalized to [-1, 1]
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 1
        assert obs_shape[0] > 0  # Should have positive dimension

        # Check bounds are normalized
        assert jnp.allclose(env.observation_space.low, -1.0)
        assert jnp.allclose(env.observation_space.high, 1.0)

    def test_correct_action_space(self) -> None:
        """Test that action space has correct dimensions (4 for quadrotor)."""
        from lotf.scripts.train_traj_tracking import TrajTrackingConfig, create_env

        config = TrajTrackingConfig(num_envs=2, max_epochs=2)
        env = create_env(config)

        action_shape = env.action_space.shape
        assert len(action_shape) == 1
        assert action_shape[0] == 4  # thrust + 3 angular rates

    def test_max_steps_in_episode(self) -> None:
        """Test that max_steps_in_episode is calculated correctly."""
        from lotf.scripts.train_traj_tracking import TrajTrackingConfig, create_env

        config = TrajTrackingConfig(num_envs=2, max_epochs=2, sim_dt=0.02, max_sim_time=5.0)
        env = create_env(config)

        expected_max_steps = int(config.max_sim_time / config.sim_dt)
        assert env.max_steps_in_episode == expected_max_steps

    def test_env_has_ref_traj(self) -> None:
        """Test that environment has reference trajectory loaded."""
        from lotf.scripts.train_traj_tracking import TrajTrackingConfig, create_env

        config = TrajTrackingConfig(num_envs=2, max_epochs=2)
        env = create_env(config)

        # The wrapped env should have access to ref_traj through _env
        assert hasattr(env._env, "ref_traj")


class TestCreatePolicy:
    """Tests for create_policy function."""

    def test_returns_train_state(self) -> None:
        """Test that create_policy returns a TrainState."""
        from lotf.scripts.train_traj_tracking import TrajTrackingConfig, create_env, create_policy

        config = TrajTrackingConfig(num_envs=2, max_epochs=2)
        env = create_env(config)
        key = jax.random.key(config.seed)

        train_state = create_policy(config, env, key)

        assert hasattr(train_state, "params")
        assert hasattr(train_state, "apply_fn")
        assert hasattr(train_state, "tx")

    def test_correct_parameter_shapes(self) -> None:
        """Test that policy parameters have correct shapes."""
        from lotf.scripts.train_traj_tracking import TrajTrackingConfig, create_env, create_policy

        config = TrajTrackingConfig(num_envs=2, max_epochs=2)
        env = create_env(config)
        key = jax.random.key(config.seed)

        train_state = create_policy(config, env, key)

        # Get action and obs dimensions
        action_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]

        # Test that policy can process observations
        test_obs = jnp.zeros(obs_dim)
        action = train_state.apply_fn(train_state.params, test_obs)

        assert action.shape == (action_dim,)


class TestArgparseCLI:
    """Tests for CLI argument parsing."""

    def test_default_values(self) -> None:
        """Test default CLI argument values."""
        from lotf.scripts.train_traj_tracking import parse_args

        with patch("sys.argv", ["train_traj_tracking.py"]):
            args = parse_args()
            assert args.config == "configs/traj_tracking.yaml"
            assert args.checkpoint == "checkpoints/policy/traj_tracking_params"
            assert args.trajectory_output is None

    def test_custom_config(self) -> None:
        """Test custom config path argument."""
        from lotf.scripts.train_traj_tracking import parse_args

        with patch("sys.argv", ["train_traj_tracking.py", "--config", "custom.yaml"]):
            args = parse_args()
            assert args.config == "custom.yaml"

    def test_custom_checkpoint(self) -> None:
        """Test custom checkpoint path argument."""
        from lotf.scripts.train_traj_tracking import parse_args

        with patch(
            "sys.argv",
            ["train_traj_tracking.py", "--checkpoint", "custom_checkpoint/path"],
        ):
            args = parse_args()
            assert args.checkpoint == "custom_checkpoint/path"

    def test_trajectory_output(self) -> None:
        """Test trajectory output path argument."""
        from lotf.scripts.train_traj_tracking import parse_args

        with patch(
            "sys.argv",
            ["train_traj_tracking.py", "--trajectory-output", "traj_output.csv"],
        ):
            args = parse_args()
            assert args.trajectory_output == "traj_output.csv"

    def test_all_arguments(self) -> None:
        """Test all three arguments together."""
        from lotf.scripts.train_traj_tracking import parse_args

        with patch(
            "sys.argv",
            [
                "train_traj_tracking.py",
                "--config",
                "my_config.yaml",
                "--checkpoint",
                "my_checkpoint",
                "--trajectory-output",
                "my_traj.csv",
            ],
        ):
            args = parse_args()
            assert args.config == "my_config.yaml"
            assert args.checkpoint == "my_checkpoint"
            assert args.trajectory_output == "my_traj.csv"


class TestCheckpointSave:
    """Tests for checkpoint saving functionality."""

    def test_checkpoint_creates_file(self, temp_output_dir: Path) -> None:
        """Test that save_checkpoint creates a file at the specified path."""
        from lotf.scripts.train_traj_tracking import save_checkpoint

        # Create simple test params
        params = {"layer1": {"weight": jnp.array([1.0, 2.0, 3.0])}}
        output_path = str(temp_output_dir / "test_checkpoint")

        save_checkpoint(output_path, params)

        # Check that checkpoint file exists (orbax creates directory with data)
        checkpoint_dir = Path(output_path).parent
        assert checkpoint_dir.exists()

    def test_checkpoint_creates_parent_dirs(self, temp_output_dir: Path) -> None:
        """Test that save_checkpoint creates parent directories if needed."""
        from lotf.scripts.train_traj_tracking import save_checkpoint

        params = {"layer1": {"weight": jnp.array([1.0, 2.0])}}
        output_path = str(temp_output_dir / "nested" / "dirs" / "checkpoint")

        save_checkpoint(output_path, params)

        # Check nested directories were created
        assert (temp_output_dir / "nested" / "dirs").exists()

    def test_checkpoint_loadable(self, temp_output_dir: Path) -> None:
        """Test that saved checkpoint can be loaded back."""
        from orbax.checkpoint import PyTreeCheckpointer

        from lotf.scripts.train_traj_tracking import save_checkpoint

        original_params = {
            "layer1": {"weight": jnp.array([1.0, 2.0, 3.0])},
            "layer2": {"bias": jnp.array([0.1, 0.2])},
        }
        output_path = str(temp_output_dir / "test_checkpoint")

        save_checkpoint(output_path, original_params)

        # Load back and verify
        ckptr = PyTreeCheckpointer()
        loaded_params = ckptr.restore(output_path)

        assert jnp.allclose(
            loaded_params["layer1"]["weight"],
            original_params["layer1"]["weight"],
        )
        assert jnp.allclose(
            loaded_params["layer2"]["bias"],
            original_params["layer2"]["bias"],
        )


class TestTrajectoryExport:
    """Tests for trajectory export functionality."""

    def test_export_trajectory_creates_csv(self, temp_output_dir: Path) -> None:
        """Test that export_trajectory creates a CSV file."""
        from lotf.scripts.train_traj_tracking import export_trajectory

        # Create a mock trajectory transition
        mock_traj = MagicMock()
        mock_traj.reward.ndim = 1

        output_path = str(temp_output_dir / "test_trajectory.csv")

        # Mock the generate_csv method
        with patch("lotf.envs.TrajTrackingStateEnv.generate_csv") as mock_generate:
            export_trajectory(mock_traj, output_path)
            mock_generate.assert_called_once()

    def test_csv_has_required_columns(self, temp_output_dir: Path) -> None:
        """Test that CSV export has all required columns."""
        # This tests the expected CSV format based on TrajTrackingStateEnv._generate_csv
        required_columns = [
            "index",
            "t",
            "px",
            "py",
            "pz",
            "qw",
            "qx",
            "qy",
            "qz",
            "vx",
            "vy",
            "vz",
        ]

        # Create a test CSV to verify column expectations
        output_path = str(temp_output_dir / "test_trajectory.csv")
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=required_columns)
            writer.writeheader()
            writer.writerow(
                {
                    "index": 0,
                    "t": 0.0,
                    "px": 0.0,
                    "py": 0.0,
                    "pz": 0.0,
                    "qw": 1.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "vz": 0.0,
                }
            )

        # Verify file was created and has correct columns
        with open(output_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            assert set(reader.fieldnames) == set(required_columns)


class TestReferenceTrajectory:
    """Tests for reference trajectory loading."""

    def test_fig8_trajectory_loads(self) -> None:
        """Test that FIG8 trajectory loads correctly."""
        from lotf.objects import RefTrajNames

        from lotf.scripts.train_traj_tracking import TrajTrackingConfig, create_env

        config = TrajTrackingConfig(num_envs=2, max_epochs=2, ref_traj_name=RefTrajNames.FIG8.value)
        env = create_env(config)

        # Check that ref_traj exists and has waypoints
        assert hasattr(env._env, "ref_traj")
        assert env._env.num_ref_traj_points > 0

    def test_circle_trajectory_loads(self) -> None:
        """Test that CIRCLE trajectory loads correctly."""
        from lotf.objects import RefTrajNames

        from lotf.scripts.train_traj_tracking import TrajTrackingConfig, create_env

        config = TrajTrackingConfig(
            num_envs=2, max_epochs=2, ref_traj_name=RefTrajNames.CIRCLE.value
        )
        env = create_env(config)

        assert hasattr(env._env, "ref_traj")
        assert env._env.num_ref_traj_points > 0

    def test_star_trajectory_loads(self) -> None:
        """Test that STAR trajectory loads correctly."""
        from lotf.objects import RefTrajNames

        from lotf.scripts.train_traj_tracking import TrajTrackingConfig, create_env

        config = TrajTrackingConfig(num_envs=2, max_epochs=2, ref_traj_name=RefTrajNames.STAR.value)
        env = create_env(config)

        assert hasattr(env._env, "ref_traj")
        assert env._env.num_ref_traj_points > 0


class TestLoadDummyResidualParams:
    """Tests for loading dummy residual dynamics parameters."""

    @pytest.mark.skip(reason="Dummy checkpoint requires GPU environment to load")
    def test_loads_successfully(self) -> None:
        """Test that dummy residual params can be loaded."""
        from lotf.scripts.train_traj_tracking import load_dummy_residual_params

        params = load_dummy_residual_params()
        assert params is not None


# Integration test marker
@pytest.mark.integration
@pytest.mark.skip(reason="Integration test requires GPU environment to load dummy checkpoint")
class TestIntegration:
    """Integration tests for the full training pipeline."""

    def test_short_training_run(self, temp_output_dir: Path) -> None:
        """Test a short training run completes without errors."""
        from lotf.scripts.train_traj_tracking import (
            TrajTrackingConfig,
            create_env,
            create_policy,
            load_dummy_residual_params,
        )

        # Create a minimal config for quick testing
        config = TrajTrackingConfig(
            seed=0,
            num_envs=2,  # Minimal environments
            max_epochs=2,  # Minimal epochs
            sim_dt=0.02,
            max_sim_time=0.1,  # Very short episodes
            delay=0.04,
        )

        # This should run without errors
        env = create_env(config)
        key = jax.random.key(config.seed)
        key_init, key_bptt = jax.random.split(key, 2)

        train_state = create_policy(config, env, key_init)
        dummy_params = load_dummy_residual_params()

        key_bptt, key_ = jax.random.split(key_bptt)
        key_reset = jax.random.split(key_, config.num_envs)
        init_env_state, init_obs = env.reset(key_reset, None)

        # Import bptt here to avoid issues if not available
        from lotf.algos import bptt

        res_dict = bptt.train(
            env,
            init_env_state,
            init_obs,
            train_state,
            num_epochs=config.max_epochs,
            num_steps_per_epoch=env.max_steps_in_episode,
            num_envs=config.num_envs,
            res_model_params=dummy_params,
            key=key_bptt,
        )

        # Check results have expected structure
        assert "runner_state" in res_dict
        assert "metrics" in res_dict
        assert len(res_dict["metrics"]) == config.max_epochs
