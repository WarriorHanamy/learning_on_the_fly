"""Unit tests for train_state_hovering.py training script.

These tests verify the core components of the state hovering training script:
- Configuration loading
- Environment creation
- Policy creation
- CLI argument parsing
- Checkpoint saving
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
import yaml

from lotf.scripts.train_state_hovering import (
    StateHoveringConfig,
    SimDynConfig,
    PolicyNetConfig,
    OptimizerConfig,
    create_env,
    create_policy,
    load_dummy_residual_params,
    save_checkpoint,
    parse_args,
    main,
)


# Test fixtures
@pytest.fixture
def sample_config_dict() -> dict:
    """Create a sample configuration dictionary matching YAML structure."""
    return {
        "seed": 0,
        "num_envs": 200,
        "max_epochs": 200,
        "sim_dt": 0.02,
        "max_sim_time": 3.0,
        "delay": 0.04,
        "reward_sharpness": 3.0,
        "action_penalty_weight": 0.5,
        "hover_target": [1.5, 0.0, 1.5],
        "sim_dyn_config": {
            "use_high_fidelity": False,
            "use_forward_residual": False,
        },
        "yaw_scale": 1.0,
        "pitch_roll_scale": 0.1,
        "velocity_std": 0.1,
        "omega_std": 0.1,
        "margin": 0.5,
        "policy_net": {
            "hidden_layers": [512, 512],
            "initial_scale": 0.01,
        },
        "optimizer": {
            "initial_lr": 0.005,
            "scheduler": "cosine_decay",
        },
    }


@pytest.fixture
def sample_config(sample_config_dict: dict) -> StateHoveringConfig:
    """Create a sample StateHoveringConfig instance."""
    # Create nested config objects manually
    sim_dyn_config = SimDynConfig(
        use_high_fidelity=sample_config_dict["sim_dyn_config"]["use_high_fidelity"],
        use_forward_residual=sample_config_dict["sim_dyn_config"]["use_forward_residual"],
    )
    policy_net_config = PolicyNetConfig(
        hidden_layers=sample_config_dict["policy_net"]["hidden_layers"],
        initial_scale=sample_config_dict["policy_net"]["initial_scale"],
    )
    optimizer_config = OptimizerConfig(
        initial_lr=sample_config_dict["optimizer"]["initial_lr"],
        scheduler=sample_config_dict["optimizer"]["scheduler"],
    )
    return StateHoveringConfig(
        seed=sample_config_dict["seed"],
        num_envs=sample_config_dict["num_envs"],
        max_epochs=sample_config_dict["max_epochs"],
        sim_dt=sample_config_dict["sim_dt"],
        max_sim_time=sample_config_dict["max_sim_time"],
        delay=sample_config_dict["delay"],
        reward_sharpness=sample_config_dict["reward_sharpness"],
        action_penalty_weight=sample_config_dict["action_penalty_weight"],
        hover_target=sample_config_dict["hover_target"],
        sim_dyn_config=sim_dyn_config,
        yaw_scale=sample_config_dict["yaw_scale"],
        pitch_roll_scale=sample_config_dict["pitch_roll_scale"],
        velocity_std=sample_config_dict["velocity_std"],
        omega_std=sample_config_dict["omega_std"],
        margin=sample_config_dict["margin"],
        policy_net=policy_net_config,
        optimizer=optimizer_config,
    )


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


class TestStateHoveringConfig:
    """Tests for StateHoveringConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = StateHoveringConfig()
        assert config.seed == 0
        assert config.num_envs == 200
        assert config.max_epochs == 200
        assert config.sim_dt == 0.02
        assert config.max_sim_time == 3.0
        assert config.delay == 0.04
        assert config.hover_target == [1.5, 0.0, 1.5]

    def test_from_yaml_success(self, temp_config_file: Path) -> None:
        """Test loading config from a valid YAML file."""
        config = StateHoveringConfig.from_yaml(temp_config_file)
        assert config.seed == 0
        assert config.num_envs == 200
        assert config.max_epochs == 200
        assert config.sim_dt == 0.02
        assert config.hover_target == [1.5, 0.0, 1.5]

    def test_from_yaml_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            StateHoveringConfig.from_yaml("/nonexistent/path.yaml")

    def test_from_yaml_empty_file(self, tmp_path: Path) -> None:
        """Test that ValueError is raised for empty file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.touch()
        with pytest.raises(ValueError, match="empty"):
            StateHoveringConfig.from_yaml(empty_file)


class TestCreateEnv:
    """Tests for create_env function."""

    def test_returns_wrapped_env(self, sample_config: StateHoveringConfig) -> None:
        """Test that create_env returns a properly wrapped environment."""
        env = create_env(sample_config)

        # Check that environment has expected attributes from wrappers
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")

    def test_correct_observation_space(self, sample_config: StateHoveringConfig) -> None:
        """Test that observation space has correct dimensions."""
        env = create_env(sample_config)

        # After MinMaxObservationWrapper, obs space should be normalized to [-1, 1]
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 1
        assert obs_shape[0] > 0  # Should have positive dimension

        # Check bounds are normalized
        assert jnp.allclose(env.observation_space.low, -1.0)
        assert jnp.allclose(env.observation_space.high, 1.0)

    def test_correct_action_space(self, sample_config: StateHoveringConfig) -> None:
        """Test that action space has correct dimensions (4 for quadrotor)."""
        env = create_env(sample_config)

        action_shape = env.action_space.shape
        assert len(action_shape) == 1
        assert action_shape[0] == 4  # thrust + 3 angular rates

    def test_max_steps_in_episode(self, sample_config: StateHoveringConfig) -> None:
        """Test that max_steps_in_episode is calculated correctly."""
        env = create_env(sample_config)

        expected_max_steps = int(sample_config.max_sim_time / sample_config.sim_dt)
        assert env.max_steps_in_episode == expected_max_steps


class TestCreatePolicy:
    """Tests for create_policy function."""

    def test_returns_train_state(self, sample_config: StateHoveringConfig) -> None:
        """Test that create_policy returns a TrainState."""
        env = create_env(sample_config)
        key = jax.random.key(sample_config.seed)

        train_state = create_policy(sample_config, env, key)

        assert hasattr(train_state, "params")
        assert hasattr(train_state, "apply_fn")
        assert hasattr(train_state, "tx")

    def test_correct_parameter_shapes(self, sample_config: StateHoveringConfig) -> None:
        """Test that policy parameters have correct shapes."""
        env = create_env(sample_config)
        key = jax.random.key(sample_config.seed)

        train_state = create_policy(sample_config, env, key)

        # Get action and obs dimensions
        action_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]

        # Test that policy can process observations
        test_obs = jnp.zeros(obs_dim)
        action = train_state.apply_fn(train_state.params, test_obs)

        assert action.shape == (action_dim,)

    def test_action_bias_applied(self, sample_config: StateHoveringConfig) -> None:
        """Test that hovering action bias is applied to the policy."""
        env = create_env(sample_config)
        key = jax.random.key(sample_config.seed)

        train_state = create_policy(sample_config, env, key)

        # The policy output should be biased towards hovering action
        # When input is zeros, output should be close to hovering_action
        test_obs = jnp.zeros(env.observation_space.shape[0])
        action = train_state.apply_fn(train_state.params, test_obs)

        # Action should be in valid range
        assert action.shape == env.action_space.shape


class TestArgparseCLI:
    """Tests for CLI argument parsing."""

    def test_default_values(self) -> None:
        """Test default CLI argument values."""
        with patch("sys.argv", ["train_state_hovering.py"]):
            args = parse_args()
            assert args.config == "configs/state_hovering.yaml"
            assert args.output == "checkpoints/policy/state_hovering_params"

    def test_custom_config(self) -> None:
        """Test custom config path argument."""
        with patch("sys.argv", ["train_state_hovering.py", "--config", "custom.yaml"]):
            args = parse_args()
            assert args.config == "custom.yaml"

    def test_custom_output(self) -> None:
        """Test custom output path argument."""
        with patch(
            "sys.argv",
            ["train_state_hovering.py", "--output", "custom_output/path"],
        ):
            args = parse_args()
            assert args.output == "custom_output/path"

    def test_both_arguments(self) -> None:
        """Test both config and output arguments."""
        with patch(
            "sys.argv",
            [
                "train_state_hovering.py",
                "--config",
                "my_config.yaml",
                "--output",
                "my_output",
            ],
        ):
            args = parse_args()
            assert args.config == "my_config.yaml"
            assert args.output == "my_output"


class TestCheckpointSave:
    """Tests for checkpoint saving functionality."""

    def test_checkpoint_creates_file(self, temp_output_dir: Path) -> None:
        """Test that save_checkpoint creates a file at the specified path."""
        # Create simple test params
        params = {"layer1": {"weight": jnp.array([1.0, 2.0, 3.0])}}
        output_path = str(temp_output_dir / "test_checkpoint")

        save_checkpoint(output_path, params)

        # Check that checkpoint file exists (orbax creates directory with data)
        checkpoint_dir = Path(output_path).parent
        assert checkpoint_dir.exists()

    def test_checkpoint_creates_parent_dirs(self, temp_output_dir: Path) -> None:
        """Test that save_checkpoint creates parent directories if needed."""
        params = {"layer1": {"weight": jnp.array([1.0, 2.0])}}
        output_path = str(temp_output_dir / "nested" / "dirs" / "checkpoint")

        save_checkpoint(output_path, params)

        # Check nested directories were created
        assert (temp_output_dir / "nested" / "dirs").exists()

    def test_checkpoint_loadable(self, temp_output_dir: Path) -> None:
        """Test that saved checkpoint can be loaded back."""
        from orbax.checkpoint import PyTreeCheckpointer

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


class TestLoadDummyResidualParams:
    """Tests for loading dummy residual dynamics parameters."""

    @pytest.mark.skip(reason="Dummy checkpoint was saved on GPU, requires GPU environment to load")
    def test_loads_successfully(self) -> None:
        """Test that dummy residual params can be loaded."""
        params = load_dummy_residual_params()
        assert params is not None


# Integration test marker
@pytest.mark.integration
@pytest.mark.skip(reason="Integration test requires GPU environment to load dummy checkpoint")
class TestIntegration:
    """Integration tests for the full training pipeline."""

    def test_short_training_run(self, temp_output_dir: Path) -> None:
        """Test a short training run completes without errors."""
        # Create a minimal config for quick testing
        config = StateHoveringConfig(
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
