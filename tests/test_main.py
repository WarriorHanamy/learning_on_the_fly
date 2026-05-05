"""Unit tests for lotf.__main__ CLI entry point.

These tests verify the CLI functionality:
- Version flag output
- List-configs flag output
- Argument parsing for subcommands
- Dispatch to training scripts
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lotf.__main__ import (
    create_parser,
    get_version,
    list_configs,
    main,
)


class TestVersion:
    """Test --version flag."""

    def test_version_flag_returns_string(self) -> None:
        """get_version should return a version string."""
        version = get_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_version_flag_exits_zero(self) -> None:
        """--version should exit with code 0."""
        exit_code = main(["--version"])
        assert exit_code == 0

    def test_version_output_contains_lotf(self, capsys: pytest.CaptureFixture) -> None:
        """--version output should contain 'lotf'."""
        main(["--version"])
        captured = capsys.readouterr()
        assert "lotf" in captured.out


class TestListConfigs:
    """Test --list-configs flag."""

    def test_list_configs_exits_zero(self) -> None:
        """--list-configs should exit with code 0."""
        exit_code = main(["--list-configs"])
        assert exit_code == 0

    def test_list_configs_shows_yaml_files(self, capsys: pytest.CaptureFixture) -> None:
        """--list-configs should show YAML files in configs/."""
        main(["--list-configs"])
        captured = capsys.readouterr()
        assert "state_hovering.yaml" in captured.out
        assert "traj_tracking.yaml" in captured.out
        assert "residual_dynamics.yaml" in captured.out

    def test_list_configs_missing_directory(self, tmp_path: Path) -> None:
        """list_configs returns 1 when configs directory is missing."""
        with patch("lotf.__main__.LOTF_ROOT", tmp_path):
            exit_code = list_configs()
            assert exit_code == 1


class TestArgumentParsing:
    """Test argument parsing for subcommands."""

    def test_parser_has_version_flag(self) -> None:
        """Parser should have --version flag."""
        parser = create_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_parser_has_list_configs_flag(self) -> None:
        """Parser should have --list-configs flag."""
        parser = create_parser()
        args = parser.parse_args(["--list-configs"])
        assert args.list_configs is True

    def test_hover_subcommand_default_config(self) -> None:
        """hover subcommand should have default config path."""
        parser = create_parser()
        args = parser.parse_args(["hover"])
        assert args.command == "hover"
        assert args.config == "configs/state_hovering.yaml"
        assert args.output == "checkpoints/policy/state_hovering_params"

    def test_hover_subcommand_custom_config(self) -> None:
        """hover subcommand should accept custom config path."""
        parser = create_parser()
        args = parser.parse_args(["hover", "--config", "my_config.yaml"])
        assert args.command == "hover"
        assert args.config == "my_config.yaml"

    def test_hover_subcommand_custom_output(self) -> None:
        """hover subcommand should accept custom output path."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "hover",
                "--config",
                "configs/state_hovering.yaml",
                "--output",
                "checkpoints/my_policy",
            ]
        )
        assert args.command == "hover"
        assert args.output == "checkpoints/my_policy"

    def test_track_subcommand_default_config(self) -> None:
        """track subcommand should have default config path."""
        parser = create_parser()
        args = parser.parse_args(["track"])
        assert args.command == "track"
        assert args.config == "configs/traj_tracking.yaml"
        assert args.checkpoint == "checkpoints/policy/traj_tracking_params"

    def test_track_subcommand_custom_checkpoint(self) -> None:
        """track subcommand should accept custom checkpoint path."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "track",
                "--checkpoint",
                "checkpoints/my_tracking",
            ]
        )
        assert args.command == "track"
        assert args.checkpoint == "checkpoints/my_tracking"

    def test_track_subcommand_trajectory_output(self) -> None:
        """track subcommand should accept trajectory output path."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "track",
                "--trajectory-output",
                "outputs/traj.csv",
            ]
        )
        assert args.command == "track"
        assert args.trajectory_output == "outputs/traj.csv"

    def test_residual_subcommand_requires_dataset(self) -> None:
        """residual subcommand should require --dataset."""
        parser = create_parser()
        # Should raise SystemExit when --dataset is missing
        with pytest.raises(SystemExit):
            parser.parse_args(["residual"])

    def test_residual_subcommand_with_dataset(self) -> None:
        """residual subcommand should accept --dataset."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "residual",
                "--dataset",
                "data.csv",
            ]
        )
        assert args.command == "residual"
        assert args.dataset == "data.csv"
        assert args.config == "configs/residual_dynamics.yaml"
        assert args.output == "checkpoints/residual_dynamics/residual_params"

    def test_residual_subcommand_custom_config(self) -> None:
        """residual subcommand should accept custom config."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "residual",
                "--config",
                "my_residual.yaml",
                "--dataset",
                "data.csv",
            ]
        )
        assert args.command == "residual"
        assert args.config == "my_residual.yaml"
        assert args.dataset == "data.csv"


class TestSubcommandDispatch:
    """Test dispatch to training scripts."""

    def test_no_subcommand_shows_help(self, capsys: pytest.CaptureFixture) -> None:
        """No subcommand should show help and exit with code 2."""
        exit_code = main([])
        captured = capsys.readouterr()
        assert exit_code == 2
        assert "subcommand is required" in captured.err

    @patch("lotf.__main__._run_with_argv")
    def test_hover_dispatches_to_train_state_hovering(self, mock_run: MagicMock) -> None:
        """hover subcommand should dispatch to train_state_hovering.main."""
        mock_run.return_value = 0
        main(["hover", "--config", "test.yaml", "--output", "out"])
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][1]  # Second argument is argv list
        assert "train_state_hovering" in call_args
        assert "--config" in call_args
        assert "test.yaml" in call_args
        assert "--output" in call_args
        assert "out" in call_args

    @patch("lotf.__main__._run_with_argv")
    def test_track_dispatches_to_train_traj_tracking(self, mock_run: MagicMock) -> None:
        """track subcommand should dispatch to train_traj_tracking.main."""
        mock_run.return_value = 0
        main(["track", "--config", "test.yaml", "--checkpoint", "ckpt"])
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][1]
        assert "train_traj_tracking" in call_args
        assert "--config" in call_args
        assert "test.yaml" in call_args
        assert "--checkpoint" in call_args
        assert "ckpt" in call_args

    @patch("lotf.__main__._run_with_argv")
    def test_track_with_trajectory_output(self, mock_run: MagicMock) -> None:
        """track subcommand should pass trajectory output."""
        mock_run.return_value = 0
        main(["track", "--trajectory-output", "traj.csv"])
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][1]
        assert "--trajectory-output" in call_args
        assert "traj.csv" in call_args

    @patch("lotf.__main__._run_with_argv")
    def test_residual_dispatches_to_train_residual(self, mock_run: MagicMock) -> None:
        """residual subcommand should dispatch to train_residual.main."""
        mock_run.return_value = 0
        main(["residual", "--config", "test.yaml", "--dataset", "data.csv", "--output", "out"])
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][1]
        assert "train_residual" in call_args
        assert "--config" in call_args
        assert "test.yaml" in call_args
        assert "--dataset" in call_args
        assert "data.csv" in call_args
        assert "--output" in call_args
        assert "out" in call_args


class TestHelp:
    """Test --help flag."""

    def test_help_exits_zero(self) -> None:
        """--help should exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_hover_help_exits_zero(self) -> None:
        """hover --help should exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["hover", "--help"])
        assert exc_info.value.code == 0

    def test_track_help_exits_zero(self) -> None:
        """track --help should exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["track", "--help"])
        assert exc_info.value.code == 0

    def test_residual_help_exits_zero(self) -> None:
        """residual --help should exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["residual", "--help"])
        assert exc_info.value.code == 0
