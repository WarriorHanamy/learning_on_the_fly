#!/usr/bin/env python3
"""Tests for build command scenario."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestBuildCommand:
    """Test build command functionality."""

    def test_build_command_checks_dockerfile_exists(self):
        """Test that build command verifies Dockerfile exists."""
        from dockerfile_utils import cmd_build

        args = type("Args", (), {})()
        original_cwd = Path.cwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            try:
                os.chdir(temp_dir)
                # Remove Dockerfile from temp dir
                dockerfile_path = temp_dir / "Dockerfile"
                if dockerfile_path.exists():
                    dockerfile_path.unlink()

                result = cmd_build(args)
                assert result == 1, "Build should return 1 when Dockerfile does not exist"
            finally:
                os.chdir(original_cwd)

    @patch("subprocess.run")
    def test_build_command_runs_docker_build(self, mock_run):
        """Test that build command runs docker build with correct tag."""
        from dockerfile_utils import cmd_build

        mock_run.return_value = MagicMock(returncode=0)
        args = type("Args", (), {})()

        original_cwd = Path.cwd()
        try:
            os.chdir(PROJECT_ROOT)
            result = cmd_build(args)
            assert result == 0, "Build should return 0 on success"

            # Verify docker build command was called
            assert mock_run.called, "subprocess.run should be called"
            cmd_args = mock_run.call_args[0][0]
            assert "docker" in cmd_args, "Docker command should be used"
            assert "build" in cmd_args, "docker build should be called"
            assert "-t" in cmd_args, "Tag flag should be present"
            assert "lotf:latest" in cmd_args, "lotf:latest tag should be used"
        finally:
            os.chdir(original_cwd)
