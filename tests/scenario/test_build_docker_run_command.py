"""Tests for build_docker_run_command function."""

import subprocess
import sys
from pathlib import Path


def test_build_docker_run_command_with_all_params():
    """Test that build_docker_run_command includes all fixed parameters."""
    dockrun_path = Path(__file__).parent.parent.parent / "dockrun.py"

    if not dockrun_path.exists():
        assert False, "dockrun.py does not exist yet"

    # Import the module to test the function
    sys.path.insert(0, str(dockrun_path.parent))
    import dockrun

    # Build command with a test command
    cmd = dockrun.build_docker_run_command("echo 'hello world'")
    cmd_str = " ".join(cmd)

    # Verify all fixed parameters are present
    assert "docker" in cmd_str, "Command should contain 'docker'"
    assert "run" in cmd_str, "Command should contain 'run'"
    assert "lotf:latest" in cmd_str, "Command should contain image lotf:latest"
    assert "--gpus=all" in cmd_str, "Command should contain --gpus=all"
    assert "--rm" in cmd_str, "Command should contain --rm"
    assert "-w" in cmd_str, "Command should contain -w flag"
    assert "/app" in cmd_str, "Command should contain /app work directory"
    assert "-v" in cmd_str, "Command should contain -v flag"
    assert "echo 'hello world'" in cmd_str, "Command should contain user command"


def test_build_docker_run_command_with_empty_command():
    """Test that build_docker_run_command handles empty command."""
    dockrun_path = Path(__file__).parent.parent.parent / "dockrun.py"

    if not dockrun_path.exists():
        assert False, "dockrun.py does not exist yet"

    sys.path.insert(0, str(dockrun_path.parent))
    import dockrun

    # Build command with empty string
    cmd = dockrun.build_docker_run_command("")
    cmd_str = " ".join(cmd)

    # Should still have all fixed parameters
    assert "docker" in cmd_str, "Command should contain 'docker'"
    assert "lotf:latest" in cmd_str, "Command should contain image lotf:latest"
    assert "--gpus=all" in cmd_str, "Command should contain --gpus=all"
