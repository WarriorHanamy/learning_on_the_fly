"""Tests for version command."""

import subprocess
import sys
from pathlib import Path


def test_version_long_form():
    """Test that dockrun --version displays version."""
    dockrun_path = Path(__file__).parent.parent.parent / "dockrun.py"

    if not dockrun_path.exists():
        assert False, "dockrun.py does not exist yet"

    result = subprocess.run(
        [sys.executable, str(dockrun_path), "--version"],
        capture_output=True,
        text=True,
    )

    # Should exit with code 0
    assert result.returncode == 0, "dockrun --version should exit with code 0"
    # Should output something (version number)
    assert len(result.stdout) > 0, "dockrun --version should output version information"


def test_version_short_form():
    """Test that dockrun -v displays version."""
    dockrun_path = Path(__file__).parent.parent.parent / "dockrun.py"

    if not dockrun_path.exists():
        assert False, "dockrun.py does not exist yet"

    result = subprocess.run(
        [sys.executable, str(dockrun_path), "-v"],
        capture_output=True,
        text=True,
    )

    # Should exit with code 0
    assert result.returncode == 0, "dockrun -v should exit with code 0"
    # Should output something (version number)
    assert len(result.stdout) > 0, "dockrun -v should output version information"
