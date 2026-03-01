"""Tests for --non-interactive flag parsing."""

import subprocess
import sys
from pathlib import Path


def test_parse_non_interactive_flag():
    """Test that dockrun recognizes --non-interactive flag."""
    dockrun_path = Path(__file__).parent.parent.parent / "dockrun.py"

    if not dockrun_path.exists():
        assert False, "dockrun.py does not exist yet"

    # Try to get help with --non-interactive flag
    result = subprocess.run(
        [sys.executable, str(dockrun_path), "--non-interactive", "--help"],
        capture_output=True,
        text=True,
    )

    # Should not crash, exit code 0 indicates successful flag parsing
    assert result.returncode == 0, "dockrun should accept --non-interactive flag"


def test_parse_without_non_interactive_flag():
    """Test that dockrun works without --non-interactive flag."""
    dockrun_path = Path(__file__).parent.parent.parent / "dockrun.py"

    if not dockrun_path.exists():
        assert False, "dockrun.py does not exist yet"

    # Try to get help without --non-interactive flag
    result = subprocess.run(
        [sys.executable, str(dockrun_path), "--help"],
        capture_output=True,
        text=True,
    )

    # Should not crash, exit code 0 indicates successful parsing
    assert result.returncode == 0, "dockrun should work without --non-interactive flag"
