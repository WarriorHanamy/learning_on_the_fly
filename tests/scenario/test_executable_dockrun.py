"""Tests for dockrun.py executable requirement."""

import os
import stat
import subprocess
from pathlib import Path


def test_dockrun_file_exists():
    """Test that dockrun.py file exists in project root."""
    dockrun_path = Path(__file__).parent.parent.parent / "dockrun.py"
    assert dockrun_path.exists(), "dockrun.py should exist in project root"


def test_dockrun_is_executable():
    """Test that dockrun.py has executable permissions."""
    dockrun_path = Path(__file__).parent.parent.parent / "dockrun.py"
    if dockrun_path.exists():
        file_stat = os.stat(dockrun_path)
        executable = bool(file_stat.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        assert executable, "dockrun.py should have executable permission (+x)"
    else:
        # File doesn't exist yet, so it can't be executable
        assert False, "dockrun.py does not exist yet, cannot verify executable permission"
