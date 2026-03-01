#!/usr/bin/env python3
"""Tests for tool structure and executability scenario."""

import os
import stat
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestToolStructure:
    """Test dockerfile_utils.py file structure and executability."""

    def test_file_exists_in_project_root(self):
        """Test that dockerfile_utils.py exists in project root."""
        dockerfile_utils_path = PROJECT_ROOT / "dockerfile_utils.py"
        assert dockerfile_utils_path.exists(), "dockerfile_utils.py should exist in project root"

    def test_file_is_executable(self):
        """Test that dockerfile_utils.py has execute permission."""
        dockerfile_utils_path = PROJECT_ROOT / "dockerfile_utils.py"
        file_stat = dockerfile_utils_path.stat()
        is_executable = bool(file_stat.st_mode & stat.S_IXUSR)
        assert is_executable, "dockerfile_utils.py should have execute permission"

    def test_file_has_proper_shebang(self):
        """Test that dockerfile_utils.py starts with proper shebang line."""
        dockerfile_utils_path = PROJECT_ROOT / "dockerfile_utils.py"
        with open(dockerfile_utils_path) as f:
            first_line = f.readline().strip()
        assert first_line == "#!/usr/bin/env python3", "First line should be #!/usr/bin/env python3"
