#!/usr/bin/env python3
"""Tests for init command scenario."""

import os
import tempfile
from pathlib import Path

import pytest

EXPECTED_DOCKERFILE_CONTENT = """# Use the already-built lotf image as base
FROM lotf:latest

# Set working directory
WORKDIR /app

# Set PATH to include uv
ENV PATH="/root/.local/bin:$PATH"

# Default command
CMD ["uv", "run", "python", "-m", "lotf", "--help"]
"""


class TestInitCommand:
    """Test init command functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            # Create dockman subdirectory
            dockman_dir = project_dir / ".dockman"
            dockman_dir.mkdir()
            yield project_dir

    def test_init_creates_dockerfile(self, temp_project_dir):
        """Test that init creates Dockerfile at .dockman/Dockerfile."""
        from dockerfile_utils import cmd_init

        args = type("Args", (), {"force": False})()
        original_cwd = Path.cwd()
        os.chdir(temp_project_dir)

        try:
            result = cmd_init(args)
            assert result == 0, "Init command should return 0 on success"
            dockerfile_path = temp_project_dir / ".dockman" / "Dockerfile"
            assert dockerfile_path.exists(), "Dockerfile should be created at .dockman/Dockerfile"
        finally:
            os.chdir(original_cwd)

    def test_init_creates_correct_content(self, temp_project_dir):
        """Test that init creates Dockerfile with correct content."""
        from dockerfile_utils import cmd_init

        args = type("Args", (), {"force": False})()
        original_cwd = Path.cwd()
        os.chdir(temp_project_dir)

        try:
            result = cmd_init(args)
            assert result == 0
            dockerfile_path = temp_project_dir / ".dockman" / "Dockerfile"
            content = dockerfile_path.read_text()
            assert content == EXPECTED_DOCKERFILE_CONTENT, (
                "Dockerfile content should match expected"
            )
        finally:
            os.chdir(original_cwd)

    def test_init_force_overwrites_existing(self, temp_project_dir):
        """Test that init --force overwrites existing Dockerfile."""
        from dockerfile_utils import cmd_init

        dockerfile_path = temp_project_dir / ".dockman" / "Dockerfile"
        dockerfile_path.write_text("OLD CONTENT")

        args = type("Args", (), {"force": True})()
        original_cwd = Path.cwd()
        os.chdir(temp_project_dir)

        try:
            result = cmd_init(args)
            assert result == 0
            content = dockerfile_path.read_text()
            assert content == EXPECTED_DOCKERFILE_CONTENT, "Dockerfile should be overwritten"
        finally:
            os.chdir(original_cwd)

    def test_init_fails_without_force_if_exists(self, temp_project_dir):
        """Test that init without --force fails if Dockerfile exists."""
        from dockerfile_utils import cmd_init

        dockerfile_path = temp_project_dir / ".dockman" / "Dockerfile"
        dockerfile_path.write_text("EXISTING CONTENT")

        args = type("Args", (), {"force": False})()
        original_cwd = Path.cwd()
        os.chdir(temp_project_dir)

        try:
            result = cmd_init(args)
            assert result == 1, "Init should return 1 when Dockerfile exists without --force"
            content = dockerfile_path.read_text()
            assert content == "EXISTING CONTENT", "Dockerfile should not be overwritten"
        finally:
            os.chdir(original_cwd)
