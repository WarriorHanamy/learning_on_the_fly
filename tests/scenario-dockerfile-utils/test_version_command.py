#!/usr/bin/env python3
"""Tests for version command scenario."""

from unittest.mock import patch

import pytest


class TestVersionCommand:
    """Test version command functionality."""

    @patch("sys.stdout")
    def test_version_command_displays_version(self, mock_stdout):
        """Test that version command displays version in correct format."""
        from dockerfile_utils import cmd_version

        args = type("Args", (), {})()
        result = cmd_version(args)

        assert result == 0, "Version command should return 0"
        assert mock_stdout.write.called, "Version should be written to stdout"

        # Check version format
        written_output = "".join([call[0][0] for call in mock_stdout.write.call_args_list])
        assert "dockerfile_utils.py" in written_output, "Tool name should be displayed"
        assert "version" in written_output, "Version keyword should be displayed"

    @patch("sys.stdout")
    def test_version_format(self, mock_stdout):
        """Test that version string follows correct pattern."""
        from dockerfile_utils import cmd_version

        args = type("Args", (), {})()
        result = cmd_version(args)

        written_output = "".join([call[0][0] for call in mock_stdout.write.call_args_list])
        # Should match pattern like "dockerfile_utils.py version 1.0.0"
        import re

        pattern = r"dockerfile_utils\.py version \d+\.\d+\.\d+"
        assert re.search(pattern, written_output), (
            f"Version should match pattern, got: {written_output}"
        )
