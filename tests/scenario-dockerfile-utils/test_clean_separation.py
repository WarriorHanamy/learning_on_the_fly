#!/usr/bin/env python3
"""Tests for clean separation scenario."""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestCleanSeparation:
    """Test that dockerfile_utils.py is cleanly separated from dockman.py."""

    def test_dockerfile_utils_no_imports_from_dockman(self):
        """Test that dockerfile_utils.py does not import from dockman."""
        dockerfile_utils_path = PROJECT_ROOT / "dockerfile_utils.py"
        with open(dockerfile_utils_path) as f:
            content = f.read()
        tree = ast.parse(content)

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "dockman" in node.module:
                    imports.append(node.module)

        assert len(imports) == 0, (
            f"dockerfile_utils.py should not import from dockman, found: {imports}"
        )

    def test_dockerfile_utils_standalone(self):
        """Test that dockerfile_utils.py can run independently."""
        from dockerfile_utils import main, SCRIPT_NAME, SCRIPT_VERSION

        # Just check that the module can be imported and has expected attributes
        assert SCRIPT_NAME == "dockerfile_utils.py", "SCRIPT_NAME should be dockerfile_utils.py"
        assert isinstance(SCRIPT_VERSION, str), "SCRIPT_VERSION should be a string"
        assert callable(main), "main function should be callable"
