#!/usr/bin/env python3
"""
dockerfile_utils.py - Dockerfile management and image building tool for LOTF project.
Standalone utility separate from dockman.py.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_NAME = Path(__file__).name
SCRIPT_VERSION = "1.0.0"

# Dockerfile content for .dockman/Dockerfile
DOCKERFILE_CONTENT = """# Use the already-built lotf image as base
FROM lotf:latest

# Set working directory
WORKDIR /app

# Set PATH to include uv
ENV PATH="/root/.local/bin:$PATH"

# Default command
CMD ["uv", "run", "python", "-m", "lotf", "--help"]
"""


def execute(cmd: list[str]) -> subprocess.CompletedProcess:
    """
    Execute a command and return the result.

    Args:
        cmd: Command to execute as a list of strings.

    Returns:
        subprocess.CompletedProcess: Result of command execution.
    """
    return subprocess.run(cmd, check=False)


def cmd_init(args: argparse.Namespace) -> int:
    """
    Create Dockerfile in .dockman directory.

    Args:
        args: argparse.Namespace with force attribute to control overwrite behavior.

    Returns:
        int: 0 on success, 1 on failure.
    """
    current_dir = Path.cwd()
    target_dockerfile = current_dir / ".dockman" / "Dockerfile"

    if target_dockerfile.exists():
        if not args.force:
            print(f"✗ Dockerfile already exists in {target_dockerfile}", file=sys.stderr)
            print(f"  Use --force to overwrite", file=sys.stderr)
            return 1
        print(f"Overwriting existing Dockerfile in {target_dockerfile}", file=sys.stderr)

    print(f"Creating Dockerfile in {target_dockerfile}", file=sys.stderr)
    target_dockerfile.parent.mkdir(parents=True, exist_ok=True)
    target_dockerfile.write_text(DOCKERFILE_CONTENT)

    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """
    Build Docker image tagged as lotf:latest.

    Args:
        args: argparse.Namespace (unused, kept for consistency).

    Returns:
        int: 0 on success, non-zero on failure from docker build command.
    """
    current_dir = Path.cwd()
    dockerfile_path = current_dir / "Dockerfile"

    if not dockerfile_path.exists():
        print(f"✗ Dockerfile not found at {dockerfile_path}", file=sys.stderr)
        print(f"  Dockerfile must exist in project root", file=sys.stderr)
        return 1

    cmd = ["docker", "build", "-t", "lotf:latest", str(current_dir)]

    result = execute(cmd)
    return result.returncode


def cmd_version(args: argparse.Namespace) -> int:
    """
    Show tool version.

    Args:
        args: argparse.Namespace (unused, kept for consistency).

    Returns:
        int: Always returns 0.
    """
    print(f"{SCRIPT_NAME} version {SCRIPT_VERSION}")
    return 0


def main() -> int:
    """
    Main entry point for dockerfile_utils CLI.

    Returns:
        int: 0 on success, 1 on failure or when help is displayed.
    """
    parser = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Dockerfile management and image building tool for LOTF project",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Create Dockerfile in .dockman directory")
    init_parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite existing Dockerfile"
    )

    # build command
    subparsers.add_parser("build", help="Build Docker image tagged as lotf:latest")

    # version command
    subparsers.add_parser("version", help="Show tool version")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "init":
        return cmd_init(args)
    elif args.command == "build":
        return cmd_build(args)
    elif args.command == "version":
        return cmd_version(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
