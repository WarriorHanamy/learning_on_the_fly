#!/usr/bin/env python3
"""dockrun.py - Run commands in LOTF Docker container with fixed parameters.

Usage:
    dockrun --non-interactive [command]
    dockrun --version

Fixed parameters:
    - Image: lotf:latest
    - GPU: --gpus=all
    - Volume: -v $(pwd):/app
    - Work directory: -w /app
    - Auto-remove: --rm
"""

import argparse
import subprocess
import sys
from pathlib import Path

__version__ = "1.0.0"


def build_docker_run_command(command: str) -> list[str]:
    """Build docker run command with fixed parameters.

    Args:
        command: Command to run inside the container

    Returns:
        List of command-line arguments for docker run
    """
    # Fixed parameters as per requirements
    base_cmd = [
        "docker",
        "run",
        "--gpus=all",
        "-v",
        f"{Path.cwd()}:/app",
        "-w",
        "/app",
        "--rm",
        "lotf:latest",
    ]

    # Append user command if provided
    if command:
        # Split command by spaces to handle multi-word commands
        # For complex commands, users should quote properly
        base_cmd.extend(command.split())

    return base_cmd


def execute(command: list[str]) -> int:
    """Execute a command and return its exit code.

    Args:
        command: List of command-line arguments

    Returns:
        Exit code from the executed command
    """
    result = subprocess.run(command)
    return result.returncode


def main() -> int:
    """Main entry point for dockrun.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(
        description="Run commands in LOTF Docker container with fixed parameters"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (default behavior)",
    )
    parser.add_argument("--version", "-v", action="version", version=f"dockrun {__version__}")
    parser.add_argument("command", nargs="*", help="Command to run inside the container")

    args = parser.parse_args()

    # If no command is provided, show help
    if not args.command:
        parser.print_help()
        return 2

    # Build docker run command
    cmd_str = " ".join(args.command) if args.command else ""
    docker_cmd = build_docker_run_command(cmd_str)

    # Execute the command
    returncode = execute(docker_cmd)

    return returncode


if __name__ == "__main__":
    sys.exit(main())
