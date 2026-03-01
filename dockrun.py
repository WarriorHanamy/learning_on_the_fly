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
import shlex
import subprocess
import sys
from pathlib import Path

__version__ = "1.0.0"


def build_docker_run_command(command_args: list[str]) -> list[str]:
    """Build docker run command with fixed parameters.

    Args:
        command_args: List of command-line arguments to run inside the container

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

    # Append user command arguments directly if provided
    if command_args:
        base_cmd.extend(command_args)

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
    # Check for help flag manually before argparse processes it
    # This allows nested commands to use --help
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        # Only show dockrun help if it's the only flag (besides --non-interactive)
        filtered_argv = [
            a
            for a in sys.argv[1:]
            if a not in ("--non-interactive", "--version", "-v", "--help", "-h")
        ]
        if not filtered_argv:
            parser = argparse.ArgumentParser(
                description="Run commands in LOTF Docker container with fixed parameters"
            )
            parser.add_argument(
                "--non-interactive",
                action="store_true",
                help="Run in non-interactive mode (default behavior)",
            )
            parser.add_argument(
                "--version", "-v", action="version", version=f"dockrun {__version__}"
            )
            parser.print_help()
            return 0

    parser = argparse.ArgumentParser(
        description="Run commands in LOTF Docker container with fixed parameters",
        add_help=False,  # Disable automatic -h/--help to allow nested commands to use them
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (default behavior)",
    )
    parser.add_argument(
        "--dockrun-version",
        action="store_true",
        help="Show dockrun version",
    )

    # Use parse_known_args to stop parsing after our flags
    # This allows nested commands with their own flags to work
    args, remaining = parser.parse_known_args()

    # Handle --dockrun-version
    if args.dockrun_version:
        print(f"dockrun {__version__}")
        return 0

    # If no command is provided, show help
    if not remaining:
        parser.print_help()
        return 2

    # Build docker run command with remaining arguments
    docker_cmd = build_docker_run_command(remaining)

    # Execute the command
    returncode = execute(docker_cmd)

    return returncode


if __name__ == "__main__":
    sys.exit(main())
