#!/usr/bin/env python3
"""
dockman.py - Simple Docker containerization CLI tool
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_NAME = Path(__file__).name
SCRIPT_VERSION = "1.0.0"


OPENCODE_CONFIG_PATHS = {
    ".config/opencode": "ro",
    ".local/share/opencode": "rw",
    ".cache/opencode": "rw",
}


class DockmanError(Exception):
    """Base exception for dockman errors."""

    pass


def execute(cmd: list[str], dry_run: bool = False) -> subprocess.CompletedProcess:
    """Execute a command, optionally in dry-run mode."""
    if dry_run:
        print(shlex.join(cmd))
        return subprocess.CompletedProcess(cmd, 0)
    return subprocess.run(cmd, check=False)


def execute_check(cmd: list[str], dry_run: bool = False) -> subprocess.CompletedProcess:
    """Execute a command with check=True, optionally in dry-run mode."""
    if dry_run:
        print(shlex.join(cmd))
        return subprocess.CompletedProcess(cmd, 0)
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def fix_proxy_for_docker(proxy: str | None) -> str:
    """Replace 127.0.0.1 with host.docker.internal for Docker bridge.
    Returns empty string if proxy is None or empty."""
    if proxy:
        return proxy.replace("127.0.0.1", "host.docker.internal")
    return ""


def get_current_dir_name() -> str:
    """Get current directory name for image naming."""
    return Path.cwd().name


def get_container_status(container_name: str) -> str:
    """Check container status.
    Returns: 'nonexistent', 'running', or 'stopped'
    """
    # Check running containers
    result = subprocess.run(
        [
            "sudo",
            "-g",
            "docker",
            "docker",
            "ps",
            "--format",
            "json",
            "--filter",
            f"name={container_name}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return "running"

    # Check all containers (including stopped)
    result = subprocess.run(
        [
            "sudo",
            "-g",
            "docker",
            "docker",
            "ps",
            "-a",
            "--format",
            "json",
            "--filter",
            f"name={container_name}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return "stopped"

    return "nonexistent"


def build_docker_run_command(args, current_dir, image_name, container_name=None):
    """Build docker run command with appropriate options."""
    # Calculate no_proxy_val before building the command list
    no_proxy_val = os.environ.get("no_proxy", "")
    if no_proxy_val:
        no_proxy_val = f"{no_proxy_val},host.docker.internal"
    else:
        no_proxy_val = "host.docker.internal"

    docker_cmd = [
        "sudo",
        "-g",
        "docker",
        "docker",
        "run",
        "--add-host=host.docker.internal:host-gateway",
        "--gpus=all",
        "-e",
        f"PATH=/root/.local/bin:$PATH",
        "-e",
        f"Z_AI_API_KEY={os.environ.get('Z_AI_API_KEY', '')}",
        "-e",
        f"DEEPSEEK_API_KEY={os.environ.get('DEEPSEEK_API_KEY', '')}",
        "-e",
        f"CONTEXT7_API_KEY={os.environ.get('CONTEXT7_API_KEY', '')}",
        "-e",
        f"TERM={os.environ.get('TERM', 'xterm')}",
        "-e",
        f"COLORTERM={os.environ.get('COLORTERM', '')}",
        "-e",
        f"http_proxy={fix_proxy_for_docker(os.environ.get('http_proxy', ''))}",
        "-e",
        f"https_proxy={fix_proxy_for_docker(os.environ.get('https_proxy', ''))}",
        "-e",
        f"all_proxy={fix_proxy_for_docker(os.environ.get('all_proxy', ''))}",
        "-e",
        f"no_proxy={no_proxy_val}",
    ]

    # Set container name if provided (for reuse)
    if container_name:
        docker_cmd.extend(["--name", container_name])

    # Set working directory
    workdir = args.workdir if args.workdir else "/app"
    docker_cmd.extend(["-w", workdir])

    # Mount current directory
    docker_cmd.extend(["-v", f"{current_dir}:{workdir}"])

    # Mount additional volumes
    for mount in args.mount or []:
        docker_cmd.extend(["-v", mount])

    # Mount git config if exists
    gitconfig_path = Path.home() / ".gitconfig"
    if gitconfig_path.exists():
        print(f"Using host git config: {gitconfig_path}", file=sys.stderr)
        docker_cmd.extend(["-v", f"{gitconfig_path}:/home/ubuntu/.gitconfig:ro"])

    # Mount mistake notebook if exists
    mistake_path = Path.home() / "MISTAKE.md"
    if mistake_path.exists():
        print(f"Using host mistake notebook: {mistake_path}", file=sys.stderr)
        docker_cmd.extend(["-v", f"{mistake_path}:/home/ubuntu/MISTAKE.md"])

    # Mount opencode config if exists
    for mount_path, mount_type in OPENCODE_CONFIG_PATHS.items():
        if (host_path := Path.home() / mount_path).exists():
            print(f"Mounting host opencode path: {host_path}", file=sys.stderr)
            docker_cmd.extend(["-v", f"{host_path}:/home/ubuntu/{mount_path}:{mount_type}"])

    # User mapping
    docker_cmd.extend(["-u", f"{os.getuid()}:{os.getgid()}"])

    # Interactive or not
    if not args.no_interactive:
        docker_cmd.extend(["--init", "-it"])
        # Only add --rm if not reusing container
        if not container_name:
            docker_cmd.extend(["--rm"])
    else:
        # Only add --rm if not reusing container
        if not container_name:
            docker_cmd.extend(["--rm"])

    docker_cmd.append(image_name)

    # Command to run
    if args.cmd:
        docker_cmd.extend(args.cmd)

    return docker_cmd


def build_docker_exec_command(args, container_name):
    """Build docker exec command."""
    docker_cmd = ["sudo", "-g", "docker", "docker", "exec"]

    # Interactive or not
    if not args.no_interactive:
        docker_cmd.extend(["-it"])

    docker_cmd.append(container_name)

    # Command to run
    if args.cmd:
        docker_cmd.extend(args.cmd)
    else:
        # Default shell
        docker_cmd.append("bash")

    return docker_cmd


def cmd_run(args: argparse.Namespace) -> int:
    """Run docker container."""
    dir_name = get_current_dir_name()
    current_dir = Path.cwd()
    image_name = f"docker-worker-{dir_name.lower()}:latest"

    # Check if image exists
    result = subprocess.run(
        ["sudo", "-g", "docker", "docker", "images", "--format", "json"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"✗ Failed to list Docker images: {result.stderr}", file=sys.stderr)
        return 1

    image_exists = False
    for line in result.stdout.splitlines():
        try:
            data = json.loads(line)
            if data.get("Repository") == f"docker-worker-{dir_name.lower()}":
                image_exists = True
                break
        except json.JSONDecodeError:
            pass

    if not image_exists:
        print(f"✗ Image not found: {image_name}", file=sys.stderr)
        return 1

    # Reuse container logic
    if args.reuse:
        container_name = f"dockman-{dir_name.lower()}"
        status = get_container_status(container_name)

        if status == "nonexistent":
            # Create new container with --name and no --rm
            docker_cmd = build_docker_run_command(args, current_dir, image_name, container_name)
            result = execute(docker_cmd, dry_run=args.dry_run)
            return result.returncode

        elif status == "stopped":
            # Start stopped container
            if args.cmd:
                # Start container first
                start_cmd = ["sudo", "-g", "docker", "docker", "start", container_name]
                start_result = execute(start_cmd, dry_run=args.dry_run)
                if start_result.returncode != 0:
                    return start_result.returncode
                # Then exec command
                exec_cmd = build_docker_exec_command(args, container_name)
                result = execute(exec_cmd, dry_run=args.dry_run)
                return result.returncode
            else:
                # Start and attach
                if args.no_interactive:
                    start_cmd = [
                        "sudo",
                        "-g",
                        "docker",
                        "docker",
                        "start",
                        container_name,
                    ]
                else:
                    start_cmd = [
                        "sudo",
                        "-g",
                        "docker",
                        "docker",
                        "start",
                        "-ai",
                        container_name,
                    ]
                result = execute(start_cmd, dry_run=args.dry_run)
                return result.returncode

        elif status == "running":
            # Container already running
            if args.cmd:
                exec_cmd = build_docker_exec_command(args, container_name)
                result = execute(exec_cmd, dry_run=args.dry_run)
                return result.returncode
            else:
                # Attach new shell
                if args.no_interactive:
                    print(
                        f"Container {container_name} is already running",
                        file=sys.stderr,
                    )
                    return 0
                else:
                    exec_cmd = [
                        "sudo",
                        "-g",
                        "docker",
                        "docker",
                        "exec",
                        "-it",
                        container_name,
                        "bash",
                    ]
                    result = execute(exec_cmd, dry_run=args.dry_run)
                    return result.returncode

        else:
            print(f"✗ Unknown container status: {status}", file=sys.stderr)
            return 1

    # Original behavior (no --reuse)
    docker_cmd = build_docker_run_command(args, current_dir, image_name)
    result = execute(docker_cmd, dry_run=args.dry_run)
    return result.returncode


def cmd_rm(args: argparse.Namespace) -> int:
    """Remove reuse container."""
    dir_name = get_current_dir_name()
    container_name = f"dockman-{dir_name.lower()}"
    status = get_container_status(container_name)

    if status == "nonexistent":
        print(f"✗ Container not found: {container_name}", file=sys.stderr)
        return 1

    if status == "running" and not args.force:
        print(f"✗ Container {container_name} is running", file=sys.stderr)
        print(f"  Use --force to stop and remove", file=sys.stderr)
        return 1

    cmd = ["sudo", "-g", "docker", "docker", "rm"]
    if args.force:
        cmd.append("-f")
    cmd.append(container_name)

    result = execute(cmd, dry_run=args.dry_run)
    if result.returncode == 0 and not args.dry_run:
        print(f"Removed container: {container_name}", file=sys.stderr)
    return result.returncode


def cmd_version(args: argparse.Namespace) -> int:
    """Show version."""
    print(f"{SCRIPT_NAME} version {SCRIPT_VERSION}")
    return 0


def set_terminal_title(title: str) -> None:
    """Set terminal title using ANSI escape sequence if TTY is detected."""
    if sys.stderr.isatty():
        print(f"\033]0;{title}\007", end="", file=sys.stderr, flush=True)


def main() -> int:
    # set_terminal_title("dockman")
    parser = argparse.ArgumentParser(
        prog=SCRIPT_NAME, description="Simple Docker containerization CLI tool"
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Show commands without executing them",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run docker container")
    run_parser.add_argument(
        "-m", "--mount", action="append", help="Mount additional volume (e.g., src:dst)"
    )
    run_parser.add_argument("-w", "--workdir", help="Working directory inside container")
    run_parser.add_argument(
        "-c",
        "--cmd",
        nargs=argparse.REMAINDER,
        help="Command to run (default: /bin/bash)",
    )
    run_parser.add_argument(
        "--no-interactive", action="store_true", help="Run in non-interactive mode"
    )
    run_parser.add_argument(
        "-r",
        "--reuse",
        action="store_true",
        help="Reuse existing container instead of creating new one",
    )

    # rm command
    rm_parser = subparsers.add_parser("rm", help="Remove reuse container")
    rm_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force remove running container",
    )

    # version command
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "rm":
        return cmd_rm(args)
    elif args.command == "version":
        return cmd_version(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
