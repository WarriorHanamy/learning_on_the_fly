"""Command-line entrypoint for LOTF micro-audits."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from lotf.audit.approx_channel import run_approx_channel_audit
from lotf.audit.schema import (
    AUDIT_CHANNELS,
    AUDIT_OUTPUT_DIR,
    DEFAULT_APPROX_PATH,
    ApproxChannelArtifactPaths,
    ApproxChannelAuditConfig,
    ApproxChannelEnvironment,
    ApproxChannelExcitation,
    ApproxChannelOutputForm,
)


def print_recipes() -> None:
    """Print concise audit recipes for users who do not know sub-test names."""
    print(
        f"""LOTF audit recipes

Available sub-tests:
  approx-channel    Check fitted inner-loop approximation channel by channel.

Recipes:
  uv run audit approx-channel
  uv run audit approx-channel --channels p,q,r
  uv run audit approx-channel --approx-path path/to/inner_loop_approx.json

Default artifacts:
  {AUDIT_OUTPUT_DIR}/

Use `uv run audit approx-channel --help` for flags.
"""
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit",
        description="LOTF micro-audits for simulator submodules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run audit
  uv run audit approx-channel
""",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="SUB-TEST")

    approx = subparsers.add_parser(
        "approx-channel",
        help="Check fitted inner-loop approximation channel by channel",
        description="Generate input/response artifacts for each delayed first-order channel model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run audit approx-channel
  uv run audit approx-channel --channels p,q,r
  uv run audit approx-channel --approx-path path/to/inner_loop_approx.json
""",
    )
    approx.add_argument(
        "--approx-path",
        default=DEFAULT_APPROX_PATH,
        help=f"Path to inner_loop_approx.json (default: {DEFAULT_APPROX_PATH})",
    )
    approx.add_argument(
        "--channels",
        default=",".join(AUDIT_CHANNELS),
        help="Comma-separated channels to audit: thrust,p,q,r (default: all)",
    )
    approx.add_argument("--dt", type=float, default=0.02, help="Sample period [s] (default: 0.02)")
    approx.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Audit duration [s] (default: 5.0)",
    )
    approx.add_argument(
        "--kind",
        choices=["log_chirp", "linear_chirp", "sine", "step"],
        default="step",
        help="Excitation type (default: step)",
    )
    approx.add_argument(
        "--amplitude",
        type=float,
        default=1.0,
        help="Excitation amplitude in channel native unit (default: 1.0)",
    )
    approx.add_argument("--f0", type=float, default=0.2, help="Start frequency [Hz]")
    approx.add_argument("--f1", type=float, default=5.0, help="End frequency [Hz]")
    approx.add_argument(
        "--step-time",
        type=float,
        default=1.0,
        help="Step onset time [s] for --kind step",
    )
    approx.add_argument(
        "--window",
        type=float,
        default=1.0,
        help="Chirp taper half-window [s] (default: 1.0)",
    )
    approx.add_argument(
        "--output-dir",
        default=AUDIT_OUTPUT_DIR,
        help=f"Artifact output directory (default: {AUDIT_OUTPUT_DIR})",
    )
    approx.add_argument(
        "--figure-format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Figure format (default: png)",
    )
    approx.add_argument("--no-figure", action="store_true", help="Do not write figure artifact")
    approx.add_argument(
        "--no-timeseries",
        action="store_true",
        help="Do not write timeseries NPZ artifact",
    )
    approx.add_argument("--no-summary", action="store_true", help="Do not write summary JSON")
    approx.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figure interactively",
    )

    return parser


def _parse_channels(value: str) -> tuple[str, ...]:
    channels = tuple(ch.strip() for ch in value.split(",") if ch.strip())
    invalid = [ch for ch in channels if ch not in AUDIT_CHANNELS]
    if invalid:
        allowed = ",".join(AUDIT_CHANNELS)
        raise ValueError(f"unknown channel(s): {','.join(invalid)}. Allowed: {allowed}")
    if not channels:
        raise ValueError("at least one channel is required")
    return channels


def _run_approx_channel(args: argparse.Namespace) -> int:
    try:
        channels = _parse_channels(args.channels)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    figure = f"approx_channel_response.{args.figure_format}"
    config = ApproxChannelAuditConfig(
        environment=ApproxChannelEnvironment(
            approx_path=args.approx_path,
            dt=args.dt,
            duration_s=args.duration,
        ),
        excitation=ApproxChannelExcitation(
            channels=channels,
            kind=args.kind,
            amplitude=args.amplitude,
            f0_hz=args.f0,
            f1_hz=args.f1,
            step_time_s=args.step_time,
            window_s=args.window,
        ),
        output=ApproxChannelOutputForm(
            save_figure=not args.no_figure,
            save_timeseries=not args.no_timeseries,
            save_summary=not args.no_summary,
            figure_format=args.figure_format,
            show=not args.no_show,
        ),
        artifacts=ApproxChannelArtifactPaths(output_dir=args.output_dir, figure=figure),
    )

    try:
        written = run_approx_channel_audit(config)
    except FileNotFoundError as e:
        print(f"Error: file not found: {e.filename}", file=sys.stderr)
        print("Hint: pass --approx-path path/to/inner_loop_approx.json", file=sys.stderr)
        return 1
    except KeyError as e:
        print(f"Error: missing expected channel or JSON key: {e}", file=sys.stderr)
        print("Hint: use inner_loop_approx.json produced by chirp analysis.", file=sys.stderr)
        return 1

    print("Approx-channel audit complete.")
    print(f"Detailed report directory: {config.artifacts.output_dir}")
    print("Artifacts:")
    for name, path in sorted(written.items()):
        print(f"  {name}: {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        print_recipes()
        return 0

    if args.command == "approx-channel":
        return _run_approx_channel(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
