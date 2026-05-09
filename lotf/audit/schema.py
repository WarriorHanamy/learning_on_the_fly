"""Micro-audit schemas for approximated inner-loop channel models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from lotf.forward_model_config import ForwardModelConfig

AUDIT_OUTPUT_DIR = "lotf/audit/_output"
DEFAULT_APPROX_PATH = "lotf/audit/default_inner_loop_approx.json"
AUDIT_CHANNELS = ("thrust", "p", "q", "r")
AuditChannel = Literal["thrust", "p", "q", "r"]


@dataclass(frozen=True)
class ApproxChannelEnvironment:
    """Environment for checking one fitted delayed first-order channel model.

    Attributes:
        approx_path: Path to ``inner_loop_approx.json`` from chirp analysis.
        dt: Discrete sample period [s].
        duration_s: Simulation duration [s].
    """

    approx_path: str = DEFAULT_APPROX_PATH
    dt: float = 0.02
    duration_s: float = 5.0


@dataclass(frozen=True)
class ApproxChannelExcitation:
    """Excitation signal used to inspect each approximated channel.

    Attributes:
        channels: Channels to audit.
        kind: Excitation type.
        amplitude: Native-unit signal amplitude.
        f0_hz: Start frequency for chirps [Hz].
        f1_hz: End frequency for chirps [Hz].
        step_time_s: Step onset time for ``kind='step'`` [s].
        window_s: Cosine taper duration at each chirp boundary [s].
    """

    channels: tuple[AuditChannel, ...] = AUDIT_CHANNELS
    kind: Literal["log_chirp", "linear_chirp", "sine", "step"] = "step"
    amplitude: float = 1.0
    f0_hz: float = 0.2
    f1_hz: float = 5.0
    step_time_s: float = 1.0
    window_s: float = 1.0


@dataclass(frozen=True)
class ApproxChannelOutputForm:
    """Requested outputs for the micro-audit.

    Attributes:
        save_figure: Save per-channel input/response figure.
        save_timeseries: Save generated input and model response arrays.
        save_summary: Save fitted channel parameters and run metadata.
        figure_format: Figure file format.
        dpi: Figure resolution [dots/in].
        show: Display figure interactively.
    """

    save_figure: bool = True
    save_timeseries: bool = True
    save_summary: bool = True
    figure_format: Literal["png", "pdf", "svg"] = "png"
    dpi: int = 160
    show: bool = True


@dataclass(frozen=True)
class ApproxChannelArtifactPaths:
    """Output artifact paths for the channel-by-channel approximation audit.

    Attributes:
        output_dir: Directory for generated artifacts. Default is git-ignored.
        config_json: Serialized audit config path relative to ``output_dir``.
        timeseries_npz: Numeric channel responses relative to ``output_dir``.
        figure: Response figure path relative to ``output_dir``.
        summary_json: Summary path relative to ``output_dir``.
    """

    output_dir: str = AUDIT_OUTPUT_DIR
    config_json: str = "approx_channel_config.json"
    timeseries_npz: str = "approx_channel_timeseries.npz"
    figure: str = "approx_channel_response.png"
    summary_json: str = "approx_channel_summary.json"


@dataclass(frozen=True)
class ApproxChannelAuditConfig:
    """Top-level micro-audit config for fitted inner-loop channel models.

    Attributes:
        environment: Approximation file and time discretization.
        excitation: Per-channel excitation signal definition.
        output: Requested output forms, typically including a figure.
        artifacts: Paths where generated outputs are written.
    """

    environment: ApproxChannelEnvironment
    excitation: ApproxChannelExcitation = field(default_factory=ApproxChannelExcitation)
    output: ApproxChannelOutputForm = field(default_factory=ApproxChannelOutputForm)
    artifacts: ApproxChannelArtifactPaths = field(default_factory=ApproxChannelArtifactPaths)


@dataclass(frozen=True)
class QuadrotorStepInput:
    """Command input for one raw ``Quadrotor.step`` call.

    Attributes:
        thrust_N: Desired total thrust [N].
        omega_body_radps: Desired body rates in p/q/r order [rad/s].
    """

    thrust_N: float
    omega_body_radps: tuple[float, float, float]


@dataclass(frozen=True)
class QuadrotorStepAuditConfig:
    """Minimal raw-step loader config for direct ``Quadrotor.step`` inspection."""

    drone_name: str = "example_quad"
    dt: float = 0.02
    forward_model_config: ForwardModelConfig = field(default_factory=ForwardModelConfig)
    residual_checkpoint: str | None = None
    jit: bool = True
