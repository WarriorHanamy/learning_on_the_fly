"""Self-contained chirp signal generator for closed-loop excitation.

Generates single-channel log-sweep chirps with smooth windowing.  Segments
are sequential — at most one channel is excited at any time.
"""

from __future__ import annotations

import numpy as np

from .schema import ChirpSegment, N_ACTIONS, P_IDX, Q_IDX, R_IDX, THRUST_IDX

_CHANNEL_IDX = {
    "thrust": THRUST_IDX,
    "p": P_IDX,
    "q": Q_IDX,
    "r": R_IDX,
}


def chirp_signal(t_rel: float, cfg: ChirpSegment) -> float:
    """Evaluate a single log or linear chirp at a relative time within the segment.

    Parameters
    ----------
    t_rel : float
        Time elapsed since segment start [s].  Values outside [0, duration]
        are clamped to zero after windowing.
    cfg : ChirpSegment
        Segment parameters.

    Returns
    -------
    float
        Chirp sample value.
    """
    T = cfg.duration
    if T <= 0.0:
        return 0.0

    # clamp time
    t = max(0.0, min(float(t_rel), T))

    # log-sweep phase
    if cfg.f1_hz <= cfg.f0_hz:
        return 0.0

    r = cfg.f1_hz / cfg.f0_hz
    if cfg.kind == "log":
        phase = 2.0 * np.pi * cfg.f0_hz * T / np.log(r) * (np.exp(t / T * np.log(r)) - 1.0)
    else:  # linear sweep
        phase = 2.0 * np.pi * t * (cfg.f0_hz + 0.5 * (cfg.f1_hz - cfg.f0_hz) * t / T)

    val = cfg.amplitude * np.sin(phase)

    # Tukey (cosine-taper) window
    w = cfg.window_s
    if w > 0.0 and w < T / 2.0:
        if t < w:
            val *= 0.5 * (1.0 - np.cos(np.pi * t / w))
        elif t > T - w:
            val *= 0.5 * (1.0 - np.cos(np.pi * (T - t) / w))

    # zero outside segment
    if t_rel < 0.0 or t_rel > T:
        val = 0.0

    return val


def chirp_vector(t: float, segments: list[ChirpSegment]) -> np.ndarray:
    """Compute the 4-channel chirp offset at the given experiment time.

    Only one segment can be active at time ``t`` (earliest match wins).

    Parameters
    ----------
    t : float
        Current experiment time [s].
    segments : list[ChirpSegment]
        Ordered chirp segments.

    Returns
    -------
    np.ndarray
        Shape ``(4,)``, channel order [thrust, p, q, r].
    """
    offset = np.zeros(N_ACTIONS, dtype=np.float64)
    for seg in segments:
        if seg.t_start <= t < seg.t_start + seg.duration:
            idx = _CHANNEL_IDX[seg.channel]
            offset[idx] = chirp_signal(t - seg.t_start, seg)
            break  # earliest active segment wins
    return offset


def segment_id(t: float, segments: list[ChirpSegment]) -> int:
    """Return the 1-based active segment index, or 0 if no chirp is active."""
    for i, seg in enumerate(segments, start=1):
        if seg.t_start <= t < seg.t_start + seg.duration:
            return i
    return 0


# ---------------------------------------------------------------------------
# convenience: build the recommended default segment list
# ---------------------------------------------------------------------------


def default_chirp_segments(mass_kg: float = 0.192) -> list[ChirpSegment]:
    """Return the recommended sequential chirp segment list (p, q, r only)."""
    return [
        # --- p-rate chirp (5-35 s) ---
        ChirpSegment(
            channel="p",
            amplitude=0.01,  # rad/s
            f0_hz=0.2,
            f1_hz=4.0,
            t_start=5.0,
            duration=30.0,
            kind="log",
            window_s=2.0,
        ),
        # --- q-rate chirp (40-70 s) ---
        ChirpSegment(
            channel="q",
            amplitude=0.01,
            f0_hz=0.2,
            f1_hz=4.0,
            t_start=40.0,
            duration=30.0,
            kind="log",
            window_s=2.0,
        ),
        # --- r-rate chirp (75-105 s) ---
        ChirpSegment(
            channel="r",
            amplitude=0.01,
            f0_hz=0.2,
            f1_hz=4.0,
            t_start=75.0,
            duration=30.0,
            kind="log",
            window_s=2.0,
        ),
    ]
