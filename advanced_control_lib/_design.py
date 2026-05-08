"""Private helpers: Butterworth polynomials, continuous TF -> SS, Tustin discretization.

All functions are stateless.  Not part of the public API.
"""

from __future__ import annotations

import numpy as np


BUTTERWORTH_DENOMINATORS = {
    1: np.array([1.0, 1.0], dtype=np.float64),
    2: np.array([1.0, 1.414213562373095, 1.0], dtype=np.float64),
    3: np.array([1.0, 2.0, 2.0, 1.0], dtype=np.float64),
    4: np.array(
        [1.0, 2.613125929752753, 3.414213562373095, 2.613125929752753, 1.0], dtype=np.float64
    ),
    5: np.array(
        [1.0, 3.236067977499789, 5.236067977499789, 5.236067977499789, 3.236067977499789, 1.0],
        dtype=np.float64,
    ),
    6: np.array(
        [
            1.0,
            3.863703305156273,
            7.464101615137753,
            9.141620172685640,
            7.464101615137753,
            3.863703305156273,
            1.0,
        ],
        dtype=np.float64,
    ),
}


def butterworth_denominator(order: int) -> np.ndarray:
    """Return normalized Butterworth denominator polynomial coefficients.

    Coefficients are in descending powers of s:
        B_N(s) = s^N + a_{N-1} s^{N-1} + ... + a_1 s + 1

    Valid for orders 1–6.
    """
    if order not in BUTTERWORTH_DENOMINATORS:
        raise ValueError(f"Butterworth order must be 1–6, got {order}")
    return BUTTERWORTH_DENOMINATORS[order].copy()


def _continuous_tf_to_ss_1st(
    numerator: np.ndarray, denominator: np.ndarray
) -> tuple[float, float, float, float]:
    """TF -> controllable canonical SS for 1st-order system.

    H(s) = (b1 s + b0) / (a1 s + a0)
    """
    a0 = denominator[1] / denominator[0]
    b0 = numerator[1] / denominator[0]
    b1 = numerator[0] / denominator[0]
    h0 = b1
    h1 = b0 - a0 * h0
    Ac = -a0
    Bc = h1
    Cc = 1.0
    Dc = h0
    return Ac, Bc, Cc, Dc


def _continuous_tf_to_ss_nth(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """TF -> controllable canonical SS for N-th order system (N >= 2)."""
    N = len(denominator) - 1

    # Normalize and reverse coefficients
    a = np.array([denominator[N - i] / denominator[0] for i in range(N)], dtype=np.float64)
    b = np.array([numerator[N - i] / denominator[0] for i in range(N + 1)], dtype=np.float64)

    # Compute h coefficients
    h = np.zeros(N + 1, dtype=np.float64)
    h[0] = b[N]
    for k in range(1, N + 1):
        s = b[N - k]
        for j in range(1, k + 1):
            s -= a[N - j] * h[k - j]
        h[k] = s

    # Controllable canonical form
    Ac = np.zeros((N, N), dtype=np.float64)
    for i in range(N - 1):
        Ac[i, i + 1] = 1.0
    Ac[N - 1, :] = -a

    Bc = np.zeros((N, 1), dtype=np.float64)
    for i in range(N):
        Bc[i, 0] = h[i + 1]

    Cc = np.zeros((1, N), dtype=np.float64)
    Cc[0, 0] = 1.0

    Dc = h[0]

    return Ac, Bc, Cc, Dc


def continuous_tf_to_ss(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert continuous-time transfer function to controllable canonical state-space.

    H(s) = (num[0] s^N + ... + num[N]) / (den[0] s^N + ... + den[N])

    For 1st-order, returns scalar Ac, Bc, Cc, Dc.
    For N >= 2, returns ndarray Ac(N,N), Bc(N,1), Cc(1,N), Dc(1,1).
    """
    N = len(denominator) - 1
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)

    if len(num) != len(den):
        raise ValueError(
            f"numerator length ({len(num)}) must equal denominator length ({len(den)})"
        )

    if N == 1:
        return _continuous_tf_to_ss_1st(num, den)
    else:
        return _continuous_tf_to_ss_nth(num, den)


def tustin_discretize_1st(
    Ac: float, Bc: float, Cc: float, Dc: float, sample_period_s: float
) -> tuple[float, float, float, float]:
    """Tustin (bilinear) discretization for 1st-order system."""
    alpha = 2.0 / sample_period_s
    Ad = (alpha + Ac) / (alpha - Ac)
    Bd = Bc / (alpha - Ac)
    Cd = Cc * (Ad + 1.0)
    Dd = Cc * Bd + Dc
    return Ad, Bd, Cd, Dd


def tustin_discretize_nth(
    Ac: np.ndarray,
    Bc: np.ndarray,
    Cc: np.ndarray,
    Dc: np.ndarray,
    sample_period_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tustin (bilinear) discretization for N-th order system (N >= 2)."""
    N = Ac.shape[0]
    alpha = 2.0 / sample_period_s
    I = np.eye(N, dtype=np.float64)

    tmp = alpha * I - Ac
    tmp_inv = np.linalg.inv(tmp)

    Ad = tmp_inv @ (alpha * I + Ac)
    Bd = tmp_inv @ Bc
    Cd = Cc @ (Ad + I)
    Dd = Cc @ Bd + Dc

    return Ad, Bd, Cd, Dd


def tustin_discretize(
    Ac: np.ndarray,
    Bc: np.ndarray,
    Cc: np.ndarray,
    Dc: np.ndarray,
    sample_period_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tustin (bilinear transform) discretization.

    Maps continuous-time (Ac,Bc,Cc,Dc) to discrete-time (Ad,Bd,Cd,Dd)
    using the trapezoidal rule:
        s = (2/T) * (z-1)/(z+1)
    """
    if np.ndim(Ac) == 0 or (isinstance(Ac, float)):
        return tustin_discretize_1st(float(Ac), float(Bc), float(Cc), float(Dc), sample_period_s)
    else:
        return tustin_discretize_nth(Ac, Bc, Cc, Dc, sample_period_s)


def compute_cd_pinv(Cd: np.ndarray) -> np.ndarray:
    """Compute right pseudo-inverse of a row vector C (1xN).

    C_pinv = C^T / (C @ C^T)   (Moore-Penrose for full row rank)
    """
    C_flat = np.asarray(Cd).ravel()
    norm_sq = float(np.dot(C_flat, C_flat))
    if norm_sq < 1e-30:
        raise ValueError("Cd row vector has zero norm; cannot compute pseudo-inverse")
    return C_flat.reshape(-1, 1) / norm_sq
