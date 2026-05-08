"""Private SISO state-space classes for fixed orders 1–6.

Not part of the public API.  Public tools build the correct-order instance
internally via the mapping rule:

    denominator degree  ->  _StateSpaceSISO{degree}

Classes:
    _StateSpaceSISO1st  (scalar matrices, no ndarray)
    _StateSpaceSISO2nd  (2x2, 2x1, 1x2)
    _StateSpaceSISO3rd  (3x3, 3x1, 1x3)
    _StateSpaceSISO4th  (4x4, 4x1, 1x4)
    _StateSpaceSISO5th  (5x5, 5x1, 1x5)
    _StateSpaceSISO6th  (6x6, 6x1, 1x6)
"""

from __future__ import annotations

import numpy as np

from ._design import (
    compute_cd_pinv,
    continuous_tf_to_ss,
    tustin_discretize,
)


# ==============================================================================
# 1st-order (scalar)
# ==============================================================================


class _StateSpaceSISO1st:
    """First-order discrete-time SISO state-space system.

           y[k] = Cd * x[k] + Dd * u[k]
           x[k+1] = Ad * x[k] + Bd * u[k]

    All matrices are scalar floats.
    """

    def __init__(self) -> None:
        self._ad: float = 1.0
        self._bd: float = 0.0
        self._cd: float = 1.0
        self._dd: float = 0.0
        self._cd_pinv: float = 1.0
        self._x: float = 0.0
        self._y: float = 0.0
        self._u: float = 0.0

    # --- construction ---

    @classmethod
    def from_continuous_tf(
        cls,
        numerator: np.ndarray,
        denominator: np.ndarray,
        sample_period_s: float,
    ) -> _StateSpaceSISO1st:
        """Build from continuous-time transfer function coefficients."""
        Ac, Bc, Cc, Dc = continuous_tf_to_ss(numerator, denominator)
        Ad, Bd, Cd, Dd = tustin_discretize(Ac, Bc, Cc, Dc, sample_period_s)

        obj = cls()
        obj._ad = float(Ad)
        obj._bd = float(Bd)
        obj._cd = float(Cd)
        obj._dd = float(Dd)
        obj._cd_pinv = 1.0 / obj._cd if abs(obj._cd) > 1e-30 else 1.0
        obj._x = 0.0
        obj._y = 0.0
        obj._u = 0.0
        return obj

    # --- runtime ---

    def update(self, u: float) -> float:
        """Advance one discrete step: output = C x + D u, then x <- A x + B u."""
        self._u = u
        self._y = self._cd * self._x + self._dd * self._u
        self._x = self._ad * self._x + self._bd * self._u
        return self._y

    def reset(self, x0: float | None = None) -> None:
        """Clear state to zero or supplied x0."""
        self._x = 0.0 if x0 is None else float(x0)
        self._y = 0.0
        self._u = 0.0

    def switch(self, u: float, y: float) -> None:
        """Bumpless transfer: set state so output matches y given input u."""
        self._u = u
        self._y = y
        self._x = self._cd_pinv * (self._y - self._dd * self._u)
        self._x = self._ad * self._x + self._bd * self._u

    @property
    def y(self) -> float:
        """Last output value."""
        return self._y


# ==============================================================================
# Base for N >= 2
# ==============================================================================


class _StateSpaceSISONth:
    """Base for N-th order (N >= 2) discrete-time SISO state-space system.

    2nd, 3rd, 4th, 5th, 6th all share the same logic; only matrix dimensions differ.
    """

    def __init__(self, order: int) -> None:
        N = order
        self._ad = np.eye(N, dtype=np.float64)
        self._bd = np.zeros((N, 1), dtype=np.float64)
        self._cd = np.zeros((1, N), dtype=np.float64)
        self._cd[0, 0] = 1.0
        self._dd = np.array([[0.0]], dtype=np.float64)
        self._cd_pinv = np.zeros((N, 1), dtype=np.float64)
        self._cd_pinv[0, 0] = 1.0
        self._x = np.zeros((N, 1), dtype=np.float64)
        self._y = np.zeros((1, 1), dtype=np.float64)
        self._u = np.zeros((1, 1), dtype=np.float64)

    @classmethod
    def from_continuous_tf(
        cls,
        numerator: np.ndarray,
        denominator: np.ndarray,
        sample_period_s: float,
        order: int,
    ) -> _StateSpaceSISONth:
        """Build from continuous-time transfer function coefficients."""
        Ac, Bc, Cc, Dc_nd = continuous_tf_to_ss(numerator, denominator)
        Ad, Bd, Cd, Dd = tustin_discretize(Ac, Bc, Cc, Dc_nd, sample_period_s)

        obj = cls.__new__(cls)
        _StateSpaceSISONth.__init__(obj, order)
        obj._ad = Ad
        obj._bd = Bd
        obj._cd = Cd
        obj._dd = Dd
        obj._cd_pinv = compute_cd_pinv(Cd)
        obj._x = np.zeros((order, 1), dtype=np.float64)
        obj._y = np.zeros((1, 1), dtype=np.float64)
        obj._u = np.zeros((1, 1), dtype=np.float64)
        return obj

    def update(self, u: float) -> float:
        """Advance one discrete step."""
        self._u[0, 0] = u
        self._y = self._cd @ self._x + self._dd @ self._u
        self._x = self._ad @ self._x + self._bd @ self._u
        return float(self._y[0, 0])

    def reset(self, x0: np.ndarray | None = None) -> None:
        """Clear state to zero or supplied x0."""
        N = self._ad.shape[0]
        if x0 is not None:
            self._x = np.asarray(x0, dtype=np.float64).reshape(N, 1)
        else:
            self._x = np.zeros((N, 1), dtype=np.float64)
        self._y.fill(0.0)
        self._u.fill(0.0)

    def switch(self, u: float, y: float) -> None:
        """Bumpless transfer."""
        self._u[0, 0] = u
        self._y[0, 0] = y
        self._x = self._cd_pinv @ (self._y - self._dd @ self._u)
        self._x = self._ad @ self._x + self._bd @ self._u

    @property
    def y(self) -> float:
        """Last output value."""
        return float(self._y[0, 0])


# ==============================================================================
# Concrete order classes
# ==============================================================================


class _StateSpaceSISO2nd(_StateSpaceSISONth):
    """Second-order discrete-time SISO state-space system."""

    def __init__(self) -> None:
        super().__init__(order=2)

    @classmethod
    def from_continuous_tf(
        cls,
        numerator: np.ndarray,
        denominator: np.ndarray,
        sample_period_s: float,
    ) -> _StateSpaceSISO2nd:
        return super().from_continuous_tf(numerator, denominator, sample_period_s, order=2)


class _StateSpaceSISO3rd(_StateSpaceSISONth):
    """Third-order discrete-time SISO state-space system."""

    def __init__(self) -> None:
        super().__init__(order=3)

    @classmethod
    def from_continuous_tf(
        cls,
        numerator: np.ndarray,
        denominator: np.ndarray,
        sample_period_s: float,
    ) -> _StateSpaceSISO3rd:
        return super().from_continuous_tf(numerator, denominator, sample_period_s, order=3)


class _StateSpaceSISO4th(_StateSpaceSISONth):
    """Fourth-order discrete-time SISO state-space system."""

    def __init__(self) -> None:
        super().__init__(order=4)

    @classmethod
    def from_continuous_tf(
        cls,
        numerator: np.ndarray,
        denominator: np.ndarray,
        sample_period_s: float,
    ) -> _StateSpaceSISO4th:
        return super().from_continuous_tf(numerator, denominator, sample_period_s, order=4)


class _StateSpaceSISO5th(_StateSpaceSISONth):
    """Fifth-order discrete-time SISO state-space system."""

    def __init__(self) -> None:
        super().__init__(order=5)

    @classmethod
    def from_continuous_tf(
        cls,
        numerator: np.ndarray,
        denominator: np.ndarray,
        sample_period_s: float,
    ) -> _StateSpaceSISO5th:
        return super().from_continuous_tf(numerator, denominator, sample_period_s, order=5)


class _StateSpaceSISO6th(_StateSpaceSISONth):
    """Sixth-order discrete-time SISO state-space system."""

    def __init__(self) -> None:
        super().__init__(order=6)

    @classmethod
    def from_continuous_tf(
        cls,
        numerator: np.ndarray,
        denominator: np.ndarray,
        sample_period_s: float,
    ) -> _StateSpaceSISO6th:
        return super().from_continuous_tf(numerator, denominator, sample_period_s, order=6)


# Mapping from denominator degree to SS class
_SS_CLASS_MAP = {
    1: _StateSpaceSISO1st,
    2: _StateSpaceSISO2nd,
    3: _StateSpaceSISO3rd,
    4: _StateSpaceSISO4th,
    5: _StateSpaceSISO5th,
    6: _StateSpaceSISO6th,
}


def build_ss_from_tf(
    numerator: np.ndarray,
    denominator: np.ndarray,
    sample_period_s: float,
):
    """Factory: build the correct-order _StateSpaceSISO from a continuous TF.

    Returns an instance of _StateSpaceSISO{1st..6th}.
    """
    N = len(denominator) - 1
    if N not in _SS_CLASS_MAP:
        raise ValueError(f"TF denominator degree must be 1–6, got {N}")
    return _SS_CLASS_MAP[N].from_continuous_tf(numerator, denominator, sample_period_s)
