from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax

if TYPE_CHECKING:
    from lotf.objects.quadrotor_state import QuadrotorState


@runtime_checkable
class Scheme(Protocol):
    """Protocol for a disjoint quadrotor integration path.

    Each scheme implements one exclusive forward dynamics method.
    Schemes are NOT composable -- they are selected via discriminated union config.
    """

    def integrate(
        self,
        ap_z: jax.Array,
        omega_d: jax.Array,
        state: QuadrotorState,
        dt: jax.Array,
    ) -> QuadrotorState:
        """Integrate one timestep.

        Parameters
        ----------
        ap_z : jax.Array
            Thrust-axis acceleration [m/s²] (scalar).
        omega_d : jax.Array
            Desired body rates [rad/s] (3,).
        state : QuadrotorState
            Current state: {p, R, v} + scheme_state dict.
        dt : jax.Array
            Integration timestep [s].

        Returns
        -------
        QuadrotorState
            Next state after dt.
        """
        ...
