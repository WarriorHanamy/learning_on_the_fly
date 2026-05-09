"""Quadrotor kinematic state with opaque scheme-private state dict."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from lotf.utils.pytrees import CustomPyTree, field_jnp


@jdc.pytree_dataclass
class QuadrotorState(CustomPyTree):
    """Kinematic state of a quadrotor plus opaque scheme-private state.

    Attributes:
        p: Position in world frame [m].
        R: Rotation matrix body→world (3×3).
        v: Velocity in world frame [m/s].
        scheme_state: Opaque dict carrying scheme-private data.
            Keys use prefix convention (e.g. ``approx_delay_buffer``,
            ``inner_loop_motor_omega``) to avoid collisions between
            different schemes.
    """

    p: jax.Array = field_jnp([0.0, 0.0, 0.0])
    R: jax.Array = field_jnp(jnp.eye(3))
    v: jax.Array = field_jnp([0.0, 0.0, 0.0])
    scheme_state: dict = jdc.field(default_factory=lambda: {})

    def detached(self) -> QuadrotorState:
        """Returns a copy with gradients stopped on all leaves."""
        return QuadrotorState(
            p=jax.lax.stop_gradient(self.p),
            R=jax.lax.stop_gradient(self.R),
            v=jax.lax.stop_gradient(self.v),
            scheme_state=jax.tree.map(jax.lax.stop_gradient, self.scheme_state),
        )

    def as_vector(self) -> jax.Array:
        """Serializes kinematic state into a flat array.

        Includes protocol fields from scheme_state when available
        for backward compatibility with existing serialization code.
        """
        ss = self.scheme_state
        omega = ss.get("inner_loop_omega", jnp.zeros(3))
        domega = ss.get("inner_loop_domega", jnp.zeros(3))
        motor_omega = ss.get("inner_loop_motor_omega", jnp.zeros(4))
        return jnp.concatenate([self.p, self.R.flatten(), self.v, omega, domega, motor_omega])

    @classmethod
    def from_vector(cls, vector: jax.Array) -> QuadrotorState:
        """Reconstructs state from a flattened array."""
        p = vector[:3]
        R = vector[3:12].reshape(3, 3)
        v = vector[12:15]
        omega = vector[15:18]
        domega = vector[18:21]
        motor_omega = vector[21:]
        return cls(
            p=p,
            R=R,
            v=v,
            scheme_state={
                "inner_loop_omega": omega,
                "inner_loop_domega": domega,
                "inner_loop_motor_omega": motor_omega,
            },
        )
