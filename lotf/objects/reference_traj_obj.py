from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from jax.scipy.spatial.transform import Rotation

from lotf import LOTF_PATH


class RefTrajNames(Enum):
    CIRCLE = "circle"
    FIG8 = "fig8"
    STAR = "star"


### csv file paths ###
CIRCLE_CSV = LOTF_PATH + "/objects/ref_traj_files/circle.csv"
FIG8_CSV = LOTF_PATH + "/objects/ref_traj_files/fig8.csv"
STAR_CSV = LOTF_PATH + "/objects/ref_traj_files/star.csv"


@dataclass
class Fig8Config:
    """Configuration for the figure-8 (lemniscate) reference trajectory.

    Attributes:
        a: Lemniscate amplitude [m].
        height: Constant flight height [m].
        duration: Total trajectory duration [s].
        num_points: Number of waypoints.
    """

    a: float = 4.0
    height: float = 1.0
    duration: float = 12.0
    num_points: int = 600

    def to_kwargs(self) -> dict[str, float | int]:
        return {
            "a": self.a,
            "height": self.height,
            "duration": self.duration,
            "num_points": self.num_points,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> Fig8Config:
        if d is None:
            return cls()
        return cls(
            a=d.get("a", 4.0),
            height=d.get("height", 1.0),
            duration=d.get("duration", 12.0),
            num_points=d.get("num_points", 600),
        )


@jdc.pytree_dataclass
class ReferenceTraj:
    """
    Represents a reference trajectory.
    Use `from_name` to load a predefined reference trajectory or
    `from_csv` to load a reference trajectory from a csv file.

    >>> ref_traj_obj = ReferenceTraj.from_name(RefTrajNames.CIRCLE)
    """

    ref_traj: jnp.array
    num_waypoints: int
    pos_bounds: jnp.array
    vel_bounds: jnp.array

    @classmethod
    def from_name(
        cls, name: Union[str, RefTrajNames], fig8_config: Fig8Config | None = None
    ) -> ReferenceTraj:

        if isinstance(name, RefTrajNames):
            name = name.value

        if name == RefTrajNames.CIRCLE.value:
            return ReferenceTraj.from_csv(CIRCLE_CSV)
        elif name == RefTrajNames.FIG8.value:
            cfg = fig8_config or Fig8Config()
            return ReferenceTraj.gen_fig8(**cfg.to_kwargs())
        elif name == RefTrajNames.STAR.value:
            return ReferenceTraj.from_csv(STAR_CSV)
        else:
            raise ValueError(f"Unknown track name: {name}")

    @classmethod
    def from_csv(cls, path: str) -> ReferenceTraj:

        ref_traj = jnp.array(np.loadtxt(path))
        assert ref_traj.shape[1] == 30, (
            f"Expected 30 columns in trajectory, got {ref_traj.shape[1]}"
        )
        num_waypoints = ref_traj.shape[0]

        # compute position and velocity bounds
        pos_bounds = jnp.array(
            [
                jnp.min(ref_traj[:, TrajColumns.POS.slice], axis=0),
                jnp.max(ref_traj[:, TrajColumns.POS.slice], axis=0),
            ]
        )
        vel_bounds = jnp.array(
            [
                jnp.min(ref_traj[:, TrajColumns.VEL.slice], axis=0),
                jnp.max(ref_traj[:, TrajColumns.VEL.slice], axis=0),
            ]
        )

        # noinspection PyArgumentList
        return cls(ref_traj, num_waypoints, pos_bounds, vel_bounds)

    @classmethod
    def gen_fig8(cls, a=4.0, height=1.0, duration=12.0, num_points=600) -> ReferenceTraj:
        """Generate a figure-8 (lemniscate) trajectory analytically.

        Parametric equation:
            x(t) = a * sin(t)
            y(t) = a * sin(t) * cos(t)
            z(t) = height

        Orientation is thrust-aligned: body z-axis aligns with net acceleration
        direction [ax, ay, az+g], body x-axis aligns with velocity direction
        projected onto the plane perpendicular to body z.

        Args:
            a: amplitude [m].
            height: constant flight height [m].
            duration: total trajectory duration [s].
            num_points: number of waypoints.

        Returns:
            ReferenceTraj with shape (num_points, 30).
        """
        shift = jnp.pi / 2.0
        t_vals = jnp.linspace(shift, shift + 2.0 * jnp.pi, num_points)

        s = jnp.sin(t_vals)
        c = jnp.cos(t_vals)
        s2 = jnp.sin(2.0 * t_vals)
        c2 = jnp.cos(2.0 * t_vals)

        # position
        px = a * s
        py = a * s * c
        pz = jnp.full_like(px, height)

        # velocity (1st derivative)
        vx = a * c
        vy = a * c2
        vz = jnp.zeros_like(vx)

        # acceleration (2nd derivative)
        ax = -a * s
        ay = -2.0 * a * s2
        az = jnp.zeros_like(ax)

        # jerk (3rd derivative)
        jx = -a * c
        jy = -4.0 * a * c2
        jz = jnp.zeros_like(jx)

        # snap (4th derivative)
        sx = a * s
        sy = 8.0 * a * s2
        sz = jnp.zeros_like(sx)

        # time
        time = jnp.linspace(0.0, duration, num_points)

        # thrust-aligned orientation: body_z || [ax, ay, g]
        g = 9.81
        thrust_dir = jnp.stack([ax, ay, jnp.full_like(az, g)], axis=-1)
        thrust_dir = thrust_dir / (jnp.linalg.norm(thrust_dir, axis=-1, keepdims=True) + 1e-8)
        body_z = thrust_dir

        # body_x: velocity direction projected onto plane perpendicular to body_z
        vel = jnp.stack([vx, vy, vz], axis=-1)
        vel_norm = jnp.linalg.norm(vel, axis=-1, keepdims=True) + 1e-8
        vel_dir = vel / vel_norm
        dot_vz = jnp.sum(vel_dir * body_z, axis=-1, keepdims=True)
        body_x_proj = vel_dir - dot_vz * body_z
        body_x_norm = jnp.linalg.norm(body_x_proj, axis=-1, keepdims=True) + 1e-8
        body_x = body_x_proj / body_x_norm

        # body_y = body_z x body_x
        body_y = jnp.cross(body_z, body_x)

        # rotation matrix: columns are body axes in world frame
        R = jnp.stack([body_x, body_y, body_z], axis=-1)

        # convert to quaternion [qw, qx, qy, qz] convention
        def _matrix_to_quat(m):
            r = Rotation.from_matrix(m)
            q = r.as_quat()
            return jnp.array([q[3], q[0], q[1], q[2]])

        quat = jax.vmap(_matrix_to_quat)(R)

        # omega: heading angular rate (in body frame, z component only)
        v_sq = vx * vx + vy * vy + 1e-8
        omega_z = (vx * ay - vy * ax) / v_sq
        omega = jnp.stack([jnp.zeros_like(omega_z), jnp.zeros_like(omega_z), omega_z], axis=-1)

        # alpha, commands: zero
        alpha = jnp.zeros((num_points, 3))
        commands = jnp.zeros((num_points, 4))

        # assemble 30-column array
        ref_traj = jnp.column_stack(
            [
                time[:, None],  # 0:1  TIME
                jnp.stack([px, py, pz], axis=-1),  # 1:4  POS
                quat,  # 4:8  QUAT
                jnp.stack([vx, vy, vz], axis=-1),  # 8:11 VEL
                omega,  # 11:14 OMEGA
                jnp.stack([ax, ay, az], axis=-1),  # 14:17 ACC
                alpha,  # 17:20 ALPHA
                commands,  # 20:24 COMMANDS
                jnp.stack([jx, jy, jz], axis=-1),  # 24:27 JERK
                jnp.stack([sx, sy, sz], axis=-1),  # 27:30 SNAP
            ]
        )

        assert ref_traj.shape[1] == 30

        pos_bounds = jnp.array(
            [
                jnp.min(ref_traj[:, TrajColumns.POS.slice], axis=0),
                jnp.max(ref_traj[:, TrajColumns.POS.slice], axis=0),
            ]
        )
        vel_bounds = jnp.array(
            [
                jnp.min(ref_traj[:, TrajColumns.VEL.slice], axis=0),
                jnp.max(ref_traj[:, TrajColumns.VEL.slice], axis=0),
            ]
        )

        # noinspection PyArgumentList
        return cls(ref_traj, num_points, pos_bounds, vel_bounds)

    @classmethod
    def default_traj(cls) -> ReferenceTraj:
        cls.from_name(RefTrajNames.CIRCLE)


class TrajColumns(Enum):
    TIME = (0, 1)
    POS = (1, 4)
    QUAT = (4, 8)
    VEL = (8, 11)
    OMEGA = (11, 14)
    ACC = (14, 17)
    ALPHA = (17, 20)
    COMMANDS = (20, 24)
    JERK = (24, 27)
    SNAP = (27, 30)

    @property
    def start(self):
        return self.value[0]

    @property
    def end(self):
        return self.value[1]

    @property
    def slice(self):
        return slice(self.start, self.end)
