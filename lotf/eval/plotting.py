"""Comparison plots for trajectory tracking benchmark suites."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lotf.eval.runner import BenchmarkRunResult
from lotf.objects.reference_traj_obj import TrajColumns


def _first_rollout_until_done(result: BenchmarkRunResult) -> np.ndarray:
    done = np.logical_or(result.transitions.terminated[0], result.transitions.truncated[0])
    done_indices = np.where(done)[0]
    end_idx = int(done_indices[0]) + 1 if len(done_indices) else result.transitions.reward.shape[1]
    return np.array(result.transitions.state.quadrotor_state.p[0, :end_idx])


def plot_benchmark_comparison(
    results: list[BenchmarkRunResult],
    ref_traj,
    save_path: str | Path,
) -> None:
    """Save one 2x2 comparison figure for benchmark suite results."""
    ref_pos = np.array(ref_traj)[:, TrajColumns.POS.slice]
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "nominal": "#4c78a8",
        "resacc": "#f58518",
        "innerloop": "#54a24b",
        "full": "#b279a2",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_xy, ax_xz, ax_rmse, ax_outcome = axes.ravel()

    ax_xy.plot(ref_pos[:, 0], ref_pos[:, 1], "k--", linewidth=2, label="reference")
    ax_xz.plot(ref_pos[:, 0], ref_pos[:, 2], "k--", linewidth=2, label="reference")

    for result in results:
        pos = _first_rollout_until_done(result)
        color = colors.get(result.label, None)
        ax_xy.plot(pos[:, 0], pos[:, 1], linewidth=1.8, color=color, label=result.label)
        ax_xz.plot(pos[:, 0], pos[:, 2], linewidth=1.8, color=color, label=result.label)

    for ax, title, ylabel in [
        (ax_xy, "XY plane", "Y position [m]"),
        (ax_xz, "XZ plane", "Z position [m]"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("X position [m]")
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small")

    labels = [r.label for r in results]
    x = np.arange(len(labels))
    width = 0.36

    pos_rmse = [r.metrics.position_rmse for r in results]
    vel_rmse = [r.metrics.velocity_rmse for r in results]
    ax_rmse.bar(x - width / 2, pos_rmse, width, label="position RMSE [m]", color="#4c78a8")
    ax_rmse.bar(x + width / 2, vel_rmse, width, label="velocity RMSE [m/s]", color="#f58518")
    ax_rmse.set_title("Tracking error")
    ax_rmse.set_xticks(x, labels, rotation=20)
    ax_rmse.grid(True, axis="y", alpha=0.25)
    ax_rmse.legend(fontsize="small")

    returns = [r.metrics.mean_episodic_return for r in results]
    collisions = [r.metrics.collision_rate for r in results]
    ax_return = ax_outcome
    ax_collision = ax_outcome.twinx()
    ax_return.bar(x - width / 2, returns, width, label="mean return", color="#54a24b")
    ax_collision.bar(x + width / 2, collisions, width, label="collision rate", color="#e45756")
    ax_return.set_title("Benchmark outcome")
    ax_return.set_xticks(x, labels, rotation=20)
    ax_return.set_ylabel("Mean return")
    ax_collision.set_ylabel("Collision rate")
    ax_return.grid(True, axis="y", alpha=0.25)

    lines_1, labels_1 = ax_return.get_legend_handles_labels()
    lines_2, labels_2 = ax_collision.get_legend_handles_labels()
    ax_return.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize="small")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
