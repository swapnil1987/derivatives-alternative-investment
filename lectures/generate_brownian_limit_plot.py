"""Generate the Brownian-motion-as-a-limit figure for lecture 5."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def simulate_brownian(seed=516181, n_steps=2_048):
    """Simulate one Brownian path on [0, 1] with Var(W_1)=1."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_steps
    increments = rng.standard_normal(n_steps) * np.sqrt(dt)
    path = np.r_[0.0, np.cumsum(increments)]
    time = np.linspace(0.0, 1.0, n_steps + 1)
    return time, path


def sampled_path(time, path, n_steps):
    idx = np.linspace(0, len(time) - 1, n_steps + 1, dtype=int)
    return time[idx], path[idx]


def main():
    out_dir = Path(__file__).resolve().parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    time, path = simulate_brownian()
    panels = [
        (16, r"$\Delta t=1/16$"),
        (64, r"$\Delta t=1/64$"),
        (256, r"$\Delta t=1/256$"),
        (2_048, r"$\Delta t=1/2048$"),
    ]

    y_min, y_max = path.min(), path.max()
    y_pad = 0.12 * (y_max - y_min)

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.6), sharey=True)

    for ax, (n_steps, subtitle) in zip(axes, panels):
        t, w = sampled_path(time, path, n_steps)
        linewidth = 1.9 if n_steps <= 64 else 1.1

        ax.plot(t, w, color="#2457A6", lw=linewidth)
        ax.axhline(0.0, color="#B6B6B6", lw=0.8, zorder=0)
        ax.set_title(f"{n_steps} steps", fontsize=13, fontweight="bold", pad=8)
        ax.text(
            0.5,
            -0.18,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            color="#555555",
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["0", "T"], fontsize=9)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#8A8A8A")
        ax.spines["bottom"].set_color("#8A8A8A")
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    axes[0].set_ylabel(r"$W_t$", fontsize=12, rotation=0, labelpad=14)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"brownian-motion-limit.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
