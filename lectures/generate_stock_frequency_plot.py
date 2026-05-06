"""Generate the stock-frequency sampling figure for lecture 5."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def simulate_path(seed=181116, s0=100.0, mu=0.08, sigma=0.25, n_seconds=23_400):
    """Simulate a single geometric Brownian motion path on a normalized horizon."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_seconds
    shocks = rng.standard_normal(n_seconds)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    prices = s0 * np.exp(np.r_[0.0, np.cumsum(log_returns)])
    time = np.linspace(0.0, 1.0, n_seconds + 1)
    return time, prices


def sampled_indices(n_total, n_obs):
    return np.linspace(0, n_total - 1, n_obs, dtype=int)


def main():
    out_dir = Path(__file__).resolve().parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    time, prices = simulate_path()
    n_total = len(time)

    panels = [
        ("Monthly", 13, "12 intervals"),
        ("Weekly", 53, "52 intervals"),
        ("Daily", 253, "252 intervals"),
        ("Minutes", 391, "390 intervals"),
        ("Seconds", n_total, "23,400 intervals"),
    ]

    fig = plt.figure(figsize=(12.8, 5.1))
    grid = fig.add_gridspec(2, 6, hspace=0.58, wspace=0.55)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]

    y_min, y_max = prices.min(), prices.max()
    y_pad = 0.08 * (y_max - y_min)

    for ax, (title, n_obs, label) in zip(axes, panels):
        idx = sampled_indices(n_total, n_obs)
        linewidth = 1.8 if n_obs < 500 else 0.55
        alpha = 1.0 if n_obs < 500 else 0.88

        ax.plot(time[idx], prices[idx], color="#2457A6", lw=linewidth, alpha=alpha)
        ax.scatter(time[idx[0]], prices[idx[0]], s=12, color="#2457A6", zorder=3)
        ax.scatter(time[idx[-1]], prices[idx[-1]], s=12, color="#2457A6", zorder=3)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.text(
            0.5,
            -0.18,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            color="#666666",
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#8A8A8A")
        ax.spines["bottom"].set_color("#8A8A8A")
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"stock-frequency-sampling.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
