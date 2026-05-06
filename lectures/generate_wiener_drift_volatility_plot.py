"""Generate drift and volatility comparison figure for lecture 5."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def simulate_wiener(seed=516001, n_steps=1_000):
    """Simulate a standard Wiener process W_t on [0, 1]."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_steps
    increments = rng.standard_normal(n_steps) * np.sqrt(dt)
    time = np.linspace(0.0, 1.0, n_steps + 1)
    wiener = np.r_[0.0, np.cumsum(increments)]
    return time, wiener


def main():
    out_dir = Path(__file__).resolve().parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    time, wiener = simulate_wiener()
    drift = 1.0
    vol = 0.65

    cases = [
        ("Wiener process", r"$x_t=W_t$", wiener, None),
        ("Drift only", r"$dx=a\,dt,\quad a\ne 0$", drift * time, None),
        ("Volatility only", r"$dx=b\,dW_t,\quad b\ne 0$", vol * wiener, None),
        (
            "Drift and volatility",
            r"$dx=a\,dt+b\,dW_t,\quad a,b\ne 0$",
            drift * time + vol * wiener,
            drift * time,
        ),
    ]

    y_min = min(values.min() for _, _, values, _ in cases)
    y_max = max(values.max() for _, _, values, _ in cases)
    y_pad = 0.12 * (y_max - y_min)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.2), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (title, subtitle, values, trend) in zip(axes, cases):
        ax.plot(time, values, color="#2457A6", lw=1.8)
        if trend is not None:
            ax.plot(time, trend, color="#7A7A7A", lw=1.2, ls="--")

        ax.axhline(0.0, color="#B6B6B6", lw=0.8, zorder=0)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.text(
            0.03,
            0.94,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
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

    axes[0].set_ylabel(r"$x_t$", fontsize=12, rotation=0, labelpad=15)
    axes[2].set_ylabel(r"$x_t$", fontsize=12, rotation=0, labelpad=15)
    fig.tight_layout(pad=1.0, h_pad=1.9, w_pad=1.8)

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"wiener-drift-volatility.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
