"""Plot the measured landing coordinates from the 2025-12-16 workbook."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    output = root / "ground_truth_plots"
    output.mkdir(exist_ok=True)
    cases = [
        ("normal", "l / Normal", "#2563eb"),
        ("clutter", "c / Clutter", "#d97706"),
        ("dark", "d / Dark", "#7c3aed"),
        ("dark_clutter", "dc / Dark clutter", "#0891b2"),
    ]
    data = {name: np.loadtxt(root / f"{name}.csv", delimiter=",")
            for name, _, _ in cases}
    valid = {name: points[np.isfinite(points).all(axis=1)]
             for name, points in data.items()}
    combined = np.concatenate(list(valid.values()))
    lower = np.floor((combined.min(axis=0) - 0.05) * 10) / 10
    upper = np.ceil((combined.max(axis=0) + 0.05) * 10) / 10
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 16,
                         "axes.labelsize": 13, "savefig.facecolor": "white"})
    for name, title, color in cases:
        points = valid[name]
        fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
        ax.scatter(points[:, 0], points[:, 1], s=38, color=color,
                   alpha=0.78, edgecolors="white", linewidths=0.5, zorder=3)
        ax.axhline(0, color="#94a3b8", linewidth=0.8)
        ax.axvline(0, color="#94a3b8", linewidth=0.8)
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
               xlabel="x_m (m)", ylabel="y_m (m)",
               title=f"Ground truth — {title}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        missing = len(data[name]) - len(points)
        ax.text(0.03, 0.97, f"n = {len(points)}" +
                (f" | missing: {missing} (ID 12)" if missing else ""),
                transform=ax.transAxes, va="top", color="#334155")
        path = output / f"ground_truth_{name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"{path}: {len(points)} points")


if __name__ == "__main__":
    main()
