"""Display-only ROC smoothing. Scores, empirical coordinates, and metrics stay fixed."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv-roc/matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth_path(fpr, tpr):
    """Gaussian smoothing along arc length handles vertical segments and ties.

    Positive kernel weights preserve coordinate monotonicity. This is a drawing
    aid, not an estimated ROC or a source for new performance measurements.
    """
    points = np.column_stack([fpr, tpr])
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 0]
    distance, points = distance[keep], points[keep]
    grid = np.linspace(0, distance[-1], 4001)
    coordinates = np.column_stack([np.interp(grid, distance, points[:, i]) for i in (0, 1)])
    offsets = np.arange(-32, 33)
    kernel = np.exp(-0.5 * (offsets / 8.0) ** 2)
    kernel /= kernel.sum()
    smoothed = np.column_stack([
        np.convolve(np.pad(coordinates[:, i], (32, 32), mode="edge"), kernel, mode="valid")
        for i in (0, 1)
    ])
    smoothed[0], smoothed[-1] = points[0], points[-1]
    if not (np.isfinite(smoothed).all() and np.all(np.diff(smoothed, axis=0) >= -1e-12)):
        raise ValueError("Smoothing produced invalid or nonmonotonic coordinates")
    if np.max(np.abs(smoothed - coordinates)) > 0.005:
        raise ValueError("Display smoothing moved a coordinate more than 0.5 percentage points")
    return smoothed


def main():
    data = json.loads((ROOT / "results/roc_curve_3class_demo_style.json").read_text())
    series = [
        ("Healthy_Baseline", "Healthy baseline", "#228833"),
        ("Continuous_Friction_and_Wear", "Friction and wear", "#0077bb"),
        ("Impulsive_Shocks_and_Structural", "Impulsive / structural", "#cc3311"),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    for detail in (False, True):
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.18, top=0.87)
        for key, name, color in series:
            curve = data["curves"][key]
            points = smooth_path(curve["fpr"], curve["tpr"])
            ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2,
                    label=f"{name} (empirical AUC = {curve['auc']:.4f})")
            # Mark direction changes in the measured path rather than hundreds
            # of collinear samples, keeping the original evidence readable.
            original = np.column_stack([curve["fpr"], curve["tpr"]])
            steps = np.diff(original, axis=0)
            corners = np.flatnonzero(np.any(np.diff(np.sign(steps), axis=0) != 0, axis=1)) + 1
            ax.scatter(original[corners, 0], original[corners, 1], s=12, color=color,
                       alpha=0.4, linewidths=0, zorder=4)
        if detail:
            ax.set_xlim(-0.001, 0.08)
            ax.set_ylim(0.92, 1.004)
            suffix = "_detail"
            title = "Three-Class ROC — Smoothed Display (Zoomed)"
        else:
            ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1.1, label="Chance (AUC = 0.5)")
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.025)
            suffix = ""
            title = "Three-Class ROC — Smoothed Display"
        ax.set_title(title, fontsize=16, pad=16)
        ax.set_xlabel("False Positive Rate (1 − Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.grid(alpha=0.15)
        ax.legend(loc="lower right", fontsize=10, framealpha=1, edgecolor="0.85")
        fig.text(0.11, 0.045, "Lines smoothed for display only; dots and AUCs use measured ROC. Cached ONNX scores, N = 430.",
                 fontsize=9, color="0.4")
        for ext in ("png", "pdf", "svg"):
            output = ROOT / f"results/visualizations/roc_curve_3class_smoothed{suffix}.{ext}"
            fig.savefig(output, dpi=250, facecolor="white")
            print(output)
        plt.close(fig)


if __name__ == "__main__":
    main()
