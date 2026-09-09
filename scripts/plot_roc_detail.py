"""Plot a clearly labeled enlargement of cached ROC coordinates; no inference."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv-roc/matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter


def main():
    data = json.loads((ROOT / "results/roc_curve_3class_demo_style.json").read_text())
    styles = [
        ("Healthy_Baseline", "Healthy", "#22804b", "-"),
        ("Continuous_Friction_and_Wear", "Friction / wear", "#2166b5", "--"),
        ("Impulsive_Shocks_and_Structural", "Impulsive / structural", "#d46425", "-."),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.23, top=0.81)
    for key, label, color, style in styles:
        curve = data["curves"][key]
        ax.plot(curve["fpr"], curve["tpr"], color=color, linestyle=style,
                linewidth=2.4, label=f"{label} | full AUC {curve['auc']:.4f}")
    ax.set_xlim(-0.001, 0.08)
    ax.set_ylim(0.92, 1.003)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("False positive rate", labelpad=10)
    ax.set_ylabel("True positive rate", labelpad=10)
    ax.grid(alpha=0.20, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.10, 0.94, "Three-class ROC — enlarged detail", fontsize=22, weight="bold")
    fig.text(0.10, 0.885, "Zoom: false positive rate 0–8% · true positive rate 92–100%",
             fontsize=12, color="#555555")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              frameon=False, fontsize=10, handlelength=3, columnspacing=1.6)
    fig.text(0.10, 0.025,
             "430 test recordings · Cached ONNX scores · AUC uses the entire ROC, not this zoomed region",
             fontsize=10, color="#666666")
    for ext in ("png", "pdf", "svg"):
        path = ROOT / f"results/visualizations/roc_curve_3class_enlarged.{ext}"
        fig.savefig(path, dpi=250, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
