"""Create a conventional full-range ROC plot from cached measured coordinates."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv-roc/matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def main():
    source = ROOT / "results/roc_curve_3class_demo_style.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    series = [
        ("Healthy_Baseline", "Healthy baseline", "#228833", "-"),
        ("Continuous_Friction_and_Wear", "Friction and wear", "#0077bb", "--"),
        ("Impulsive_Shocks_and_Structural", "Impulsive / structural", "#cc3311", "-."),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1.2,
            label="Chance (AUC = 0.5000)", zorder=1)
    for key, label, color, style in series:
        curve = data["curves"][key]
        ax.plot(curve["fpr"], curve["tpr"], color=color, linestyle=style,
                linewidth=2.0, label=f"{label} (AUC = {curve['auc']:.4f})", zorder=3)
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.025),
           xlabel="False Positive Rate (1 − Specificity)",
           ylabel="True Positive Rate (Sensitivity)",
           title="ROC Curves — Three-Class TF-FaultNet")
    ax.set_title("ROC Curves — Three-Class TF-FaultNet", fontsize=14, pad=14)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(0.2))
        axis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(alpha=0.18, linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1],
              loc="lower right", framealpha=1.0, edgecolor="0.85", fontsize=10)
    for extension in ("png", "pdf", "svg"):
        output = ROOT / f"results/visualizations/roc_curve_3class_standard.{extension}"
        fig.savefig(output, dpi=300, facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
