"""Restyle the locked three-class model's ROC without tuning or retraining.

Run with .venv-roc/Scripts/python.exe scripts/plot_roc_demo_style.py.
Requires numpy, matplotlib, and onnxruntime. Cached scores avoid repeated inference.
"""

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv-roc" / "matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empirical_roc(target, scores):
    """Binary ROC at every distinct score, keeping tied observations together."""
    target = np.asarray(target, dtype=np.int64)
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or not 0 < target.sum() < len(target):
        raise ValueError("ROC requires finite scores and both positive and negative samples")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    true_positive = np.cumsum(target[order])[ends]
    false_positive = 1 + ends - true_positive
    tpr = np.r_[0.0, true_positive / target.sum()]
    fpr = np.r_[0.0, false_positive / (len(target) - target.sum())]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def main():
    model_path = ROOT / "results/checkpoints/best_tf_faultnet_3class_backbone.onnx"
    data_path = ROOT / "results/heldout_test_set_3class.npz"
    report = json.loads((ROOT / "results/heldout_test_evaluation_report_3class.json").read_text())
    output_dir = ROOT / "results/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = ROOT / "results/roc_demo_style_scores_3class.npz"
    fingerprints = {"model_sha256": sha256(model_path), "data_sha256": sha256(data_path)}

    with np.load(data_path, allow_pickle=False) as archive:
        labels = archive["labels"]
        filenames = archive["filenames"]
        specs = archive["spectrogram_images"]
        class_names = archive["class_names"].tolist()
    if class_names != report["classes"] or len(labels) != report["sample_count"]:
        raise ValueError("Archive taxonomy or sample count differs from saved report")
    with (ROOT / "data/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        frozen = {
            row["filepath"].replace("\\", "/"): int(row["macro_class_index"])
            for row in csv.DictReader(handle) if row["split_multiclass_3class"] == "test"
        }
    archive_labels = {}
    for filename, label in zip(filenames, labels):
        normalized = str(filename).replace("\\", "/")
        normalized = "Base_de_Dados/" + normalized.split("Base_de_Dados/", 1)[1]
        archive_labels[normalized] = int(label)
    if archive_labels != frozen or len(archive_labels) != len(labels):
        raise ValueError("Test archive does not match frozen manifest")

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if any(str(cache[key]) != value for key, value in fingerprints.items()):
                raise ValueError("Cached scores belong to different model/data; review before replacing")
            probs = cache["probabilities"]
    else:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(model_path), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        logits = np.concatenate([
            session.run(None, {input_name: np.ascontiguousarray(spec[None], dtype=np.float32)})[0]
            for spec in specs
        ])
        exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exponentials / exponentials.sum(axis=1, keepdims=True)
        np.savez_compressed(cache_path, probabilities=probs, labels=labels,
                            filenames=filenames, **fingerprints)

    accuracy = float(np.mean(probs.argmax(axis=1) == labels))
    if not np.isclose(accuracy, report["metrics"]["test_accuracy"], atol=1e-12, rtol=0):
        raise ValueError(f"ONNX accuracy {accuracy} differs from saved report; investigate before plotting")
    curves = [empirical_roc(labels == i, probs[:, i]) for i in range(3)]
    auc_differences = {}
    for i, (_, _, area) in enumerate(curves):
        difference = area - report["per_class"][i]["roc_auc"]
        auc_differences[class_names[i]] = difference
        # A single positive/negative pair is the smallest full ranking change.
        # Exported-runtime differences are recorded, never replaced by old scores.
        positives = int(np.sum(labels == i))
        pair_resolution = 1.0 / (positives * (len(labels) - positives))
        if abs(difference) > pair_resolution + 1e-12:
            raise ValueError(f"Class {i} AUC differs from saved report: {area}")

    # Match the reference's uncluttered full-scale ROC layout. Curves remain empirical.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    colors = ["#1565c0", "#e97924", "#238b45"]
    styles = ["-", "--", ":"]
    names = ["Healthy baseline", "Friction and wear", "Impulsive / structural"]
    for i, (fpr, tpr, area) in enumerate(curves):
        ax.plot(fpr, tpr, color=colors[i], linestyle=styles[i], linewidth=2.2,
                zorder=4 + i, label=f"{names[i]}  (AUC = {area:.4f})")
    ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.3, linestyle="--",
            label="Chance reference  (AUC = 0.5000)")
    ax.text(0.48, 0.42, "Chance reference", rotation=45, rotation_mode="anchor",
            color="#777777", fontsize=10)
    ax.annotate("Curves near the upper-left corner", xy=(0.018, 0.978),
                xytext=(0.20, 0.77), fontsize=11, color="#333333",
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1})
    ax.set(xlim=(-0.025, 1.025), ylim=(-0.025, 1.025),
           xlabel="False positive rate (1 - specificity)",
           ylabel="True positive rate (sensitivity)")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("ROC Curve", fontsize=21, weight="bold", pad=28)
    ax.text(0.5, 1.02, "TF-FaultNet | One-vs-rest | N = 430 | ONNX reconstruction",
            transform=ax.transAxes, ha="center", fontsize=10, color="#555555")
    ax.grid(False)
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=1,
              edgecolor="#dddddd", borderpad=1.0)
    for extension in ("png", "svg", "pdf"):
        path = output_dir / f"roc_curve_3class_demo_style.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)
    metadata = {
        "purpose": "Presentation-only reconstruction from locked ONNX model; no training or tuning",
        "comparison": "Accuracy matches saved report; AUC differences are at most one positive-negative pair ranking contribution",
        "auc_differences_from_saved_report": auc_differences,
        "test_accuracy": accuracy,
        "curves": {class_names[i]: {"fpr": f.tolist(), "tpr": t.tolist(), "auc": a}
                   for i, (f, t, a) in enumerate(curves)},
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        **fingerprints,
    }
    (ROOT / "results/roc_curve_3class_demo_style.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"accuracy": accuracy, "auc": [curve[2] for curve in curves]}, indent=2))


if __name__ == "__main__":
    main()
