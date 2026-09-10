"""Read-only research checks; writes a separate audit, never trains or tunes."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import wave
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = {"scope": "Saved-artifact verification, not new model evaluation"}
    sources = sorted(ROOT.glob("*.py")) + sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    inventory = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        inventory.append({"file": str(path.relative_to(ROOT)), "lines": len(path.read_text(encoding="utf-8-sig").splitlines()), "definitions": [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]})
    result["python_inventory"] = inventory
    result["notebooks"] = []
    for path in ROOT.glob("*.ipynb"):
        nb = json.loads(path.read_text(encoding="utf-8"))
        cells = ["".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"]
        result["notebooks"].append({"file": path.name, "code_cells": len(cells), "saved_outputs": sum(bool(c.get("outputs")) for c in nb["cells"]), "code": cells})
    rows = list(csv.DictReader((ROOT / "data/splits.csv").open(encoding="utf-8-sig")))
    result["split_counts"] = dict(Counter(r["split_multiclass_3class"] for r in rows))
    result["class_counts"] = dict(Counter(r["macro_class_index"] for r in rows))
    result["manifest_sha256"] = digest(ROOT / "data/splits.csv")
    wavs = sorted((ROOT / "Base_de_Dados").rglob("*.wav"))
    formats, pcm_hashes = Counter(), Counter()
    pcm = {}
    minimum, maximum, clipped, total = 32767, -32768, 0, 0
    for path in wavs:
        with wave.open(str(path), "rb") as w:
            key = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes())
            formats[str(key)] += 1
            data = w.readframes(w.getnframes())
        pcm_hashes[hashlib.sha256(data).hexdigest()] += 1
        assert key == (1, 2, 44100, 44100), (path, key)
        samples = np.frombuffer(data, dtype="<i2")
        minimum, maximum = min(minimum, int(samples.min())), max(maximum, int(samples.max()))
        clipped += int(np.sum((samples == -32768) | (samples == 32767)))
        total += len(samples)
        pcm[path.relative_to(ROOT).as_posix()] = samples
    result["audio"] = {"count": len(wavs), "formats_channels_bytes_rate_frames": dict(formats), "duplicate_pcm_excess": sum(n - 1 for n in pcm_hashes.values()), "normalized_min": minimum / 32768, "normalized_max": maximum / 32768, "full_scale_samples": clipped, "total_samples": total}
    cache = np.load(ROOT / "results/roc_demo_style_scores_3class.npz", allow_pickle=False)
    archive = np.load(ROOT / "results/heldout_test_set_3class.npz", allow_pickle=False)
    y, probs = cache["labels"], cache["probabilities"]
    assert np.array_equal(y, archive["labels"])
    assert np.array_equal(cache["filenames"], archive["filenames"])
    for key, path in [("model_sha256", "results/checkpoints/best_tf_faultnet_3class_backbone.onnx"), ("data_sha256", "results/heldout_test_set_3class.npz")]:
        assert str(cache[key]) == digest(ROOT / path)
    def normalized(name):
        return "Base_de_Dados/" + str(name).replace("\\", "/").split("Base_de_Dados/", 1)[1]
    expected = {r["filepath"].replace("\\", "/"): int(r["macro_class_index"]) for r in rows if r["split_multiclass_3class"] == "test"}
    actual = {normalized(f): int(l) for f, l in zip(archive["filenames"], y)}
    assert actual == expected and len(actual) == len(y)
    archive_audio = archive["audio"]
    assert all(np.array_equal(archive_audio[i], pcm[normalized(f)].astype(np.float32) / 32768) for i, f in enumerate(archive["filenames"]))
    pred = probs.argmax(1)
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (y, pred), 1)
    precision = np.diag(cm) / cm.sum(0)
    recall = np.diag(cm) / cm.sum(1)
    f1 = 2 * precision * recall / (precision + recall)
    p, n, z = float(np.mean(y == pred)), len(y), 1.959963984540054
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    result["cached_prediction_check"] = {"model_and_archive_hashes_match": True, "archive_matches_manifest_and_raw_audio": True, "confusion_matrix_true_rows": cm.tolist(), "accuracy": p, "macro_f1": float(f1.mean()), "wilson_95_accuracy_iid_only": [center-half, center+half], "max_probability_sum_error": float(np.max(np.abs(probs.sum(1)-1)))}
    features = np.load(ROOT / "data/baseline_statistical_features_3class.npz", allow_pickle=False)
    result["feature_cache"] = {"keys": features.files}
    for split, key in [("train", "X_train"), ("val", "X_val"), ("test", "X_test")]:
        selected = [r for r in rows if r["split_multiclass_3class"] == split]
        old, corrected, rms = [], [], []
        for row in selected:
            a = pcm[row["filepath"].replace("\\", "/")].astype(np.float32) / 32768
            old.append(np.mean(np.diff(np.sign(a) != 0)))
            corrected.append(np.mean(np.signbit(a[:-1]) != np.signbit(a[1:])))
            rms.append(np.sqrt(np.mean(a*a)+1e-10))
        x = features[key]
        result["feature_cache"][split] = {"shape": list(x.shape), "finite": bool(np.isfinite(x).all()), "rms_matches_manifest_order": bool(np.allclose(x[:, 0], rms)), "old_zcr_matches": bool(np.allclose(x[:, 13], old)), "corrected_zcr_matches": bool(np.allclose(x[:, 13], corrected)), "cached_zcr_range": [float(x[:, 13].min()), float(x[:, 13].max())], "corrected_zcr_range": [float(min(corrected)), float(max(corrected))]}
    out = ROOT / "outputs/supervisor_review"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("python_inventory", "notebooks")}, indent=2))
    print(f"Parsed {len(inventory)} Python files; inspected {len(result['notebooks'])} notebooks. Audit: {out / 'audit.json'}")


if __name__ == "__main__":
    main()
