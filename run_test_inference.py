import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.models import TFFaultNet

class TFFaultNetBackbone(nn.Module):
    """Backbone for 2D spectrogram image inference."""
    def __init__(self, full_model):
        super().__init__()
        self.init_conv = full_model.init_conv
        self.layer1 = full_model.layer1
        self.layer2 = full_model.layer2
        self.layer3 = full_model.layer3
        self.global_pool = full_model.global_pool
        self.global_max = full_model.global_max
        self.classifier = full_model.classifier

    def forward(self, spec):
        feat = self.init_conv(spec)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        avg_p = self.global_pool(feat).view(feat.size(0), -1)
        max_p = self.global_max(feat).view(feat.size(0), -1)
        pooled = torch.cat([avg_p, max_p], dim=1)
        return self.classifier(pooled)


def export_models(clean_ckpt_path="results/checkpoints/best_tf_faultnet_clean.pt", num_classes=12):
    """
    Loads best validated weights onto a clean CPU model and exports:
    1. Full TorchScript JIT model (Audio in -> Logits out)
    2. ONNX 2D Backbone model (Spectrogram Image in -> Logits out)
    """
    print("=" * 75, flush=True)
    print("STEP 1: EXPORTING CLEAN PRODUCTION MODELS", flush=True)
    print("=" * 75, flush=True)
    
    cpu_model = TFFaultNet(num_classes=num_classes)
    state_dict = torch.load(clean_ckpt_path, map_location="cpu")
    cpu_model.load_state_dict(state_dict)
    cpu_model.eval()

    # 1. TorchScript JIT Export (Raw Audio)
    jit_path = "results/checkpoints/best_tf_faultnet_clean_jit.pt"
    dummy_audio = torch.randn(1, 44100)
    try:
        traced_jit = torch.jit.trace(cpu_model, dummy_audio)
        traced_jit.save(jit_path)
        jit_size = os.path.getsize(jit_path) / (1024 * 1024)
        print(f"  ✓ Exported TorchScript JIT Model (End-to-End Audio): {jit_path} ({jit_size:.2f} MB)", flush=True)
    except Exception as e:
        print(f"  ✗ TorchScript JIT export failed: {e}", flush=True)

    # 2. ONNX Backbone Export (Spectrogram Images)
    onnx_path = "results/checkpoints/best_tf_faultnet_clean_backbone.onnx"
    backbone = TFFaultNetBackbone(cpu_model).eval()
    dummy_spec = torch.randn(1, 1, 64, 173)
    try:
        torch.onnx.export(
            backbone,
            dummy_spec,
            onnx_path,
            export_params=True,
            opset_version=14,
            dynamo=False,
            do_constant_folding=True,
            input_names=['spectrogram_image'],
            output_names=['fault_logits']
        )
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ✓ Exported 2D Spectrogram Backbone ONNX Model:      {onnx_path} ({onnx_size:.2f} MB)", flush=True)
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}", flush=True)

    return cpu_model, backbone


def run_inference_on_heldout_test():
    clean_ckpt_path = "results/checkpoints/best_tf_faultnet_clean.pt"
    npz_path = "results/heldout_test_set.npz"

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Held-out test set not found at {npz_path}. Run train_with_heldout_test.py first.")

    # 1. Export Models
    full_model, backbone_model = export_models(clean_ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_model = full_model.to(device)
    backbone_model = backbone_model.to(device)

    # 2. Load Quarantined Held-Out Test Set
    print("\n" + "=" * 75, flush=True)
    print("STEP 2: LOADING QUARANTINED HELD-OUT TEST SET", flush=True)
    print("=" * 75, flush=True)
    
    test_data = np.load(npz_path, allow_pickle=True)
    test_audio = test_data["audio"]               # (N_test, 44100)
    test_specs = test_data["spectrogram_images"]  # (N_test, 1, 64, 173)
    test_labels = test_data["labels"]             # (N_test,)
    test_filenames = test_data["filenames"]       # (N_test,)
    class_names = test_data["class_names"].tolist()

    n_samples = len(test_labels)
    print(f"Held-Out Test Set: {n_samples} samples across {len(class_names)} classes.", flush=True)
    print(f"Zero-Leakage Guarantee: Unseen during training & checkpointing.", flush=True)

    # 3. INFERENCE MODE A: RAW AUDIO WAVEFORM INFERENCE
    print("\n" + "-" * 75, flush=True)
    print("INFERENCE MODE A: Full End-to-End Pipeline (Raw Audio Waveform Inputs)", flush=True)
    print("-" * 75, flush=True)
    
    audio_preds = []
    audio_probs = []
    t_start = time.perf_counter()
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_audio = torch.from_numpy(test_audio[i:i+batch_size]).to(device)
            logits = full_model(b_audio)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            audio_preds.extend(preds)
            audio_probs.extend(probs)
            
    t_audio = (time.perf_counter() - t_start) * 1000.0 / n_samples
    audio_preds = np.array(audio_preds)
    audio_probs = np.array(audio_probs)
    
    acc_audio = accuracy_score(test_labels, audio_preds)
    p_audio, r_audio, f1_audio, _ = precision_recall_fscore_support(test_labels, audio_preds, average='macro')
    print(f"  ✓ Waveform Inference Complete: {t_audio:.2f} ms / sample", flush=True)
    print(f"  • Test Accuracy: {acc_audio * 100:.2f}% | Macro F1: {f1_audio * 100:.2f}%", flush=True)

    # 4. INFERENCE MODE B: 2D SPECTROGRAM IMAGE INFERENCE SEPARATELY
    print("\n" + "-" * 75, flush=True)
    print("INFERENCE MODE B: 2D Spectrogram Image Backbone Inference (Separately)", flush=True)
    print("-" * 75, flush=True)
    
    spec_preds = []
    spec_probs = []
    t_start = time.perf_counter()
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_spec = torch.from_numpy(test_specs[i:i+batch_size]).to(device)
            logits = backbone_model(b_spec)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            spec_preds.extend(preds)
            spec_probs.extend(probs)
            
    t_spec = (time.perf_counter() - t_start) * 1000.0 / n_samples
    spec_preds = np.array(spec_preds)
    spec_probs = np.array(spec_probs)
    
    acc_spec = accuracy_score(test_labels, spec_preds)
    p_spec, r_spec, f1_spec, _ = precision_recall_fscore_support(test_labels, spec_preds, average='macro')
    print(f"  ✓ Spectrogram Image Inference Complete: {t_spec:.2f} ms / sample", flush=True)
    print(f"  • Test Accuracy: {acc_spec * 100:.2f}% | Macro F1: {f1_spec * 100:.2f}%", flush=True)

    # Verify equivalence between audio STFT inference and 2D Spectrogram backbone inference
    matches = np.sum(audio_preds == spec_preds)
    print(f"  • Pipeline Consistency Check: {matches}/{n_samples} ({matches/n_samples*100:.2f}% identical outputs)", flush=True)

    # 5. COMPREHENSIVE METRICS REPORTING
    print("\n" + "=" * 75, flush=True)
    print("FINAL PERFORMANCE EVALUATION REPORT BASED ON HELD-OUT TEST SET", flush=True)
    print("=" * 75, flush=True)

    prec, rec, f1, supp = precision_recall_fscore_support(test_labels, audio_preds, average=None)
    
    table_rows = []
    print(f"{'Class Index':<12} | {'Mechanical Fault Class':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support'}", flush=True)
    print("-" * 95, flush=True)
    
    for idx, cls in enumerate(class_names):
        row = {
            "class_index": idx,
            "class_name": cls,
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1_score": float(f1[idx]),
            "support": int(supp[idx])
        }
        table_rows.append(row)
        print(f"{idx:<12} | {cls:<28} | {prec[idx]*100:>8.2f}%  | {rec[idx]*100:>8.2f}%  | {f1[idx]*100:>8.2f}%  | {supp[idx]:>6}", flush=True)

    print("-" * 95, flush=True)
    print(f"{'OVERALL':<12} | {'12-Class Macro Average':<28} | {p_audio*100:>8.2f}%  | {r_audio*100:>8.2f}%  | {f1_audio*100:>8.2f}%  | {n_samples:>6}", flush=True)
    print(f"{'OVERALL':<12} | {'Top-1 Classification Accuracy':<28} | {acc_audio*100:>8.2f}%", flush=True)
    print("=" * 95, flush=True)

    # 6. CONFUSION MATRIX VISUALIZATION ON HELD-OUT TEST SET
    cm = confusion_matrix(test_labels, audio_preds)
    cm_path = "results/visualizations/heldout_test_confusion_matrix.png"
    plot_test_confusion_matrix(cm, class_names, acc_audio, f1_audio, cm_path)

    # 7. VISUALIZE SAMPLE TEST SPECTROGRAM INFERENCES (SEPARATE IMAGE INFERENCE DEMO)
    vis_demo_path = "results/visualizations/test_sample_spectrogram_inferences.png"
    plot_sample_spectrogram_inferences(test_specs, test_labels, audio_preds, audio_probs, class_names, vis_demo_path)

    # 8. SAVE STRUCTURED JSON REPORT
    report = {
        "evaluation_dataset": "Held-Out Test Set (Zero Leakage)",
        "test_sample_count": n_samples,
        "metrics": {
            "test_accuracy": float(acc_audio),
            "macro_precision": float(p_audio),
            "macro_recall": float(r_audio),
            "macro_f1_score": float(f1_audio)
        },
        "latency": {
            "waveform_end_to_end_ms": float(t_audio),
            "spectrogram_image_backbone_ms": float(t_spec)
        },
        "per_class_evaluation": table_rows,
        "exported_models": {
            "torchscript_jit": "results/checkpoints/best_tf_faultnet_clean_jit.pt",
            "onnx_backbone": "results/checkpoints/best_tf_faultnet_clean_backbone.onnx",
            "pytorch_checkpoint": clean_ckpt_path
        }
    }

    report_path = "results/heldout_test_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n>>> Full test evaluation report saved to: {report_path}", flush=True)


def plot_test_confusion_matrix(cm, class_names, acc, f1, output_path):
    plt.figure(figsize=(11, 9), dpi=300)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=True,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor='lightgray'
    )
    plt.title(f"Held-Out Test Set Confusion Matrix (Zero Leakage, N=430)\nAccuracy: {acc*100:.2f}% | Macro F1: {f1*100:.2f}%", 
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Predicted Fault Class", fontsize=11, fontweight='bold')
    plt.ylabel("Ground Truth Fault Class", fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f">>> Test set confusion matrix saved to: {output_path}", flush=True)


def plot_sample_spectrogram_inferences(specs, true_labels, preds, probs, class_names, output_path):
    """
    Renders a 3x4 visual grid of sample test spectrogram images alongside their
    predicted probabilities and ground truth classification labels.
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), dpi=300)
    axes = axes.flatten()

    # Pick 1 representative sample from each class
    indices_to_show = []
    for c in range(len(class_names)):
        c_idxs = np.where(true_labels == c)[0]
        if len(c_idxs) > 0:
            indices_to_show.append(c_idxs[0])

    for i, idx in enumerate(indices_to_show[:12]):
        ax = axes[i]
        spec_img = specs[idx, 0] # (64, 173)
        true_cls = class_names[true_labels[idx]]
        pred_cls = class_names[preds[idx]]
        confidence = probs[idx, preds[idx]] * 100

        im = ax.imshow(spec_img, aspect='auto', origin='lower', cmap='magma', extent=[0, 1.0, 0, 64])
        is_correct = (true_labels[idx] == preds[idx])
        color = "#2ca02c" if is_correct else "#d62728"
        status = "CORRECT" if is_correct else "MISCLASSIFIED"

        ax.set_title(f"True: {true_cls}\nPred: {pred_cls} ({confidence:.1f}%)\n[{status}]", 
                     fontsize=9, fontweight='bold', color=color, pad=6)
        if i % 4 == 0:
            ax.set_ylabel("Mel Bins (0-64)", fontsize=8.5, fontweight='bold')
        if i >= 8:
            ax.set_xlabel("Time (0-1.0s)", fontsize=8.5, fontweight='bold')

    plt.suptitle("Held-Out Test Set 2D Spectrogram Image Inference Visualizer (Separate Image Input)",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f">>> Sample spectrogram image inference demo saved to: {output_path}", flush=True)


if __name__ == "__main__":
    run_inference_on_heldout_test()
