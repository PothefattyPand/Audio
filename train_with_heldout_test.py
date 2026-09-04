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
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import preload_all_audio, MemoryFaultDataset
from src.audio_features import LogMelSpectrogramExtractor
from src.models import TFFaultNet
from train import gather_dataset, train_epoch, eval_epoch

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TFFaultNetBackbone(nn.Module):
    """Backbone without STFT for 2D spectrogram image inference and ONNX deployment."""
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

def main():
    print("=" * 75, flush=True)
    print("   ZERO-LEAKAGE CONVERGENCE TRAINING & MODEL EXPORT PIPELINE", flush=True)
    print("=" * 75, flush=True)

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({device_name})", flush=True)

    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)

    # 1. Dataset Loading
    data_dir = "Base_de_Dados"
    filepaths, labels, class_names = gather_dataset(data_dir)
    total_samples = len(filepaths)
    num_classes = len(class_names)
    print(f"Total dataset: {total_samples} samples across {num_classes} classes.", flush=True)

    print("Preloading all raw audio into RAM...", flush=True)
    audio_matrix = preload_all_audio(filepaths, target_len=44100)

    # 2. STRICT 3-WAY SPLIT: TRAIN (64%), VAL (16%), HELD-OUT TEST (20%)
    # Zero leakage guarantee: Held-out test set is partitioned FIRST and quarantined completely.
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_val_idx, test_idx = next(sss_test.split(audio_matrix, labels))

    # Split train_val into train (80%) and validation (20%)
    train_val_labels = labels[train_val_idx]
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_sub_idx, val_sub_idx = next(sss_val.split(audio_matrix[train_val_idx], train_val_labels))

    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    # Verify zero leakage mathematically
    set_train = set(train_idx)
    set_val = set(val_idx)
    set_test = set(test_idx)

    assert len(set_train.intersection(set_val)) == 0, "DATA LEAKAGE DETECTED: Train and Val overlap!"
    assert len(set_train.intersection(set_test)) == 0, "DATA LEAKAGE DETECTED: Train and Test overlap!"
    assert len(set_val.intersection(set_test)) == 0, "DATA LEAKAGE DETECTED: Val and Test overlap!"
    assert len(set_train) + len(set_val) + len(set_test) == total_samples, "Index mismatch!"

    print("\n" + "-" * 75, flush=True)
    print("DATASET PARTITIONING (STRICT ZERO-LEAKAGE AUDIT):", flush=True)
    print(f"  • Training Set:    {len(train_idx)} samples ({len(train_idx)/total_samples*100:.1f}%) -> Used for model backprop", flush=True)
    print(f"  • Validation Set:  {len(val_idx)} samples ({len(val_idx)/total_samples*100:.1f}%) -> Used for convergence & checkpointing", flush=True)
    print(f"  • Held-Out Test:   {len(test_idx)} samples ({len(test_idx)/total_samples*100:.1f}%) -> STRICTLY QUARANTINED (Untouched during training)", flush=True)
    print("-" * 75, flush=True)

    # Save exact held-out test indices and metadata for auditability
    test_info = {
        "total_test_samples": int(len(test_idx)),
        "class_distribution": {cls: int(np.sum(labels[test_idx] == idx)) for idx, cls in enumerate(class_names)},
        "test_filepaths": [filepaths[i] for i in test_idx],
        "test_indices": test_idx.tolist()
    }
    with open("results/heldout_test_indices.json", "w") as f:
        json.dump(test_info, f, indent=2)

    # Extract 2D Log-Mel spectrogram images for the test set so the user can inspect/infer separately
    spec_extractor = LogMelSpectrogramExtractor(sr=44100, n_mels=64).to(device)
    test_audio_tensor = torch.from_numpy(audio_matrix[test_idx]).to(device)
    with torch.no_grad():
        test_specs = spec_extractor(test_audio_tensor).cpu().numpy() # (N_test, 1, 64, 173)

    # Save the held-out test set separately into an NPZ archive
    np.savez_compressed(
        "results/heldout_test_set.npz",
        audio=audio_matrix[test_idx],
        spectrogram_images=test_specs,
        labels=labels[test_idx],
        filenames=filepaths[test_idx],
        class_names=np.array(class_names)
    )
    print(f"Quarantined held-out test set saved to: results/heldout_test_set.npz ({test_specs.shape[0]} spectrograms + waveforms)", flush=True)

    # 3. DataLoaders
    train_ds = MemoryFaultDataset(audio_matrix[train_idx], labels[train_idx], augment=True)
    val_ds = MemoryFaultDataset(audio_matrix[val_idx], labels[val_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, pin_memory=True)

    # 4. Model, Optimizer, Scheduler, Loss
    model = TFFaultNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 25
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("\n" + "=" * 75, flush=True)
    print(f"STARTING CONVERGENCE TRAINING ({epochs} EPOCHS)", flush=True)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Val Loss':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Val F1':<10} | {'Status'}", flush=True)
    print("=" * 75, flush=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
        "lr": []
    }

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    clean_ckpt_path = "results/checkpoints/best_tf_faultnet_clean.pt"

    for epoch in range(1, epochs + 1):
        # Training epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # Validation epoch
        val_loss, val_acc, val_f1, _, _ = eval_epoch(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        status_str = ""
        # Checkpoint based on best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), clean_ckpt_path)
            status_str = f"★ Best Model Saved (Val Loss: {val_loss:.4f})"

        print(f"{epoch:<6} | {train_loss:<11.4f} | {val_loss:<10.4f} | {train_acc*100:>8.2f}%  | {val_acc*100:>8.2f}%  | {val_f1*100:>8.2f}%  | {status_str}", flush=True)

    print("=" * 75, flush=True)
    print(f"Training Complete! Optimal Checkpoint at Epoch {best_epoch}: Val Loss = {best_val_loss:.4f}, Val Acc = {best_val_acc*100:.2f}%", flush=True)

    # Save training history to CSV
    df_history = pd.DataFrame(history)
    df_history.to_csv("results/training_convergence_history.csv", index=False)

    # 5. GENERATE THE MASTER LOSS & ACCURACY CONVERGENCE FIGURE (IN ONE MASTER FIGURE)
    plot_convergence_curves(df_history, best_epoch, "results/visualizations/train_val_loss_convergence.png")

    # 6. EXPORT BEST MODEL CHECKPOINTS (TORCHSCRIPT JIT & ONNX)
    print("\n" + "=" * 75, flush=True)
    print("EXPORTING BEST VALIDATED MODEL TO EDGE RUNTIMES", flush=True)
    print("=" * 75, flush=True)
    
    # Reload best checkpoint
    model.load_state_dict(torch.load(clean_ckpt_path, map_location="cpu"))
    model.eval()

    # A. TorchScript JIT Export (Full End-to-End model: Audio -> Class logits)
    jit_path = "results/checkpoints/best_tf_faultnet_clean_jit.pt"
    dummy_audio = torch.randn(1, 44100)
    try:
        traced_jit = torch.jit.trace(model, dummy_audio)
        traced_jit.save(jit_path)
        jit_size = os.path.getsize(jit_path) / (1024 * 1024)
        print(f"  ✓ Exported Full End-to-End TorchScript JIT model: {jit_path} ({jit_size:.2f} MB)", flush=True)
    except Exception as e:
        print(f"  ✗ TorchScript JIT export failed: {e}", flush=True)

    # B. ONNX Backbone Export (Spectrogram Image -> Class logits)
    onnx_path = "results/checkpoints/best_tf_faultnet_clean_backbone.onnx"
    backbone = TFFaultNetBackbone(model).eval()
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
        print(f"  ✓ Exported 2D Spectrogram Backbone ONNX model:    {onnx_path} ({onnx_size:.2f} MB)", flush=True)
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}", flush=True)

    print("\nTraining and export successfully completed! Proceeding to test set evaluation...", flush=True)


def plot_convergence_curves(df, best_epoch, output_path):
    """
    Plots the master convergence figure:
    Panel 1: BOTH Train Loss and Validation Loss curves in ONE plot.
    Panel 2: BOTH Train Accuracy and Validation Accuracy curves in ONE plot.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    epochs = df["epoch"].values
    train_loss = df["train_loss"].values
    val_loss = df["val_loss"].values
    train_acc = df["train_acc"].values * 100
    val_acc = df["val_acc"].values * 100

    # ------------------ PANEL 1: LOSS CURVES (BOTH IN ONE FIG) ------------------
    ax_loss.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", linewidth=2.5, marker="o", markersize=4)
    ax_loss.plot(epochs, val_loss, label="Validation Loss", color="#d62728", linewidth=2.5, marker="s", markersize=4)
    
    # Fill between curves to show generalization margin
    ax_loss.fill_between(epochs, train_loss, val_loss, color="gray", alpha=0.12, label="Generalization Margin")

    # Annotate optimal checkpoint
    best_loss_val = val_loss[best_epoch - 1]
    ax_loss.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_loss.scatter([best_epoch], [best_loss_val], color="#2ca02c", s=140, zorder=5, 
                    edgecolor="black", linewidth=1.5, label=f"Best Checkpoint (Epoch {best_epoch})")
    ax_loss.annotate(f"Optimal Checkpoint\nLoss: {best_loss_val:.4f}",
                     xy=(best_epoch, best_loss_val),
                     xytext=(best_epoch + 1.2, best_loss_val + 0.15),
                     arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
                     fontsize=9.5, fontweight="bold", color="#2ca02c",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", lw=1))

    ax_loss.set_title("(a) Training vs. Validation Loss Convergence (In One Plot)", fontsize=12, fontweight="bold", pad=12)
    ax_loss.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_loss.set_ylabel("Cross-Entropy Loss (Label Smoothing=0.05)", fontsize=11, fontweight="bold")
    ax_loss.set_xticks(np.arange(1, len(epochs) + 1, 2))
    ax_loss.set_ylim(0, max(max(train_loss), max(val_loss)) * 1.15)
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    ax_loss.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9.5)

    # ------------------ PANEL 2: ACCURACY CURVES ------------------
    ax_acc.plot(epochs, train_acc, label="Training Accuracy", color="#1f77b4", linewidth=2.5, marker="o", markersize=4)
    ax_acc.plot(epochs, val_acc, label="Validation Accuracy", color="#d62728", linewidth=2.5, marker="s", markersize=4)
    
    best_acc_val = val_acc[best_epoch - 1]
    ax_acc.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_acc.scatter([best_epoch], [best_acc_val], color="#2ca02c", s=140, zorder=5,
                   edgecolor="black", linewidth=1.5, label=f"Best Model Acc: {best_acc_val:.2f}%")
    ax_acc.annotate(f"Val Acc: {best_acc_val:.2f}%\nVal F1: {df['val_f1'].values[best_epoch-1]*100:.2f}%",
                    xy=(best_epoch, best_acc_val),
                    xytext=(best_epoch - 7.5, best_acc_val - 12.0),
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
                    fontsize=9.5, fontweight="bold", color="#2ca02c",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", lw=1))

    ax_acc.set_title("(b) Training vs. Validation Accuracy Trajectory", fontsize=12, fontweight="bold", pad=12)
    ax_acc.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_acc.set_ylabel("Classification Accuracy (%)", fontsize=11, fontweight="bold")
    ax_acc.set_xticks(np.arange(1, len(epochs) + 1, 2))
    ax_acc.set_ylim(40, 105)
    ax_acc.grid(True, linestyle="--", alpha=0.6)
    ax_acc.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9.5)

    plt.suptitle("TF-FaultNet Empirical Convergence Profiles (Zero-Leakage Partition)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Master convergence curves figure saved to: {output_path}", flush=True)

if __name__ == "__main__":
    main()
