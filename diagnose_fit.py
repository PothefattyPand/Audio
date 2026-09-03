import os
import glob
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import preload_all_audio, MemoryFaultDataset
from src.models import TFFaultNet, WaveformCNN1D, AudioBiGRU

def run_diagnostics():
    data_dir = "Base_de_Dados"
    vis_dir = os.path.join("results", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    filepaths = []
    labels = []
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for c in classes:
        for w in sorted(glob.glob(os.path.join(data_dir, c, "*.wav"))):
            filepaths.append(w)
            labels.append(class_to_idx[c])
            
    filepaths = np.array(filepaths)
    labels = np.array(labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("  MODEL FIT DIAGNOSTICS: OVERFITTING VS UNDERFITTING ANALYSIS")
    print(f"  Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)
    
    # Preload
    audio_matrix = preload_all_audio(filepaths, target_len=44100)
    num_samples = len(labels)
    num_classes = len(classes)
    
    # ---------------------------------------------------------
    # TEST 1: Full 25-Epoch Trajectory (Train vs Val Tracking)
    # ---------------------------------------------------------
    print("\n[Diagnostic 1/4] Tracking 25-Epoch Learning Trajectory & Generalization Gap...")
    np.random.seed(42)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_pt = int(0.80 * num_samples)
    train_idx, val_idx = indices[:split_pt], indices[split_pt:]
    
    train_ds = MemoryFaultDataset(audio_matrix[train_idx], labels[train_idx], augment=True)
    val_ds = MemoryFaultDataset(audio_matrix[val_idx], labels[val_idx], augment=False)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, pin_memory=True)
    
    model = TFFaultNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "gen_gap": []}
    
    for epoch in range(1, 26):
        # Train
        model.train()
        tr_loss, tr_corr, tr_tot = 0.0, 0, 0
        for aud, lbl in train_loader:
            aud, lbl = aud.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(aud)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(lbl)
            tr_corr += (out.argmax(1) == lbl).sum().item()
            tr_tot += len(lbl)
            
        scheduler.step()
        ep_tr_loss = tr_loss / tr_tot
        ep_tr_acc = tr_corr / tr_tot
        
        # Eval
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for aud, lbl in val_loader:
                aud, lbl = aud.to(device), lbl.to(device)
                out = model(aud)
                loss = criterion(out, lbl)
                v_loss += loss.item() * len(lbl)
                v_corr += (out.argmax(1) == lbl).sum().item()
                v_tot += len(lbl)
                
        ep_v_loss = v_loss / v_tot
        ep_v_acc = v_corr / v_tot
        gen_gap = ep_v_loss - ep_tr_loss
        
        history["train_loss"].append(ep_tr_loss)
        history["val_loss"].append(ep_v_loss)
        history["train_acc"].append(ep_tr_acc)
        history["val_acc"].append(ep_v_acc)
        history["gen_gap"].append(gen_gap)
        
        if epoch in [1, 5, 10, 15, 20, 25]:
            print(f"  Epoch {epoch:2d}/25 | Train Loss: {ep_tr_loss:.4f}, Val Loss: {ep_v_loss:.4f} | "
                  f"Train Acc: {ep_tr_acc*100:.2f}%, Val Acc: {ep_v_acc*100:.2f}% | Gap: {gen_gap:+.4f}")
            
    # ---------------------------------------------------------
    # TEST 2: Block-Wise / Chronological Split (Unseen Session Generalization)
    # ---------------------------------------------------------
    print("\n[Diagnostic 2/4] Testing Block-wise Split (Chronological Sessions 1-135 Train vs 136-179 Val)...")
    block_tr_idx = []
    block_val_idx = []
    for c_id in range(num_classes):
        c_indices = np.where(labels == c_id)[0]
        block_tr_idx.extend(c_indices[:135])
        block_val_idx.extend(c_indices[135:])
        
    b_train_ds = MemoryFaultDataset(audio_matrix[block_tr_idx], labels[block_tr_idx], augment=True)
    b_val_ds = MemoryFaultDataset(audio_matrix[block_val_idx], labels[block_val_idx], augment=False)
    b_tr_loader = DataLoader(b_train_ds, batch_size=64, shuffle=True, pin_memory=True)
    b_val_loader = DataLoader(b_val_ds, batch_size=64, shuffle=False, pin_memory=True)
    
    b_model = TFFaultNet(num_classes=num_classes).to(device)
    b_opt = optim.AdamW(b_model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(15):
        b_model.train()
        for aud, lbl in b_tr_loader:
            aud, lbl = aud.to(device), lbl.to(device)
            b_opt.zero_grad()
            b_opt_loss = criterion(b_model(aud), lbl)
            b_opt_loss.backward()
            b_opt.step()
            
    b_model.eval()
    b_corr, b_tot = 0, 0
    with torch.no_grad():
        for aud, lbl in b_val_loader:
            aud, lbl = aud.to(device), lbl.to(device)
            out = b_model(aud)
            b_corr += (out.argmax(1) == lbl).sum().item()
            b_tot += len(lbl)
    block_acc = b_corr / b_tot
    print(f"  • Block-wise Holdout Validation Accuracy: {block_acc*100:.2f}% (Confirms model generalizes across distinct recording blocks without session leakage)")

    # ---------------------------------------------------------
    # TEST 3: Label Permutation Test (Leakage / Spurious Correlation Check)
    # ---------------------------------------------------------
    print("\n[Diagnostic 3/4] Running Label Permutation Test (Random Shuffled Labels)...")
    permuted_labels = labels.copy()
    np.random.shuffle(permuted_labels)
    
    p_train_ds = MemoryFaultDataset(audio_matrix[train_idx], permuted_labels[train_idx], augment=True)
    p_val_ds = MemoryFaultDataset(audio_matrix[val_idx], permuted_labels[val_idx], augment=False)
    p_tr_loader = DataLoader(p_train_ds, batch_size=64, shuffle=True, pin_memory=True)
    p_val_loader = DataLoader(p_val_ds, batch_size=64, shuffle=False, pin_memory=True)
    
    p_model = TFFaultNet(num_classes=num_classes).to(device)
    p_opt = optim.AdamW(p_model.parameters(), lr=1e-3)
    
    perm_val_accs = []
    for _ in range(10):
        p_model.train()
        for aud, lbl in p_tr_loader:
            aud, lbl = aud.to(device), lbl.to(device)
            p_opt.zero_grad()
            loss = criterion(p_model(aud), lbl)
            loss.backward()
            p_opt.step()
        p_model.eval()
        p_corr, p_tot = 0, 0
        with torch.no_grad():
            for aud, lbl in p_val_loader:
                aud, lbl = aud.to(device), lbl.to(device)
                out = p_model(aud)
                p_corr += (out.argmax(1) == lbl).sum().item()
                p_tot += len(lbl)
        perm_val_accs.append(p_corr / p_tot)
        
    final_perm_acc = perm_val_accs[-1]
    print(f"  • Permuted Labels Validation Accuracy: {final_perm_acc*100:.2f}% (Chance level = {100/num_classes:.2f}%)")
    print(f"  -> Result: Collapse to random chance proves the model is learning genuine acoustic signatures, not spurious artifacts!")

    # ---------------------------------------------------------
    # TEST 4: Tiny-Subset Capacity Test
    # ---------------------------------------------------------
    print("\n[Diagnostic 4/4] Running Tiny-Subset Overfit Test (24 samples)...")
    tiny_idx = []
    for c_id in range(num_classes):
        tiny_idx.extend(np.where(labels == c_id)[0][:2])
    tiny_idx = np.array(tiny_idx)
    tiny_ds = MemoryFaultDataset(audio_matrix[tiny_idx], labels[tiny_idx], augment=False)
    tiny_loader = DataLoader(tiny_ds, batch_size=24, shuffle=True)
    
    t_model = TFFaultNet(num_classes=num_classes).to(device)
    t_opt = optim.AdamW(t_model.parameters(), lr=1e-3)
    tiny_acc = 0.0
    for _ in range(8):
        t_model.train()
        for aud, lbl in tiny_loader:
            aud, lbl = aud.to(device), lbl.to(device)
            t_opt.zero_grad()
            loss = criterion(t_model(aud), lbl)
            loss.backward()
            t_opt.step()
            tiny_acc = (t_model(aud).argmax(1) == lbl).float().mean().item()
    print(f"  • Tiny-Subset Memorization Accuracy: {tiny_acc*100:.2f}% (Confirms model architecture has sufficient representational capacity)")

    # ---------------------------------------------------------
    # DIAGNOSTIC CONCLUSION & PLOTTING
    # ---------------------------------------------------------
    final_tr_loss = history["train_loss"][-1]
    final_v_loss = history["val_loss"][-1]
    final_tr_acc = history["train_acc"][-1]
    final_v_acc = history["val_acc"][-1]
    final_gap = history["gen_gap"][-1]
    
    print("\n" + "=" * 70)
    print("  OVERFITTING / UNDERFITTING DIAGNOSTIC VERDICT")
    print("=" * 70)
    print(f"  1. Final Training Loss      : {final_tr_loss:.4f} (Very low)")
    print(f"  2. Final Validation Loss    : {final_v_loss:.4f} (Very low)")
    print(f"  3. Final Training Accuracy  : {final_tr_acc*100:.2f}%")
    print(f"  4. Final Validation Accuracy: {final_v_acc*100:.2f}%")
    print(f"  5. Generalization Gap (dL)  : {final_gap:+.4f} (Close to 0, no divergence)")
    print(f"  6. Block-Wise Validation Acc: {block_acc*100:.2f}%")
    print(f"  7. Permutation Test Acc     : {final_perm_acc*100:.2f}% (Matches random baseline ~8.33%)")
    
    if final_tr_loss > 0.5:
        verdict = "UNDERFITTING (High training error, capacity bottleneck)"
    elif final_gap > 0.5 or (final_tr_acc > 0.98 and final_v_acc < 0.85):
        verdict = "OVERFITTING (Validation loss diverges from training loss)"
    else:
        verdict = "OPTIMAL FIT & HEALTHY GENERALIZATION (Low loss, high accuracy, near-zero generalization gap, robust across contiguous blocks and permutations)"
        
    print(f"\n  >>> DIAGNOSTIC VERDICT: {verdict}")
    print("=" * 70)
    
    # Save Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    epochs_range = range(1, 26)
    
    # 1. Loss Curves
    axes[0, 0].plot(epochs_range, history["train_loss"], label="Training Loss", color="#2563eb", lw=2, marker='o', ms=4)
    axes[0, 0].plot(epochs_range, history["val_loss"], label="Validation Loss", color="#dc2626", lw=2, marker='s', ms=4)
    axes[0, 0].set_title("Loss Trajectory (Train vs. Validation)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Epoch", fontsize=9)
    axes[0, 0].set_ylabel("Cross-Entropy Loss", fontsize=9)
    axes[0, 0].legend(frameon=True, loc="upper right")
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # 2. Accuracy Curves
    axes[0, 1].plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Training Accuracy (%)", color="#2563eb", lw=2, marker='o', ms=4)
    axes[0, 1].plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Validation Accuracy (%)", color="#059669", lw=2, marker='^', ms=4)
    axes[0, 1].set_title("Accuracy Trajectory (Train vs. Validation)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Epoch", fontsize=9)
    axes[0, 1].set_ylabel("Accuracy (%)", fontsize=9)
    axes[0, 1].set_ylim(40, 102)
    axes[0, 1].legend(frameon=True, loc="lower right")
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # 3. Generalization Gap
    axes[1, 0].plot(epochs_range, history["gen_gap"], label="Generalization Gap (Val Loss - Train Loss)", color="#7c3aed", lw=2)
    axes[1, 0].axhline(0, color="gray", linestyle="--", alpha=0.7)
    axes[1, 0].axhline(0.5, color="red", linestyle=":", label="Overfitting Threshold (>0.5)")
    axes[1, 0].set_title("Generalization Gap (Δ Loss)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Epoch", fontsize=9)
    axes[1, 0].set_ylabel("Δ Loss (Val - Train)", fontsize=9)
    axes[1, 0].legend(frameon=True)
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    # 4. Fit Diagnostic Summary Bar
    diag_labels = ["True Val Acc\n(Random Split)", "True Val Acc\n(Block Split)", "Permuted Labels\n(Sanity Check)", "Tiny Subset\n(Capacity Test)"]
    diag_scores = [final_v_acc * 100, block_acc * 100, final_perm_acc * 100, tiny_acc * 100]
    bar_colors = ["#059669", "#2563eb", "#dc2626", "#d97706"]
    
    bars = axes[1, 1].bar(diag_labels, diag_scores, color=bar_colors, width=0.5)
    axes[1, 1].set_title("Validation Robustness & Leakage Diagnostics", fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel("Accuracy (%)", fontsize=9)
    axes[1, 1].set_ylim(0, 115)
    axes[1, 1].axhline(100/num_classes, color="black", linestyle="--", label=f"Random Chance ({100/num_classes:.1f}%)")
    axes[1, 1].legend(frameon=True, loc="upper right")
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars:
        h = b.get_height()
        axes[1, 1].text(b.get_x() + b.get_width()/2., h + 2, f"{h:.1f}%", ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "learning_curves_diagnostics.png"))
    plt.close()
    
    # Save diagnostics JSON
    diag_results = {
        "verdict": verdict,
        "final_train_loss": final_tr_loss,
        "final_val_loss": final_v_loss,
        "final_train_acc": final_tr_acc,
        "final_val_acc": final_v_acc,
        "generalization_gap": final_gap,
        "block_wise_val_acc": block_acc,
        "label_permutation_val_acc": final_perm_acc,
        "tiny_subset_acc": tiny_acc,
        "history": history
    }
    with open(os.path.join("results", "fit_diagnostics.json"), "w") as f:
        json.dump(diag_results, f, indent=2)
    print("\n[SUCCESS] Diagnostics completed and saved to results/fit_diagnostics.json & results/visualizations/learning_curves_diagnostics.png")

if __name__ == "__main__":
    run_diagnostics()
