import os
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb

from src.dataset import preload_all_audio, MemoryFaultDataset
from src.audio_features import extract_signal_statistical_features
from src.models import TFFaultNet, WaveformCNN1D, AudioBiGRU

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def gather_dataset(data_dir="Base_de_Dados"):
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    filepaths = []
    labels = []
    class_names = subdirs
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    for cls in class_names:
        wavs = sorted(glob.glob(os.path.join(data_dir, cls, "*.wav")))
        for w in wavs:
            filepaths.append(w)
            labels.append(class_to_idx[cls])
            
    return np.array(filepaths), np.array(labels), class_names

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for audio, labels in loader:
        audio, labels = audio.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for audio, labels in loader:
            audio, labels = audio.to(device), labels.to(device)
            outputs = model(audio)
            loss = criterion(outputs, labels)
                
            total_loss += loss.item() * len(labels)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    return total_loss / len(all_targets), acc, macro_f1, all_preds, all_targets

def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('True Label', fontsize=11, fontweight='bold')
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_benchmark(benchmark_df, out_path):
    plt.figure(figsize=(10, 6), dpi=150)
    x = np.arange(len(benchmark_df))
    width = 0.35
    
    plt.bar(x - width/2, benchmark_df['Accuracy'] * 100, width, label='Accuracy (%)', color='#2563eb')
    plt.bar(x + width/2, benchmark_df['Macro F1'] * 100, width, label='Macro F1 (%)', color='#10b981')
    
    plt.ylabel('Score (%)', fontsize=11, fontweight='bold')
    plt.title('Acoustic Fault Diagnosis - Model Benchmark (5-Fold CV)', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, benchmark_df['Model'], fontsize=10, fontweight='bold')
    plt.ylim(85, 102)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    for i, v in enumerate(benchmark_df['Accuracy']):
        plt.text(i - width/2, v * 100 + 0.3, f"{v*100:.2f}%", ha='center', fontsize=9, fontweight='bold')
    for i, v in enumerate(benchmark_df['Macro F1']):
        plt.text(i + width/2, v * 100 + 0.3, f"{v*100:.2f}%", ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Acoustic Fault Diagnosis Model Pipeline")
    parser.add_argument("--data-dir", type=str, default="Base_de_Dados")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70, flush=True)
    print("   Acoustic Fault Diagnosis Pipeline - Deep Learning & ML", flush=True)
    print(f"   Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 70, flush=True)
    
    args.results_dir = os.path.abspath(args.results_dir)
    vis_dir = os.path.join(args.results_dir, "visualizations")
    ckpt_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    filepaths, labels, class_names = gather_dataset(args.data_dir)
    num_classes = len(class_names)
    print(f"Discovered {len(filepaths)} audio samples across {num_classes} classes.", flush=True)
    
    # Preload all raw audio in RAM
    print("\n[1/4] Preloading all 2,148 audio files into RAM...", flush=True)
    t0 = time.time()
    audio_matrix = preload_all_audio(filepaths, target_len=44100)
    print(f"  Preloaded {audio_matrix.shape[0]} signals ({audio_matrix.nbytes / 1e6:.1f} MB) in {time.time() - t0:.2f}s", flush=True)
    
    # Extract Tabular Statistical Features for XGBoost
    print("\n[2/4] Extracting acoustic statistical features for XGBoost baseline...", flush=True)
    t0 = time.time()
    tab_features = [extract_signal_statistical_features(audio_matrix[i], sr=44100) for i in range(len(audio_matrix))]
    X_tab = np.array(tab_features, dtype=np.float32)
    print(f"  Extracted {X_tab.shape[1]} statistical features in {time.time() - t0:.2f}s", flush=True)
    
    # 5-Fold Stratified Split
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    
    models_to_evaluate = ["TF-FaultNet (Custom)", "WaveformCNN1D", "AudioBiGRU", "XGBoost"]
    fold_results = {m: {"acc": [], "macro_f1": []} for m in models_to_evaluate}
    
    best_overall_acc = 0.0
    all_fold_preds = []
    all_fold_targets = []
    
    print(f"\n[3/4] Starting {args.n_splits}-Fold Stratified Cross Validation...", flush=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(filepaths, labels), 1):
        print(f"\n--- Fold {fold}/{args.n_splits} ---", flush=True)
        
        # 1. XGBoost
        xgb_clf = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric='mlogloss',
            random_state=args.seed,
            n_jobs=-1
        )
        xgb_clf.fit(X_tab[train_idx], labels[train_idx])
        xgb_preds = xgb_clf.predict(X_tab[val_idx])
        xgb_acc = accuracy_score(labels[val_idx], xgb_preds)
        xgb_f1 = f1_score(labels[val_idx], xgb_preds, average='macro')
        fold_results["XGBoost"]["acc"].append(xgb_acc)
        fold_results["XGBoost"]["macro_f1"].append(xgb_f1)
        print(f"  [XGBoost]        Acc: {xgb_acc * 100:.2f}% | Macro F1: {xgb_f1 * 100:.2f}%", flush=True)
        
        # DataLoaders for PyTorch Models (from in-memory arrays)
        train_ds = MemoryFaultDataset(audio_matrix[train_idx], labels[train_idx], augment=True)
        val_ds = MemoryFaultDataset(audio_matrix[val_idx], labels[val_idx], augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        
        # 2. WaveformCNN1D
        cnn1d = WaveformCNN1D(num_classes=num_classes).to(device)
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)
        opt_cnn = optim.AdamW(cnn1d.parameters(), lr=args.lr, weight_decay=1e-4)
        best_val_f1 = 0.0
        best_val_acc = 0.0
        
        for ep in range(args.epochs):
            train_epoch(cnn1d, train_loader, crit, opt_cnn, device)
            _, v_acc, v_f1, _, _ = eval_epoch(cnn1d, val_loader, crit, device)
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_val_acc = v_acc
        fold_results["WaveformCNN1D"]["acc"].append(best_val_acc)
        fold_results["WaveformCNN1D"]["macro_f1"].append(best_val_f1)
        print(f"  [WaveformCNN1D]  Acc: {best_val_acc * 100:.2f}% | Macro F1: {best_val_f1 * 100:.2f}%", flush=True)
        
        # 3. AudioBiGRU
        gru_model = AudioBiGRU(num_classes=num_classes).to(device)
        opt_gru = optim.AdamW(gru_model.parameters(), lr=args.lr, weight_decay=1e-4)
        best_val_f1 = 0.0
        best_val_acc = 0.0
        
        for ep in range(args.epochs):
            train_epoch(gru_model, train_loader, crit, opt_gru, device)
            _, v_acc, v_f1, _, _ = eval_epoch(gru_model, val_loader, crit, device)
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_val_acc = v_acc
        fold_results["AudioBiGRU"]["acc"].append(best_val_acc)
        fold_results["AudioBiGRU"]["macro_f1"].append(best_val_f1)
        print(f"  [AudioBiGRU]     Acc: {best_val_acc * 100:.2f}% | Macro F1: {best_val_f1 * 100:.2f}%", flush=True)
        
        # 4. Custom TF-FaultNet (Main Model)
        tf_model = TFFaultNet(num_classes=num_classes).to(device)
        opt_tf = optim.AdamW(tf_model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched_tf = optim.lr_scheduler.CosineAnnealingLR(opt_tf, T_max=args.epochs, eta_min=1e-5)
        best_val_f1 = 0.0
        best_val_acc = 0.0
        best_fold_preds = None
        best_fold_targets = None
        
        for ep in range(args.epochs):
            train_loss, train_acc = train_epoch(tf_model, train_loader, crit, opt_tf, device)
            val_loss, v_acc, v_f1, preds, targets = eval_epoch(tf_model, val_loader, crit, device)
            sched_tf.step()
            
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_val_acc = v_acc
                best_fold_preds = preds
                best_fold_targets = targets
                if best_val_acc > best_overall_acc:
                    best_overall_acc = best_val_acc
                    torch.save(tf_model.state_dict(), os.path.join(ckpt_dir, "best_tf_faultnet.pt"))
                    
        fold_results["TF-FaultNet (Custom)"]["acc"].append(best_val_acc)
        fold_results["TF-FaultNet (Custom)"]["macro_f1"].append(best_val_f1)
        all_fold_preds.extend(best_fold_preds)
        all_fold_targets.extend(best_fold_targets)
        print(f"  [TF-FaultNet]    Acc: {best_val_acc * 100:.2f}% | Macro F1: {best_val_f1 * 100:.2f}%", flush=True)
        
    print("\n" + "=" * 70, flush=True)
    print("Cross Validation Results Summary Across 5 Folds:", flush=True)
    print("=" * 70, flush=True)
    summary_rows = []
    for model_name in models_to_evaluate:
        mean_acc = np.mean(fold_results[model_name]["acc"])
        std_acc = np.std(fold_results[model_name]["acc"])
        mean_f1 = np.mean(fold_results[model_name]["macro_f1"])
        std_f1 = np.std(fold_results[model_name]["macro_f1"])
        summary_rows.append({
            "Model": model_name,
            "Accuracy": mean_acc,
            "Accuracy_std": std_acc,
            "Macro F1": mean_f1,
            "Macro_F1_std": std_f1
        })
        print(f"  {model_name:22s} -> Acc: {mean_acc * 100:.2f}% +- {std_acc * 100:.2f}% | Macro F1: {mean_f1 * 100:.2f}% +- {std_f1 * 100:.2f}%", flush=True)
        
    benchmark_df = pd.DataFrame(summary_rows)
    benchmark_df.to_csv(os.path.join(args.results_dir, "benchmark_summary.csv"), index=False)
    
    # Save Plots
    print("\n[4/4] Generating visual diagnostics and plots...", flush=True)
    plot_benchmark(benchmark_df, os.path.join(vis_dir, "model_benchmark.png"))
    
    # Global Confusion Matrix for TF-FaultNet
    cm = confusion_matrix(all_fold_targets, all_fold_preds)
    plot_confusion_matrix(cm, class_names, os.path.join(vis_dir, "confusion_matrix.png"),
                          title="TF-FaultNet 5-Fold Cross Validation Confusion Matrix")
    
    # Per-Class Precision, Recall, F1 for TF-FaultNet
    prec, rec, f1, supp = precision_recall_fscore_support(all_fold_targets, all_fold_preds, zero_division=0)
    class_report_df = pd.DataFrame({
        "Class": class_names,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Support": supp
    })
    class_report_df.to_csv(os.path.join(args.results_dir, "class_report.csv"), index=False)
    print("\nPer-Class Detailed Report for TF-FaultNet:", flush=True)
    print(class_report_df.to_string(index=False), flush=True)
    
    # Export summary json
    summary_json = {
        "device": str(device),
        "total_samples": len(filepaths),
        "num_classes": num_classes,
        "benchmark": summary_rows,
        "per_class": class_report_df.to_dict(orient="records")
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)
        
    print("\n[SUCCESS] Pipeline completed successfully! Outputs saved to:", args.results_dir, flush=True)

if __name__ == "__main__":
    main()
