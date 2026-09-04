import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import time
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
from src.models import TFFaultNet, WaveformCNN1D, AudioBiGRU
from train import gather_dataset, train_epoch

def add_noise_to_signal(audio, snr_db):
    """
    Injects additive white Gaussian noise to match a specific SNR in dB.
    audio: numpy array or torch tensor of shape (N, T) or (T,)
    """
    if snr_db is None:
        return audio
    
    # Calculate signal power
    p_signal = np.mean(audio ** 2, axis=-1, keepdims=True)
    # Target noise power
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    # Generate Gaussian noise
    noise = np.random.normal(0, 1, size=audio.shape).astype(np.float32)
    p_current = np.mean(noise ** 2, axis=-1, keepdims=True)
    scaled_noise = noise * np.sqrt(p_noise / (p_current + 1e-12))
    
    noisy_audio = audio + scaled_noise
    return noisy_audio.astype(np.float32)


def evaluate_under_snr(model, audio_test, labels_test, snr_db, device, batch_size=64):
    model.eval()
    noisy_audio = add_noise_to_signal(audio_test, snr_db)
    dataset = MemoryFaultDataset(noisy_audio, labels_test, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return acc, f1


def main():
    print("=" * 75)
    print("STARTING EMPIRICAL NOISE ROBUSTNESS & ENVIRONMENTAL SNR BENCHMARK")
    print("=" * 75)
    
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing benchmark on computing device: {device}")
    
    # 1. Load Dataset
    data_dir = "Base_de_Dados"
    filepaths, labels, class_names = gather_dataset(data_dir)
    print(f"Loaded {len(filepaths)} audio samples across {len(class_names)} classes.")
    
    print("Preloading audio waveforms into memory...")
    audio_matrix = preload_all_audio(filepaths, target_len=44100)
    
    # Stratified 80/20 train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(sss.split(audio_matrix, labels))
    
    x_train, y_train = audio_matrix[train_idx], labels[train_idx]
    x_test, y_test = audio_matrix[test_idx], labels[test_idx]
    print(f"Dataset split: Train = {len(x_train)} samples, Test = {len(x_test)} samples.")
    
    # 2. Setup Models
    num_classes = len(class_names)
    
    # A. TF-FaultNet (Load trained weights)
    tf_net = TFFaultNet(num_classes=num_classes).to(device)
    ckpt_path = "results/checkpoints/best_tf_faultnet.pt"
    if os.path.exists(ckpt_path):
        tf_net.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded trained TF-FaultNet weights from {ckpt_path}")
    else:
        print("Warning: best_tf_faultnet.pt not found, training TF-FaultNet...")
    
    # B. Train Baseline WaveformCNN1D on train set
    print("\nTraining baseline WaveformCNN1D on identical train split (15 epochs)...")
    cnn1d = WaveformCNN1D(num_classes=num_classes).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt_cnn = optim.AdamW(cnn1d.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(MemoryFaultDataset(x_train, y_train, augment=True), batch_size=64, shuffle=True)
    
    for ep in range(15):
        train_epoch(cnn1d, train_loader, crit, opt_cnn, device)
    print("  WaveformCNN1D training complete.")

    # C. Train Baseline AudioBiGRU on train set
    print("\nTraining baseline AudioBiGRU on identical train split (15 epochs)...")
    bigru = AudioBiGRU(num_classes=num_classes).to(device)
    opt_gru = optim.AdamW(bigru.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for ep in range(15):
        train_epoch(bigru, train_loader, crit, opt_gru, device)
    print("  AudioBiGRU training complete.")

    # 3. Benchmark Across SNR Levels
    snr_levels = [None, 20, 15, 10, 5, 0, -5]
    snr_labels = ["Clean (Inf dB)", "+20 dB (Mild)", "+15 dB (Moderate)", "+10 dB (High Noise)", 
                  "+5 dB (Severe)", "0 dB (Equal Noise)", "-5 dB (Sub-Noise Regime)"]
    
    models = {
        "TF-FaultNet (Proposed)": tf_net,
        "WaveformCNN1D": cnn1d,
        "AudioBiGRU": bigru
    }
    
    results = []
    
    print("\n" + "=" * 75)
    print(f"{'Model':<24} | {'SNR Level':<20} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("=" * 75)
    
    for snr, snr_label in zip(snr_levels, snr_labels):
        snr_val_numeric = 999 if snr is None else snr
        for model_name, model in models.items():
            acc, f1 = evaluate_under_snr(model, x_test, y_test, snr, device)
            results.append({
                "Model": model_name,
                "SNR_dB": snr_val_numeric,
                "SNR_Label": snr_label,
                "Accuracy": acc,
                "Macro_F1": f1
            })
            print(f"{model_name:<24} | {snr_label:<20} | {acc * 100:>8.2f}% | {f1 * 100:>8.2f}%")
        print("-" * 75)
        
    df_results = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/noise_robustness_benchmark.csv", index=False)
    print("\n>>> Results successfully saved to results/noise_robustness_benchmark.csv")
    
    # 4. Generate Publication-Quality Visualization
    os.makedirs("results/visualizations", exist_ok=True)
    plot_snr_curves(df_results, "results/visualizations/snr_robustness_curves.png")
    

def plot_snr_curves(df, output_path):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Mapping for x-axis ordering
    snr_order = [999, 20, 15, 10, 5, 0, -5]
    x_positions = np.arange(len(snr_order))
    x_labels = ["Clean\n(Inf dB)", "+20 dB\n(Mild)", "+15 dB\n(Moderate)", "+10 dB\n(High)", 
                "+5 dB\n(Severe)", "0 dB\n(Critical)", "-5 dB\n(Sub-Noise)"]
    
    color_map = {
        "TF-FaultNet (Proposed)": "#1f77b4",  # Bold Blue
        "WaveformCNN1D": "#ff7f0e",           # Vivid Orange
        "AudioBiGRU": "#2ca02c"               # Emerald Green
    }
    marker_map = {
        "TF-FaultNet (Proposed)": "o",
        "WaveformCNN1D": "s",
        "AudioBiGRU": "^"
    }
    
    # Plot 1: Accuracy Degradation
    for model_name, group in df.groupby("Model"):
        # Sort by snr_order
        group_sorted = group.set_index("SNR_dB").reindex(snr_order).reset_index()
        acc_values = group_sorted["Accuracy"].values * 100
        
        lw = 2.8 if "Proposed" in model_name else 1.8
        ms = 8 if "Proposed" in model_name else 7
        ax1.plot(x_positions, acc_values, label=model_name,
                 color=color_map.get(model_name, "black"),
                 marker=marker_map.get(model_name, "o"),
                 linewidth=lw, markersize=ms)
        
        # Annotate values for proposed model
        if "Proposed" in model_name:
            for x_idx, acc_val in zip(x_positions, acc_values):
                ax1.annotate(f"{acc_val:.1f}%", 
                             (x_idx, acc_val),
                             textcoords="offset points", 
                             xytext=(0, 9), 
                             ha='center', fontsize=8.5, fontweight='bold',
                             color='#1f77b4')

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, fontsize=9.5)
    ax1.set_ylabel("Classification Accuracy (%)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Injected Environmental Noise Level (Signal-to-Noise Ratio)", fontsize=11, fontweight='bold')
    ax1.set_title("(a) Classification Accuracy vs. Industrial Noise (SNR)", fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylim(40, 105)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9.5)

    # Plot 2: Macro F1-Score Degradation
    for model_name, group in df.groupby("Model"):
        group_sorted = group.set_index("SNR_dB").reindex(snr_order).reset_index()
        f1_values = group_sorted["Macro_F1"].values * 100
        
        lw = 2.8 if "Proposed" in model_name else 1.8
        ms = 8 if "Proposed" in model_name else 7
        ax2.plot(x_positions, f1_values, label=model_name,
                 color=color_map.get(model_name, "black"),
                 marker=marker_map.get(model_name, "o"),
                 linewidth=lw, markersize=ms)
        
        if "Proposed" in model_name:
            for x_idx, f1_val in zip(x_positions, f1_values):
                ax2.annotate(f"{f1_val:.1f}%", 
                             (x_idx, f1_val),
                             textcoords="offset points", 
                             xytext=(0, 9), 
                             ha='center', fontsize=8.5, fontweight='bold',
                             color='#1f77b4')

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, fontsize=9.5)
    ax2.set_ylabel("Macro F1-Score (%)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Injected Environmental Noise Level (Signal-to-Noise Ratio)", fontsize=11, fontweight='bold')
    ax2.set_title("(b) Macro F1-Score vs. Industrial Noise (SNR)", fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylim(40, 105)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f">>> Publication figure generated at: {output_path}")


if __name__ == "__main__":
    main()
