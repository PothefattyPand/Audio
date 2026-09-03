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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.dataset import preload_all_audio, MemoryFaultDataset
from src.audio_features import LogMelSpectrogramExtractor

# -------------------------------------------------------------
# Ablation Model Variants
# -------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w

class AblationResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, use_se=True, use_res=True):
        super().__init__()
        self.use_se = use_se
        self.use_res = use_res
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if use_se:
            self.se = SqueezeExcitation(out_c)
        if use_res:
            if stride != 1 or in_c != out_c:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_c)
                )
            else:
                self.shortcut = nn.Identity()
                
    def forward(self, x):
        res = self.shortcut(x) if self.use_res else 0
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        out = F.relu(out + res if self.use_res else out, inplace=True)
        return out


class LogMelSpectrogramExtractorNoNorm(nn.Module):
    """Ablation: Log-Mel Spectrogram WITHOUT Instance Normalization."""
    def __init__(self, sr=44100, n_fft=1024, hop_length=256, n_mels=64):
        super().__init__()
        from src.audio_features import get_mel_filterbank
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        self.register_buffer("mel_filterbank", get_mel_filterbank(sr, n_fft, n_mels))
        
    def forward(self, x):
        stft = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
            window=self.window, return_complex=True, center=True, pad_mode='reflect'
        )
        power = torch.abs(stft) ** 2
        mel_spec = torch.matmul(self.mel_filterbank, power)
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-6))
        # Raw unnormalized log-mel
        return log_mel.unsqueeze(1)


class FlexibleFaultNet(nn.Module):
    """
    Configurable FaultNet for systematic ablation testing.
    """
    def __init__(
        self,
        num_classes=12,
        use_se=True,
        pooling_mode="dual",      # "dual" (Avg+Max), "avg_only", "max_only"
        use_instance_norm=True,
        use_residual=True
    ):
        super().__init__()
        self.pooling_mode = pooling_mode
        if use_instance_norm:
            self.spec_extractor = LogMelSpectrogramExtractor(sr=44100, n_mels=64)
        else:
            self.spec_extractor = LogMelSpectrogramExtractorNoNorm(sr=44100, n_mels=64)
            
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.layer1 = AblationResBlock(32, 64, stride=2, use_se=use_se, use_res=use_residual)
        self.layer2 = AblationResBlock(64, 128, stride=2, use_se=use_se, use_res=use_residual)
        self.layer3 = AblationResBlock(128, 256, stride=2, use_se=use_se, use_res=use_residual)
        
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max = nn.AdaptiveMaxPool2d((1, 1))
        
        feat_dim = 256 * 2 if pooling_mode == "dual" else 256
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        spec = self.spec_extractor(x)
        feat = self.init_conv(spec)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        
        if self.pooling_mode == "dual":
            avg_p = self.global_avg(feat).view(feat.size(0), -1)
            max_p = self.global_max(feat).view(feat.size(0), -1)
            pooled = torch.cat([avg_p, max_p], dim=1)
        elif self.pooling_mode == "avg_only":
            pooled = self.global_avg(feat).view(feat.size(0), -1)
        elif self.pooling_mode == "max_only":
            pooled = self.global_max(feat).view(feat.size(0), -1)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")
            
        return self.classifier(pooled)


def train_eval_model(model, train_loader, val_loader, epochs=15, lr=1e-3, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_acc = 0.0
    best_preds = None
    best_targets = None
    
    for ep in range(epochs):
        model.train()
        for aud, lbl in train_loader:
            aud, lbl = aud.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(aud)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # Eval
        model.eval()
        p_list, t_list = [], []
        with torch.no_grad():
            for aud, lbl in val_loader:
                aud = aud.to(device)
                out = model(aud)
                p_list.extend(out.argmax(1).cpu().numpy())
                t_list.extend(lbl.numpy())
                
        acc = accuracy_score(t_list, p_list)
        if acc > best_acc:
            best_acc = acc
            best_preds = np.array(p_list)
            best_targets = np.array(t_list)
            
    macro_f1 = f1_score(best_targets, best_preds, average='macro')
    return best_acc, macro_f1, best_preds, best_targets


def run_ablation_experiments():
    data_dir = "Base_de_Dados"
    res_dir = os.path.join("results", "ablations")
    vis_dir = os.path.join("results", "visualizations")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("      SYSTEMATIC ABLATION STUDY: MEASURING EACH ARCHITECTURAL COMPONENT")
    print(f"      Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)
    
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
    num_classes = len(classes)
    
    print("\nPreloading 2,148 audio files into RAM...")
    audio_matrix = preload_all_audio(filepaths, target_len=44100)
    
    # Class group indices for specific physical fault diagnostics
    slip_classes = [c for c in classes if "Escorregamento" in c]
    tooth_classes = [c for c in classes if "Perda_concentrada" in c]
    wear_classes = [c for c in classes if "Perda_material" in c]
    
    slip_ids = [class_to_idx[c] for c in slip_classes]
    tooth_ids = [class_to_idx[c] for c in tooth_classes]
    wear_ids = [class_to_idx[c] for c in wear_classes]
    
    # Chronological Block Split setup (Train on sessions 1-135, Test on sessions 136-179)
    block_tr_idx = []
    block_val_idx = []
    for c_id in range(num_classes):
        c_indices = np.where(labels == c_id)[0]
        block_tr_idx.extend(c_indices[:135])
        block_val_idx.extend(c_indices[135:])
        
    b_tr_ds = MemoryFaultDataset(audio_matrix[block_tr_idx], labels[block_tr_idx], augment=True)
    b_val_ds = MemoryFaultDataset(audio_matrix[block_val_idx], labels[block_val_idx], augment=False)
    b_tr_loader = DataLoader(b_tr_ds, batch_size=64, shuffle=True, pin_memory=True)
    b_val_loader = DataLoader(b_val_ds, batch_size=64, shuffle=False, pin_memory=True)
    
    # Stratified 5-Fold split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define Ablation Configurations
    ablation_configs = [
        {
            "id": "M0",
            "name": "Full TF-FaultNet (Proposed)",
            "use_se": True,
            "pooling_mode": "dual",
            "use_instance_norm": True,
            "use_residual": True,
            "description": "Complete architecture (SE Attention + Dual Avg/Max Pool + InstanceNorm + ResNet)"
        },
        {
            "id": "A1",
            "name": "A1: No SE Channel Attention",
            "use_se": False,
            "pooling_mode": "dual",
            "use_instance_norm": True,
            "use_residual": True,
            "description": "Removes Squeeze-and-Excitation channel gating (no dynamic frequency recalibration)"
        },
        {
            "id": "A2",
            "name": "A2: Average Pooling Only",
            "use_se": True,
            "pooling_mode": "avg_only",
            "use_instance_norm": True,
            "use_residual": True,
            "description": "Replaces Dual Pooling with Global Avg Pool only (tests loss of shock/click detection)"
        },
        {
            "id": "A3",
            "name": "A3: Max Pooling Only",
            "use_se": True,
            "pooling_mode": "max_only",
            "use_instance_norm": True,
            "use_residual": True,
            "description": "Replaces Dual Pooling with Global Max Pool only (tests loss of continuous friction detection)"
        },
        {
            "id": "A4",
            "name": "A4: No Instance Normalization",
            "use_se": True,
            "pooling_mode": "dual",
            "use_instance_norm": False,
            "use_residual": True,
            "description": "Removes per-spectrogram Z-Score normalization (raw log-mel input)"
        },
        {
            "id": "A5",
            "name": "A5: Plain CNN (No Residual Skip)",
            "use_se": True,
            "pooling_mode": "dual",
            "use_instance_norm": True,
            "use_residual": False,
            "description": "Removes residual shortcut connections (plain sequential convolutional blocks)"
        }
    ]
    
    ablation_results = []
    
    for cfg in ablation_configs:
        cfg_id = cfg["id"]
        cfg_name = cfg["name"]
        print(f"\n[{cfg_id}] Evaluating {cfg_name}...")
        print(f"      {cfg['description']}")
        
        # 1. 5-Fold Stratified Cross-Validation
        fold_accs = []
        fold_f1s = []
        fold_slip_f1s = []
        fold_tooth_f1s = []
        fold_wear_f1s = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(filepaths, labels), 1):
            train_ds = MemoryFaultDataset(audio_matrix[train_idx], labels[train_idx], augment=True)
            val_ds = MemoryFaultDataset(audio_matrix[val_idx], labels[val_idx], augment=False)
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, pin_memory=True)
            
            torch.manual_seed(42 + fold)
            model = FlexibleFaultNet(
                num_classes=num_classes,
                use_se=cfg["use_se"],
                pooling_mode=cfg["pooling_mode"],
                use_instance_norm=cfg["use_instance_norm"],
                use_residual=cfg["use_residual"]
            ).to(device)
            
            acc, f1, preds, targets = train_eval_model(model, train_loader, val_loader, epochs=15, device=device)
            fold_accs.append(acc)
            fold_f1s.append(f1)
            
            # Per-fault group F1 scores
            prec_all, rec_all, f1_all, _ = precision_recall_fscore_support(targets, preds, labels=range(num_classes), zero_division=0)
            fold_slip_f1s.append(np.mean([f1_all[i] for i in slip_ids]))
            fold_tooth_f1s.append(np.mean([f1_all[i] for i in tooth_ids]))
            fold_wear_f1s.append(np.mean([f1_all[i] for i in wear_ids]))
            
        mean_cv_acc = np.mean(fold_accs)
        std_cv_acc = np.std(fold_accs)
        mean_cv_f1 = np.mean(fold_f1s)
        std_cv_f1 = np.std(fold_f1s)
        mean_slip_f1 = np.mean(fold_slip_f1s)
        mean_tooth_f1 = np.mean(fold_tooth_f1s)
        mean_wear_f1 = np.mean(fold_wear_f1s)
        
        # 2. Chronological Block Holdout Evaluation (Unseen Sessions)
        torch.manual_seed(42)
        b_model = FlexibleFaultNet(
            num_classes=num_classes,
            use_se=cfg["use_se"],
            pooling_mode=cfg["pooling_mode"],
            use_instance_norm=cfg["use_instance_norm"],
            use_residual=cfg["use_residual"]
        ).to(device)
        b_acc, b_f1, b_preds, b_targets = train_eval_model(b_model, b_tr_loader, b_val_loader, epochs=15, device=device)
        
        print(f"   -> 5-Fold Acc: {mean_cv_acc*100:.2f}% +- {std_cv_acc*100:.2f}% | Unseen Session Holdout: {b_acc*100:.2f}% | Tooth Loss F1: {mean_tooth_f1*100:.2f}% | Slip F1: {mean_slip_f1*100:.2f}%")
        
        ablation_results.append({
            "Ablation_ID": cfg_id,
            "Configuration": cfg_name,
            "Description": cfg["description"],
            "CV_Accuracy": mean_cv_acc,
            "CV_Accuracy_Std": std_cv_acc,
            "CV_Macro_F1": mean_cv_f1,
            "CV_Macro_F1_Std": std_cv_f1,
            "Session_Holdout_Acc": b_acc,
            "Session_Holdout_F1": b_f1,
            "Belt_Slip_F1": mean_slip_f1,
            "Tooth_Loss_F1": mean_tooth_f1,
            "Material_Wear_F1": mean_wear_f1,
            "Delta_vs_Full_CV": (mean_cv_acc - 0.9977) * 100,
            "Delta_vs_Full_Session": (b_acc - 0.9716) * 100
        })
        
    df_ablation = pd.DataFrame(ablation_results)
    df_ablation.to_csv(os.path.join(res_dir, "ablation_study_results.csv"), index=False)
    
    print("\n" + "=" * 90)
    print("                    ABLATION STUDY EXPERIMENTAL RESULTS TABLE")
    print("=" * 90)
    display_cols = ["Ablation_ID", "Configuration", "CV_Accuracy", "Session_Holdout_Acc", "Tooth_Loss_F1", "Belt_Slip_F1"]
    print(df_ablation[display_cols].to_string(index=False))
    
    # -------------------------------------------------------------
    # Visualization: Ablation Comparison Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    x = np.arange(len(df_ablation))
    width = 0.35
    
    # Plot 1: CV vs Session Holdout Accuracy
    axes[0].bar(x - width/2, df_ablation["CV_Accuracy"] * 100, width, label="5-Fold CV Accuracy (%)", color="#2563eb")
    axes[0].bar(x + width/2, df_ablation["Session_Holdout_Acc"] * 100, width, label="Unseen Session Holdout (%)", color="#059669")
    axes[0].set_title("Ablation Study: Cross-Validation vs. Out-of-Session Generalization", fontsize=11, fontweight='bold', pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_ablation["Ablation_ID"], fontsize=10, fontweight='bold')
    axes[0].set_ylabel("Accuracy (%)", fontsize=10, fontweight='bold')
    axes[0].set_ylim(80, 103)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].legend(frameon=True, loc="lower right")
    
    for i, row in df_ablation.iterrows():
        axes[0].text(i - width/2, row["CV_Accuracy"]*100 + 0.4, f"{row['CV_Accuracy']*100:.1f}%", ha='center', fontsize=8, fontweight='bold')
        axes[0].text(i + width/2, row["Session_Holdout_Acc"]*100 + 0.4, f"{row['Session_Holdout_Acc']*100:.1f}%", ha='center', fontsize=8, fontweight='bold')
        
    # Plot 2: Impact of Dual Pooling on Spiky (Tooth Loss) vs Smeared (Belt Slip) Faults
    axes[1].bar(x - width/2, df_ablation["Tooth_Loss_F1"] * 100, width, label="Tooth Loss F1 (Spiky Faults)", color="#dc2626")
    axes[1].bar(x + width/2, df_ablation["Belt_Slip_F1"] * 100, width, label="Belt Slip F1 (Continuous Smeared)", color="#d97706")
    axes[1].set_title("Dual Pooling Impact: Spiky Impacts (Tooth) vs. Smeared Friction (Slip)", fontsize=11, fontweight='bold', pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_ablation["Ablation_ID"], fontsize=10, fontweight='bold')
    axes[1].set_ylabel("F1-Score (%)", fontsize=10, fontweight='bold')
    axes[1].set_ylim(85, 103)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1].legend(frameon=True, loc="lower right")
    
    for i, row in df_ablation.iterrows():
        axes[1].text(i - width/2, row["Tooth_Loss_F1"]*100 + 0.4, f"{row['Tooth_Loss_F1']*100:.1f}%", ha='center', fontsize=8, fontweight='bold')
        axes[1].text(i + width/2, row["Belt_Slip_F1"]*100 + 0.4, f"{row['Belt_Slip_F1']*100:.1f}%", ha='center', fontsize=8, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "ablation_comparison.png"))
    plt.close()
    
    print(f"\n[SUCCESS] Ablation study successfully completed! Plot saved to {os.path.join(vis_dir, 'ablation_comparison.png')}")

if __name__ == "__main__":
    run_ablation_experiments()
