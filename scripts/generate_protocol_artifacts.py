"""
Generate protocol-required artifacts:
1. data/splits.csv — frozen 3-class file-level split manifest (§7, §3.9)
2. configs/experiment_3class.json — reproducibility config (§3.10, §26)
3. metadata/class_mapping_3class.json — class mapping (§29)
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# 3-Class Mapping
MACRO_CLASSES = [
    "Healthy_Baseline",
    "Continuous_Friction_and_Wear",
    "Impulsive_Shocks_and_Structural"
]

FOLDER_TO_3CLASS = {
    "Normal": 0,
    "Escorregamento": 1,
    "Escorregamento_P1": 1,
    "Escorregamento_P1P4": 1,
    "Perda_material": 1,
    "Perda_material_P1": 1,
    "Perda_material_P1P4": 1,
    "Perda_concentrada": 2,
    "Perda_concentrada_P1": 2,
    "Perda_concentrada_P1P4": 2,
    "Sem_P1": 2,
    "Sem_P1P4": 2
}

def main():
    data_dir = "Base_de_Dados"
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    filepaths = []
    labels_3class = []
    original_folders = []
    
    for d in subdirs:
        if d not in FOLDER_TO_3CLASS:
            continue
        c3 = FOLDER_TO_3CLASS[d]
        wavs = sorted(glob.glob(os.path.join(data_dir, d, "*.wav")))
        for w in wavs:
            filepaths.append(w)
            labels_3class.append(c3)
            original_folders.append(d)
    
    filepaths = np.array(filepaths)
    labels = np.array(labels_3class)
    original_folders = np.array(original_folders)
    
    # Replicate exact same split (seed=42)
    np.random.seed(42)
    
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_val_idx, test_idx = next(sss_test.split(filepaths, labels))
    
    train_val_labels = labels[train_val_idx]
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_sub_idx, val_sub_idx = next(sss_val.split(filepaths[train_val_idx], train_val_labels))
    
    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]
    
    # Verify zero leakage
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    
    # Build split column
    split_col = np.full(len(filepaths), "", dtype=object)
    split_col[train_idx] = "train"
    split_col[val_idx] = "val"
    split_col[test_idx] = "test"
    
    # ──────────────────────────────────────────────────
    # 1. data/splits.csv
    # ──────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    
    df = pd.DataFrame({
        "filename": [os.path.basename(f) for f in filepaths],
        "filepath": filepaths,
        "original_subfolder": original_folders,
        "macro_class_index": labels,
        "macro_class_name": [MACRO_CLASSES[l] for l in labels],
        "split_multiclass_3class": split_col
    })
    
    manifest_path = "data/splits.csv"
    if os.path.exists(manifest_path):
        raise FileExistsError(
            f"Refusing to overwrite frozen split manifest: {manifest_path}. "
            "Move it explicitly before generating a new experimental protocol."
        )
    df.to_csv(manifest_path, index=False)
    print(f"[1/3] Saved data/splits.csv ({len(df)} rows)")
    print(f"       Train: {(split_col == 'train').sum()}, Val: {(split_col == 'val').sum()}, Test: {(split_col == 'test').sum()}")
    
    # ──────────────────────────────────────────────────
    # 2. configs/experiment_3class.json
    # ──────────────────────────────────────────────────
    os.makedirs("configs", exist_ok=True)
    
    config = {
        "experiment_name": "TF-FaultNet 3-Class Macro Taxonomy",
        "seed": 42,
        "model": {
            "architecture": "TFFaultNet",
            "num_classes": 3,
            "input_type": "raw_audio_waveform",
            "input_shape": [44100],
            "spectrogram": {
                "sr": 44100,
                "n_fft": 1024,
                "hop_length": 256,
                "n_mels": 64
            },
            "backbone": "Custom ResNet + SE-Attention",
            "pooling": "AdaptiveAvgPool2d + AdaptiveMaxPool2d (concatenated)",
            "classifier_dropout": [0.35, 0.2]
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "epochs": 25,
            "scheduler": "CosineAnnealingLR",
            "scheduler_params": {
                "T_max": 25,
                "eta_min": 1e-5
            },
            "loss_function": "CrossEntropyLoss",
            "label_smoothing": 0.05,
            "augmentation": "train_only (via MemoryFaultDataset augment=True)"
        },
        "data_split": {
            "method": "StratifiedShuffleSplit",
            "ratios": {"train": 0.64, "val": 0.16, "test": 0.20},
            "random_state": 42,
            "manifest": "data/splits.csv"
        },
        "checkpoint_selection": {
            "criterion": "lowest_validation_loss",
            "best_epoch": 14,
            "best_val_loss": 0.1992,
            "best_val_acc": 0.9884,
            "checkpoint_path": "results/checkpoints/best_tf_faultnet_3class.pt"
        },
        "evaluation": {
            "primary_metric": "macro_f1_score",
            "secondary_metrics": ["accuracy", "macro_precision", "macro_recall", "macro_roc_auc", "per_class_roc_auc"],
            "test_set_size": 430
        }
    }
    
    with open("configs/experiment_3class.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[2/3] Saved configs/experiment_3class.json")
    
    # ──────────────────────────────────────────────────
    # 3. metadata/class_mapping_3class.json
    # ──────────────────────────────────────────────────
    os.makedirs("metadata", exist_ok=True)
    
    mapping = {
        "taxonomy": "3-Class Macro Taxonomy",
        "classes": {
            "0": {
                "name": "Healthy_Baseline",
                "description": "Normal operating condition with no faults",
                "original_subfolders": ["Normal"]
            },
            "1": {
                "name": "Continuous_Friction_and_Wear",
                "description": "Slippage and diffuse material loss conditions producing continuous friction-type acoustic signatures",
                "original_subfolders": ["Escorregamento", "Escorregamento_P1", "Escorregamento_P1P4", "Perda_material", "Perda_material_P1", "Perda_material_P1P4"]
            },
            "2": {
                "name": "Impulsive_Shocks_and_Structural",
                "description": "Concentrated tooth loss and missing pulley conditions producing impulsive shock-type acoustic signatures",
                "original_subfolders": ["Perda_concentrada", "Perda_concentrada_P1", "Perda_concentrada_P1P4", "Sem_P1", "Sem_P1P4"]
            }
        },
        "folder_to_class_index": FOLDER_TO_3CLASS
    }
    
    with open("metadata/class_mapping_3class.json", "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[3/3] Saved metadata/class_mapping_3class.json")
    
    print("\nAll protocol artifacts generated successfully.")

if __name__ == "__main__":
    main()
