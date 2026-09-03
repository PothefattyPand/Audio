# Audio Fault Diagnosis — TF-FaultNet

Comprehensive deep learning benchmark and acoustic fault diagnosis system for industrial machinery and rotating equipment.

## Overview

This repository contains the complete dataset, models, training scripts, ablation studies, and evaluation benchmarks for acoustic fault diagnosis across 12 balanced mechanical health conditions.

- **Primary Architecture**: `TF-FaultNet` (InstanceNorm Log-Mel + 2D ResNet + Squeeze-and-Excitation + Dual Pooling)
- **Dataset**: `Base_de_Dados` (2,148 audio recordings, 12 classes, 44.1 kHz sampling rate)
- **Benchmark Performance**: 99.81% ± 0.18% 5-fold cross-validation accuracy; 98.30% unseen session holdout accuracy

---

## Directory Structure

```
├── Base_de_Dados/             # 12-class acoustic fault audio dataset (.wav)
├── results/                   # Evaluation results, checkpoints, and visualizations
│   ├── ablations/             # Ablation experiment CSV logs
│   ├── checkpoints/           # Model weights (best_tf_faultnet.pt)
│   ├── visualizations/        # Signal profiles, t-SNE, confusion matrices, etc.
│   └── summary.json           # Model performance summary
├── src/                       # Source modules (dataset loaders, models, utilities)
├── train.py                   # Main model training pipeline
├── ablation_study.py          # Multi-architecture ablation testing
├── eda_analysis.py            # Exploratory data analysis and feature extraction
├── diagnose_fit.py            # Model fit diagnostic and generalizability checks
├── generate_pdf_report.py     # PDF report generators
├── kaggle_run_custom_model.ipynb # Kaggle-compatible notebook workflow
└── MODEL_BENCHMARK_AND_ABLATION_COMPARISON.md # Detailed benchmarking dossier
```

---

## Quick Start

### 1. Installation

Ensure Python 3.9+ is installed, then install the required dependencies:

```bash
pip install torch torchaudio numpy pandas scipy scikit-learn matplotlib seaborn librosa
```

### 2. Exploratory Data Analysis

Run the feature extraction and distribution analysis:

```bash
python eda_analysis.py
```

### 3. Model Training

Train the primary TF-FaultNet model:

```bash
python train.py
```

### 4. Ablation Study & Benchmarking

Execute multi-model comparative benchmarking:

```bash
python ablation_study.py
```

---

## Results Summary

| Model | 5-Fold CV Accuracy | 5-Fold Macro F1 | Unseen Session Holdout | Parameters |
| :--- | :---: | :---: | :---: | :---: |
| **TF-FaultNet** | **99.81% ± 0.18%** | **99.81% ± 0.18%** | **98.30%** | **1.42 M** |
| WaveformCNN1D | 99.67% ± 0.24% | 99.67% ± 0.24% | 96.28% | 0.88 M |
| AudioBiGRU | 99.30% ± 0.33% | 99.30% ± 0.33% | 94.88% | 1.95 M |
| XGBoost Classifier | 96.46% ± 1.04% | 96.47% ± 1.03% | 91.63% | N/A |
