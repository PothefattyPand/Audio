# Problem Specification — TF-FaultNet Acoustic Fault Diagnosis

> Required by CLAUDE.md §2 before writing model code.

---

## Task Definition (Mitchell's Framework)

| Field | Value |
|-------|-------|
| **Task T** | Classify the mechanical health condition of a belt-driven transmission system from 1-second raw audio recordings |
| **Performance Metric P** | Macro F1-Score (primary), Top-1 Classification Accuracy, Macro ROC-AUC |
| **Experience / Data E** | 2,148 monaural WAV recordings (44.1 kHz, 1 s each) captured from a laboratory belt-drive test rig under 12 controlled fault conditions, grouped into 3 macro-level health categories |

---

## Input / Output Specification

| Field | Value |
|-------|-------|
| **Input X** | Raw audio waveform vector, shape `(44100,)` — 1 second at 44.1 kHz sample rate |
| **Target t** | Integer class label ∈ {0, 1, 2} representing one of 3 macro health conditions |
| **Unit of Analysis** | One 1-second audio recording (one WAV file) |

---

## Task Classification

| Field | Value |
|-------|-------|
| **Task Type** | Multiclass classification (K=3 mutually exclusive classes) |
| **Output Layer** | 3 logits → Softmax → class probabilities |
| **Loss Function** | Cross-Entropy Loss (with label smoothing ε=0.05) |
| **Decision Rule** | argmax over softmax probabilities |

---

## Metrics

| Field | Value |
|-------|-------|
| **Primary Metric** | Macro F1-Score (equally weights all 3 classes regardless of support) |
| **Secondary Metrics** | Top-1 Accuracy, Macro Precision, Macro Recall, Per-Class ROC-AUC, Per-Class PR-AUC, Confusion Matrix |

---

## Baseline

| Model | Type | Purpose |
|-------|------|---------|
| XGBoost (24 handcrafted features) | Simple ML baseline | Verify that deep learning is justified |
| Logistic Regression (24 features) | Naive baseline | Minimum viable comparator |
| TF-FaultNet (proposed) | Deep CNN with SE-Attention | Proposed method |

---

## Data Source

| Field | Value |
|-------|-------|
| **Dataset** | Base_de_Dados (belt-drive acoustic fault dataset) |
| **Origin** | Laboratory test rig with controlled mechanical fault injection |
| **Format** | 2,148 monaural WAV files, 44.1 kHz, ~1 second each |
| **12 Original Subfolders** | Normal, Escorregamento, Escorregamento_P1, Escorregamento_P1P4, Perda_material, Perda_material_P1, Perda_material_P1P4, Perda_concentrada, Perda_concentrada_P1, Perda_concentrada_P1P4, Sem_P1, Sem_P1P4 |
| **3-Class Macro Mapping** | 0: Healthy_Baseline (Normal), 1: Continuous_Friction_and_Wear (Escorregamento + Perda_material variants), 2: Impulsive_Shocks_and_Structural (Perda_concentrada + Sem variants) |

---

## Split Strategy

| Field | Value |
|-------|-------|
| **Method** | Stratified random split (class-proportional) via `StratifiedShuffleSplit` |
| **Ratios** | 64% Train / 16% Validation / 20% Held-Out Test |
| **Random Seed** | 42 |
| **Manifest** | `data/splits.csv`, column `split_multiclass_3class` (frozen file-level split assignment and single source of truth) |

---

## Leakage Risks

| Risk | Mitigation |
|------|------------|
| Same recording in train and test | Strict index-level disjoint assertion (`set_train ∩ set_test = ∅`) |
| Preprocessing fit on test data | All augmentation/normalization applied to train split only |
| Test-driven hyperparameter tuning | All tuning uses validation loss; test set evaluated once after model lock |
| Temporal autocorrelation within recordings | Each recording is an independent 1-second capture; no temporal sequence across files |
| Same rig condition across splits | Stratified split ensures each class is proportionally represented; however, all recordings come from the same physical rig (single-site limitation) |

---

## Constraints

- Single-site laboratory data (one test rig, one microphone position)
- Fixed rotational speed during recording sessions
- No cross-domain transfer tested (e.g., different machines, environments)
- Audio only — no vibration, current, or thermal sensor fusion
- Model must be exportable to TorchScript JIT and ONNX for edge deployment

---

## Ethical / Privacy Risks

| Dimension | Assessment |
|-----------|------------|
| **Personal data** | None — all recordings are from an industrial test rig, no human subjects or PII |
| **Dual use** | Low risk — fault diagnosis is a safety-improving application |
| **Bias** | Single-rig bias: model may not generalize to different machine types, environments, or microphone placements without retraining |
| **Safety** | If deployed for real-time condition monitoring, false negatives (missed faults) could lead to continued unsafe operation. A human-in-the-loop verification step is recommended for critical machinery |
| **Reproducibility** | Seed-locked (42), split manifest saved, all hyperparameters documented |
