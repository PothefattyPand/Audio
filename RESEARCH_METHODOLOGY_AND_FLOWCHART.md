# Research Methodology & Pipeline Flowchart
## TF-FaultNet: End-to-End Acoustic Fault Diagnosis via Time-Frequency Attention Networks

> Based on the governing ML pipeline defined in the project protocol (CLAUDE.md §1–§30).

---

## 1. End-to-End Research Pipeline Flowchart

```mermaid
flowchart TD
    subgraph P1["Phase 1: Problem Definition"]
        A["Define Task T, Metric P, Data E"] --> B["Identify Input X, Target t, Unit of Analysis"]
        B --> C["Route Task Type: Multiclass Classification K=3"]
        C --> D["Select Output/Loss: 3 Logits → Softmax → CrossEntropyLoss"]
    end

    subgraph P2["Phase 2: Data Acquisition & EDA"]
        E["Load Base_de_Dados: 2,148 WAV files, 44.1 kHz, 1s each"] --> F["Map 12 Subfolders → 3 Macro Classes"]
        F --> G["EDA Checks: Counts, Distribution, Missingness, Duplicates, Outliers"]
        G --> H["Report Class Imbalance: 8.3% / 50.0% / 41.7%"]
        H --> I["Check Group Structure: Single-Site Lab Rig"]
    end

    subgraph P3["Phase 3: Partitioning"]
        J["Stratified 3-Way Split: 64% / 16% / 20%"] --> K["Train: 1,374 samples"]
        J --> L["Validation: 344 samples"]
        J --> M["Test: 430 samples — Quarantined"]
        K --> N["Zero-Leakage Assertions"]
        L --> N
        M --> N
        N --> O["Save Frozen Split Manifest: data/splits.csv"]
    end

    subgraph P4["Phase 4: Preprocessing & Feature Extraction"]
        Q["Raw Audio x ∈ ℝ^44100"] --> R["STFT: N=1024, H=256, Hann Window"]
        R --> S["Mel Filterbank: 64 Triangular Filters, 20 Hz–22 kHz"]
        S --> T["Log Compression: log(mel + 1e-6)"]
        T --> U["Instance Z-Score Normalization"]
        U --> V["Log-Mel Spectrogram: S̃ ∈ ℝ^(1×64×173)"]
        W["Train-Only Augmentation"] --> X["Gaussian Noise σ=0.003, p=0.35"]
        W --> Y["Random Gain 0.85–1.15×, p=0.35"]
    end

    subgraph P5["Phase 5: Baseline Models"]
        BA["Logistic Regression"] --> BD["24 Handcrafted Statistical Features"]
        BB["Random Forest 200 Trees"] --> BD
        BC["XGBoost depth=6"] --> BD
        BD --> BE["Evaluate All on Same Split"]
    end

    subgraph P6["Phase 6: Proposed Model — TF-FaultNet"]
        CA["Init Conv: Conv2D 1→32, k=5, s=2 + BN + ReLU + MaxPool"] --> CB["Stage 1: ConvBlock 32→64, stride=2 + SE Attention"]
        CB --> CC["Stage 2: ConvBlock 64→128, stride=2 + SE Attention"]
        CC --> CD["Stage 3: ConvBlock 128→256, stride=2 + SE Attention"]
        CD --> CE["Dual Pooling: GAP ∥ GMP → 512-D"]
        CE --> CF["Classifier: Dropout 0.35 → FC 512→256 → ReLU → Dropout 0.2 → FC 256→3"]
    end

    subgraph P7["Phase 7: Training & Validation"]
        DA["AdamW: lr=1e-3, wd=1e-4"] --> DB["CosineAnnealingLR: T_max=25, η_min=1e-5"]
        DB --> DC["CrossEntropyLoss + Label Smoothing ε=0.05"]
        DC --> DD["Train 25 Epochs, Batch=64"]
        DD --> DE["Checkpoint Selection: Best Validation Loss"]
        DE --> DF["Best Epoch 14: Val Loss=0.1992, Val Acc=98.84%"]
    end

    subgraph P8["Phase 8: Final Test Evaluation"]
        EA["Load Locked Checkpoint"] --> EB["model.eval + torch.no_grad"]
        EB --> EC["Run Quarantined Test Set Once"]
        EC --> ED["Compute Metrics: Accuracy, F1, ROC-AUC, PR-AUC"]
        ED --> EE["Confusion Matrix + Per-Class Breakdown"]
        EE --> EF["Error Analysis: 6 Misclassifications Inspected"]
    end

    subgraph P9["Phase 9: Export & Deployment"]
        FA["PyTorch Weights .pt"] --> FD["Dual Inference Verification"]
        FB["TorchScript JIT .pt"] --> FD
        FC["ONNX Backbone .onnx"] --> FD
        FD --> FE["430/430 Predictions Match: Audio ≡ Image Inference"]
    end

    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8 --> P9
```

---

## 2. Detailed Phase Descriptions

### Phase 1: Problem Definition (Protocol §2, §4, §5)

| Field | Value |
|-------|-------|
| **Task T** | Classify mechanical health condition from 1-second audio recordings |
| **Performance Metric P** | Macro F1-Score (primary); Accuracy, ROC-AUC (secondary) |
| **Experience / Data E** | 2,148 monaural WAV recordings from a belt-drive test rig |
| **Input X** | Raw waveform vector $x \in \mathbb{R}^{44100}$ (1 s @ 44.1 kHz) |
| **Target t** | Integer class label $\in \{0, 1, 2\}$ |
| **Unit of Analysis** | One 1-second WAV file |
| **Task Type** | Multiclass classification ($K = 3$ mutually exclusive classes) |
| **Task Routing** | §4: *K mutually exclusive classes → multiclass classification* |
| **Output/Loss Routing** | §5: *K logits → Softmax → Cross-Entropy → argmax prediction* |

---

### Phase 2: Data Acquisition & Exploratory Data Analysis (Protocol §6)

#### 2.1 Dataset Overview

The `Base_de_Dados` dataset contains acoustic recordings from a laboratory belt-driven transmission system under 12 controlled fault injection conditions. Each recording is a 1-second monaural WAV captured at 44.1 kHz.

#### 2.2 Three-Class Macro Taxonomy Mapping

The 12 original subfolders are collapsed into 3 physics-grounded macro classes via a deterministic surjective mapping $\mathcal{M}: \mathcal{Y}_{12} \to \mathcal{Y}_3$:

```mermaid
flowchart LR
    subgraph Original["12 Original Subfolders"]
        N["Normal (179)"]
        E1["Escorregamento (179)"]
        E2["Escorregamento_P1 (179)"]
        E3["Escorregamento_P1P4 (179)"]
        PM1["Perda_material (179)"]
        PM2["Perda_material_P1 (179)"]
        PM3["Perda_material_P1P4 (179)"]
        PC1["Perda_concentrada (179)"]
        PC2["Perda_concentrada_P1 (179)"]
        PC3["Perda_concentrada_P1P4 (179)"]
        S1["Sem_P1 (179)"]
        S2["Sem_P1P4 (179)"]
    end

    subgraph Macro["3 Macro Classes"]
        C0["Class 0: Healthy Baseline\n(179 samples, 8.3%)"]
        C1["Class 1: Continuous Friction & Wear\n(1,074 samples, 50.0%)"]
        C2["Class 2: Impulsive Shocks & Structural\n(895 samples, 41.7%)"]
    end

    N --> C0
    E1 --> C1
    E2 --> C1
    E3 --> C1
    PM1 --> C1
    PM2 --> C1
    PM3 --> C1
    PC1 --> C2
    PC2 --> C2
    PC3 --> C2
    S1 --> C2
    S2 --> C2
```

#### 2.3 EDA Checklist (Protocol §6 Compliance)

| Check | Result |
|-------|--------|
| Sample count | 2,148 total |
| Feature types | Continuous 1D audio waveform (44,100 float32 samples) |
| Target distribution | Imbalanced: 8.3% / 50.0% / 41.7% |
| Missingness | None — all files complete |
| Duplicates | No exact duplicates |
| Outliers | No signal-level outliers (all normalized to [-1, 1]) |
| Class imbalance | Severe for Class 0 — mitigated by stratified split + Macro F1 |
| Group structure | Single-site laboratory rig, fixed microphone |
| Temporal ordering | Independent captures — IID assumption valid |
| Spectrogram shape | $(1, 64, 173)$ — 1 channel, 64 mel bins, 173 time frames |

---

### Phase 3: Data Partitioning (Protocol §3.1, §3.9, §7)

#### 3.1 Stratified Three-Way Split

```mermaid
flowchart LR
    D["2,148 Samples"] -->|"StratifiedShuffleSplit\nseed=42"| TV["Train + Val\n(1,718 = 80%)"]
    D -->|"20%"| TEST["Held-Out Test\n(430 samples)\n🔒 QUARANTINED"]
    TV -->|"StratifiedShuffleSplit\nseed=42"| TRAIN["Train\n(1,374 = 64%)"]
    TV -->|"20% of 80%"| VAL["Validation\n(344 = 16%)"]

    TRAIN -.->|"Used for"| BP["Backpropagation\n& Augmentation"]
    VAL -.->|"Used for"| CS["Checkpoint Selection\n& Convergence Monitoring"]
    TEST -.->|"Used for"| FE["Final Evaluation\n(Run Once, After Lock)"]
```

#### 3.2 Zero-Leakage Guarantees (Protocol §3.1)

- **Assertion 1:** $\text{Train} \cap \text{Val} = \emptyset$
- **Assertion 2:** $\text{Train} \cap \text{Test} = \emptyset$
- **Assertion 3:** $\text{Val} \cap \text{Test} = \emptyset$
- **Assertion 4:** $|\text{Train}| + |\text{Val}| + |\text{Test}| = 2{,}148$
- **Frozen manifest:** Saved to `data/splits.csv` (Protocol §3.9)

---

### Phase 4: Preprocessing & Time-Frequency Transformation (Protocol §8)

#### 4.1 End-to-End In-Graph Feature Extraction

Unlike offline pipelines, TF-FaultNet embeds the spectrogram extraction **inside the computational graph** as a differentiable module. No pre-rendered images are stored.

```mermaid
flowchart TD
    A["Raw Audio\nx ∈ ℝ^44100\n(1s @ 44.1 kHz)"] --> B["STFT\nN=1024, H=256\nHann Window\nCenter=True, Reflect Pad"]
    B --> C["Power Spectrum\n|X(m,k)|²\n∈ ℝ^(513 × 173)"]
    C --> D["Mel Filterbank\n64 Triangular Filters\n20 Hz – 22,050 Hz"]
    D --> E["Mel Power Spectrum\nM(m,b) ∈ ℝ^(64 × 173)"]
    E --> F["Log Compression\nS_log = ln(M + 1e-6)"]
    F --> G["Instance Z-Score\nS̃ = (S_log − μ) / (σ + 1e-6)"]
    G --> H["Output\nS̃ ∈ ℝ^(1 × 64 × 173)"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style H fill:#1a1a2e,stroke:#0f3460,color:#fff
```

#### 4.2 Data Augmentation (Train Only — Protocol §8)

| Augmentation | Parameter | Probability | Applied To |
|---|---|---|---|
| Additive Gaussian Noise | $\sigma = 0.003$ | 35% | Train only |
| Random Gain Scaling | $[0.85, 1.15] \times$ | 35% | Train only |

---

### Phase 5: Baseline Models (Protocol §9, §28)

**Protocol Requirement (§9):** *"Use the simplest justified model first."*

| Model | Input | Features | Purpose |
|-------|-------|----------|---------|
| Logistic Regression | 24-D feature vector | Handcrafted statistical descriptors (RMS, crest factor, spectral centroid, kurtosis, etc.) | Naive baseline |
| Random Forest (200 trees) | 24-D feature vector | Same handcrafted features | Simple ML baseline |
| XGBoost (depth=6) | 24-D feature vector | Same handcrafted features | Established ML baseline |
| **TF-FaultNet** | Raw 44,100 samples | **End-to-end learned** (Log-Mel + CNN + SE-Attention) | **Proposed method** |

> Complexity justification (§3.5): TF-FaultNet is only adopted because baseline models show measurable limitations on the validation set, validating the need for automatic feature learning.

---

### Phase 6: Proposed Architecture — TF-FaultNet (Protocol §11, §17–§21)

#### 6.1 Architectural Justification (Protocol §11, §17)

| Criterion | Justification |
|-----------|---------------|
| **Why deep learning?** | Unstructured audio input; automatic time-frequency feature learning outperforms handcrafted descriptors |
| **Why CNN?** | Spectrograms are 2D grid-structured data → spatial convolution is the natural inductive bias |
| **Why ResNet-style blocks?** | Skip connections mitigate vanishing gradients in deeper networks (§15) |
| **Why SE-Attention?** | Channel recalibration suppresses irrelevant frequency bands (motor hum) and amplifies diagnostic bands |
| **Why Dual GAP+GMP?** | GAP captures sustained energy patterns (friction), GMP isolates peak activations (shock transients) |

#### 6.2 Layer-by-Layer Architecture Table (Protocol §20)

| Layer | Input Shape | Kernel | Stride | Padding | Filters | Output Shape | Trainable Params |
|-------|-------------|--------|--------|---------|---------|-------------|-----------------|
| **Input** | $(B, 1, 64, 173)$ | — | — | — | — | $(B, 1, 64, 173)$ | 0 |
| **Init Conv2D** | $(B, 1, 64, 173)$ | $5 \times 5$ | 2 | 2 | 32 | $(B, 32, 32, 87)$ | 800 |
| BatchNorm2D | $(B, 32, 32, 87)$ | — | — | — | — | $(B, 32, 32, 87)$ | 64 |
| ReLU | — | — | — | — | — | — | 0 |
| MaxPool2D | $(B, 32, 32, 87)$ | $2 \times 2$ | 2 | 0 | — | $(B, 32, 16, 43)$ | 0 |
| **ConvBlock Stage 1** | $(B, 32, 16, 43)$ | $3 \times 3$ | 2 | 1 | 64 | $(B, 64, 8, 22)$ | 55,936 |
| **ConvBlock Stage 2** | $(B, 64, 8, 22)$ | $3 \times 3$ | 2 | 1 | 128 | $(B, 128, 4, 11)$ | 222,464 |
| **ConvBlock Stage 3** | $(B, 128, 4, 11)$ | $3 \times 3$ | 2 | 1 | 256 | $(B, 256, 2, 6)$ | 887,296 |
| **AdaptiveAvgPool2D** | $(B, 256, 2, 6)$ | global | — | — | — | $(B, 256, 1, 1)$ | 0 |
| **AdaptiveMaxPool2D** | $(B, 256, 2, 6)$ | global | — | — | — | $(B, 256, 1, 1)$ | 0 |
| Concatenate GAP∥GMP | — | — | — | — | — | $(B, 512)$ | 0 |
| Dropout (0.35) | $(B, 512)$ | — | — | — | — | $(B, 512)$ | 0 |
| **FC Layer 1** | $(B, 512)$ | — | — | — | 256 | $(B, 256)$ | 131,328 |
| ReLU | — | — | — | — | — | — | 0 |
| Dropout (0.2) | $(B, 256)$ | — | — | — | — | $(B, 256)$ | 0 |
| **FC Output** | $(B, 256)$ | — | — | — | 3 | $(B, 3)$ | 771 |
| **Total** | | | | | | | **~1.3M** |

#### 6.3 Squeeze-and-Excitation (SE) Attention Block

```mermaid
flowchart LR
    U["Feature Map\nU ∈ ℝ^(C×H×W)"] --> GAP2["Global Avg Pool\nz ∈ ℝ^C"]
    GAP2 --> FC1["FC: C → C/r\n+ ReLU"]
    FC1 --> FC2["FC: C/r → C\n+ Sigmoid"]
    FC2 --> Scale["Channel-wise Scale\nŨ_c = s_c · U_c"]
    U --> Scale
```

$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z))$, with reduction ratio $r = 8$.

#### 6.4 Residual ConvBlock with SE

```mermaid
flowchart TD
    X["Input x"] --> CONV1["Conv2D 3×3 + BN + ReLU"]
    CONV1 --> CONV2["Conv2D 3×3 + BN"]
    CONV2 --> SE["SE Attention Block"]
    X --> SHORT["Shortcut\n(1×1 Conv if dims differ)"]
    SE --> ADD["+ Residual Addition"]
    SHORT --> ADD
    ADD --> RELU["ReLU"]
    RELU --> OUT["Output\nH(x) = F(x) + x"]
```

---

### Phase 7: Training & Optimization Protocol (Protocol §10, §15, §16)

#### 7.1 Hyperparameter Configuration

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay for better generalization |
| Learning rate ($\eta_0$) | $1 \times 10^{-3}$ | Standard for Adam-family optimizers |
| Weight decay ($\lambda$) | $1 \times 10^{-4}$ | Mild L2 regularization |
| Batch size ($B$) | 64 | Full GPU utilization without memory overflow |
| Epochs | 25 | Sufficient for convergence with cosine annealing |
| LR Scheduler | CosineAnnealingLR | Smooth decay: $\eta_0 = 10^{-3} \to \eta_{\min} = 10^{-5}$ |
| Loss | CrossEntropyLoss | Standard for multiclass (Protocol §5) |
| Label smoothing ($\epsilon$) | 0.05 | Prevents overconfident predictions |
| Dropout | 0.35 (pre-FC), 0.2 (mid-FC) | Regularization in classifier head (Protocol §16) |

#### 7.2 Training Loop Order (Protocol §10)

```mermaid
flowchart LR
    FP["1. Forward Propagation\nŷ = f_θ(x)"] --> CL["2. Compute Loss\nℒ = CE(y, ŷ)"]
    CL --> ZG["3. Zero Gradients\noptimizer.zero_grad()"]
    ZG --> BP["4. Backward Propagation\nℒ.backward()"]
    BP --> US["5. Optimizer Step\nθ ← θ − η·∇ℒ"]
    US --> LR["6. Scheduler Step\nUpdate η"]
```

#### 7.3 Checkpoint Selection

- **Criterion:** Minimum validation loss (not test performance — Protocol §3.1)
- **Best epoch:** 14 out of 25
- **Best val loss:** 0.1992
- **Best val accuracy:** 98.84%
- **Convergence indicator:** Train and val losses decrease monotonically and plateau together (no divergence = no overfitting)

---

### Phase 8: Final Test Evaluation (Protocol §27)

#### 8.1 Evaluation Protocol

```mermaid
flowchart TD
    A["Load Locked Checkpoint\n(Epoch 14)"] --> B["model.eval()\nDisable Dropout & BN training mode"]
    B --> C["torch.no_grad()\nDisable gradient computation"]
    C --> D["Forward pass on\n430 quarantined test samples"]
    D --> E["Softmax → argmax\nGet predicted labels"]
    E --> F["Compute Metrics"]
    F --> G["Accuracy: 98.60%"]
    F --> H["Macro F1: 98.59%"]
    F --> I["Macro ROC-AUC: 0.99954"]
    F --> J["Confusion Matrix"]
    F --> K["Per-Class ROC & PR Curves"]
```

#### 8.2 Final Test Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **98.60%** (424/430) |
| Macro Precision | 98.92% |
| Macro Recall | 98.30% |
| **Macro F1-Score** | **98.59%** |
| **Macro ROC-AUC** | **0.99954** |
| Macro PR-AUC | 0.99937 |

#### 8.3 Per-Class Performance

| Class | Precision | Recall | F1-Score | ROC-AUC | Support |
|-------|-----------|--------|----------|---------|---------|
| Healthy Baseline | 100.00% | 97.22% | 98.59% | 1.00000 | 36 |
| Continuous Friction & Wear | 100.00% | 97.67% | 98.82% | 0.99918 | 215 |
| Impulsive Shocks & Structural | 96.76% | 100.00% | 98.35% | 0.99922 | 179 |

#### 8.4 Error Analysis (Protocol §27)

Total misclassifications: **6 out of 430** (1.40%)

| Error Type | Count | Interpretation |
|------------|-------|----------------|
| Healthy → Impulsive | 1 | Mild structural resonance near decision boundary |
| Friction → Impulsive | 5 | Overlapping spectral energy at friction–impact transition |

---

### Phase 9: Model Export & Deployment Readiness (Protocol §30)

| Format | File | Purpose |
|--------|------|---------|
| PyTorch Weights | `best_tf_faultnet_3class.pt` | Research reproducibility |
| TorchScript JIT | `best_tf_faultnet_3class_jit.pt` | Production inference (Python-free) |
| ONNX Backbone | `best_tf_faultnet_3class_backbone.onnx` | Cross-platform edge deployment |

**Dual-inference verification:** 430/430 predictions are **identical** between end-to-end audio inference and separate 2D spectrogram image inference → confirms the in-graph feature extractor is faithful.

---

## 3. Protocol Compliance Summary

```mermaid
flowchart LR
    subgraph Compliant["✅ Protocol Rules Followed"]
        R1["§3.1 No test leakage"]
        R2["§3.2 Fit on train only"]
        R3["§3.3 Baseline before complex"]
        R4["§3.8 Raw data immutable"]
        R5["§3.9 Split manifest saved"]
        R6["§3.10 Reproducible: seed=42"]
        R7["§5 Correct output/loss routing"]
        R8["§10 Proper gradient order"]
        R9["§16 Dropout/BN eval mode"]
        R10["§27 model.eval + no_grad"]
    end
```

---

## 4. Limitations & Future Work

1. **Single-site data:** All recordings from one laboratory rig — cross-domain generalization untested
2. **Fixed operating speed:** No variable-speed or transient startup conditions
3. **Audio-only modality:** No vibration, current, or thermal sensor fusion
4. **Small healthy class:** Only 179/2,148 samples (8.3%) are healthy baseline
5. **No real-world noise:** Lab conditions may not reflect factory floor acoustics

---

## 5. Ethical Considerations

- **No personal data:** Industrial machine recordings, no human subjects or PII
- **Safety implication:** False negatives could allow continued unsafe operation — a human-in-the-loop is recommended for safety-critical deployments
- **Reproducibility:** Seed-locked, manifest-saved, config-exported — fully reproducible
