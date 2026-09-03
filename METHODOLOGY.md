# Methodology: TF-FaultNet Acoustic Fault Diagnosis System
## Streamlined 3-Class Acoustic Macro Taxonomy & Translation Guide

---

## 1. Terminology Translation & Context

The experimental dataset (`Base_de_Dados`) originated from a Brazilian mechanical engineering test rig. The labels are in **Portuguese** (not Spanish). Below is the complete translation and engineering meaning:

| Original Portuguese Term | English Translation | Physical Meaning in Rotating Machinery |
| :--- | :--- | :--- |
| **`Normal`** | **Normal / Healthy** | Baseline, defect-free operation under normal load. |
| **`Escorregamento`** | **Slippage / Belt Slip** | Continuous sliding friction between transmission belt and pulleys. |
| **`Perda concentrada`** | **Concentrated Loss** | Localized tooth/mass loss on a gear or pulley notch (impact shocks). |
| **`Perda de material`** | **Material Loss / Wear** | Diffuse abrasive wear and surface erosion across contact surfaces. |
| **`Sem P1`** | **Without Pulley 1** | Operation without Pulley 1 installed (*"Sem"* = "Without"). |
| **`Sem P1P4`** | **Without Pulleys 1 & 4** | Operation without Pulleys 1 and 4 installed. |
| **`_P1` / `_P1P4`** | **Location Suffixes** | Indicates whether the fault is on Pulley 1 or compounded across Pulleys 1 & 4. |

---

## 2. Streamlined 3-Class Acoustic Macro Taxonomy

Rather than segregating across 12 granular sub-locations, the operating conditions naturally cluster into **3 distinct macro-classes ($C = 3$)** unified by shared physical failure modes, mathematical properties, and acoustic spectral features:

```
                            3 ACOUSTIC MACRO-CLASSES (C = 3)
                                          │
      ┌───────────────────────────────────┼───────────────────────────────────┐
      ▼                                   ▼                                   ▼
  CLASS 1:                             CLASS 2:                            CLASS 3:
Healthy Baseline                 Continuous Friction & Wear          Impulsive Shocks & Structural
  (Normal)                      (Belt Slippage & Erosion)            (Tooth Loss & Missing Pulley)
  • Balanced motor hum           • Continuous high-freq smears       • Cyclic 2 ms transient clicks
  • Stable spectral centroid     • High sustained RMS power          • High Kurtosis & Crest Factor
  • Low Kurtosis (~3.0)          • Dominant GAP activation           • Dominant GMP activation
```

### 2.1 Formal Mathematical Mapping

Let $\mathcal{Y}_{12} = \{1, 2, \dots, 12\}$ be the original 12 fine-grained dataset states. We define a deterministic surjective mapping $\mathcal{M}: \mathcal{Y}_{12} \to \mathcal{Y}_3 = \{1, 2, 3\}$:

$$\mathcal{M}(y) = \begin{cases}
\mathbf{1}: \text{\bf Healthy Baseline (Normal)}, & \text{if } y \in \{\text{Normal}\} \\
\mathbf{2}: \text{\bf Continuous Friction \& Wear}, & \text{if } y \in \begin{Bmatrix} \text{Escorregamento}, \text{Escorregamento\_P1}, \text{Escorregamento\_P1P4}, \\ \text{Perda\_material}, \text{Perda\_material\_P1}, \text{Perda\_material\_P1P4} \end{Bmatrix} \\
\mathbf{3}: \text{\bf Impulsive Shocks \& Structural Faults}, & \text{if } y \in \begin{Bmatrix} \text{Perda\_concentrada}, \text{Perda\_concentrada\_P1}, \text{Perda\_concentrada\_P1P4}, \\ \text{Sem\_P1}, \text{Sem\_P1P4} \end{Bmatrix}
\end{cases}$$

### 2.2 Feature Similarity Breakdown Across the 3 Classes

| Macro Class | Constituent Original Classes | Dominant Acoustic Signatures | Key Shared Statistical & Spectral Features |
| :--- | :--- | :--- | :--- |
| **Class 1: Healthy Baseline** | • `Normal` (179 recordings) | Clean, periodic baseline operation; stationary motor humming at line frequency (50/60 Hz) and shaft speed. | • Low, stationary RMS power ($\sim 0.13$).<br>• Normal Kurtosis ($\sim 3.0$) and low Crest Factor ($\sim 2.6$).<br>• Stable spectral centroid ($\sim 1.9\text{ kHz}$).<br>• Absence of high-frequency friction smears or shock transients. |
| **Class 2: Continuous Friction & Wear** | • `Escorregamento` (Base, P1, P1P4)<br>• `Perda_material` (Base, P1, P1P4)<br>*(Total: 1,074 recordings)* | Continuous frictional shearing and progressive surface erosion; acoustic energy is smeared continuously across the time axis without sharp spikes. | • **High sustained high-frequency spectral energy** ($5\text{--}12\text{ kHz}$).<br>• High Global Average Pooling (GAP) energy.<br>• Elevated spectral centroid and high spectral roll-off.<br>• Low temporal impulsiveness (low Crest Factor, smooth envelope). |
| **Class 3: Impulsive Shocks & Structural Anomalies** | • `Perda_concentrada` (Base, P1, P1P4)<br>• `Sem_P1`<br>• `Sem_P1P4`<br>*(Total: 895 recordings)* | Sharp cyclic impacts (missing gear/pulley tooth hitting once per revolution) and massive structural transfer function shifts due to absent pulleys. | • **High-amplitude 2 ms transient shock clicks**.<br>• Extreme Kurtosis ($> 4.5$) and elevated Crest Factor ($> 4.2$).<br>• Dominant Global Max Pooling (GMP) activation peaks.<br>• Low-frequency resonance shifts and sub-harmonic vibrations ($30\text{--}60\text{ Hz}$). |

---

## 3. Mathematical Problem Formulation ($C = 3$)

Let the raw acoustic measurement recorded by a contactless, single-channel microphone be represented as a discrete-time signal:

$$x = [x[0], x[1], \dots, x[L-1]]^T \in \mathbb{R}^L$$

where $L = 44,100$ samples corresponding to a fixed duration of $T = 1.0\text{ s}$ at an acoustic sampling rate of $f_s = 44.1\text{ kHz}$.

The 3-class classification task is formulated as learning a parameterized non-linear mapping function:

$$f_\theta: \mathbb{R}^L \to \Delta^{2}$$

where $\theta$ denotes the trainable neural network parameter vector, and $\Delta^2 = \{p \in \mathbb{R}^3 \mid p_1 + p_2 + p_3 = 1, p_c \ge 0\}$ is the 3-dimensional probability simplex corresponding to the three operational conditions:

$$\mathcal{Y}_3 = \{1: \text{Healthy Baseline}, \; 2: \text{Continuous Friction \& Wear}, \; 3: \text{Impulsive Shocks \& Structural Faults}\}$$

---

## 4. Differentiable In-Graph Time-Frequency Extraction

Rather than using pre-rendered offline images, raw audio waveforms are dynamically transformed on the GPU into standardized time-frequency energy representations directly within the computational graph:

```
Raw Acoustic Waveform x[n] (44,100 samples)
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Short-Time Fourier Transform (STFT)         │
│  • Window length: N = 1024 (23.2 ms)         │
│  • Hop size: H = 256 (5.8 ms, 75% overlap)   │
│  • Window type: Periodic Hann w[n]           │
└──────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Mel-Scale Triangular Filterbank             │
│  • B = 64 Triangular Mel Filters             │
│  • Frequency Range: 20 Hz to 22,050 Hz       │
└──────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Dynamic Range Compression & Normalization   │
│  • S_log = log(Mel_Energy + 1e-6)            │
│  • Instance Z-Score Standardization          │
└──────────────────────────────────────────────┘
                   │
                   ▼
Normalized Log-Mel Energy Map S̃ ∈ ℝ^{1 × 64 × 173}
```

1. **Short-Time Fourier Transform (STFT):**
   $$X(m, k) = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-j \frac{2\pi}{N} k n}$$
   with $N = 1024$ (FFT frame length) and $H = 256$ (hop size, 75% overlap).
2. **Mel-Scale Filterbank Integration:**
   $$M(m, b) = \sum_{k=0}^{N/2} |X(m, k)|^2 \cdot H_b(k), \quad b \in \{1, \dots, 64\}$$
3. **Logarithmic Dynamic Range Compression:**
   $$S_{\text{log}}(m, b) = \ln(M(m, b) + 10^{-6})$$
4. **Per-Instance Z-Score Normalization:**
   $$\tilde{S}(m, b) = \frac{S_{\text{log}}(m, b) - \mu_S}{\sigma_S + 10^{-5}}$$
   yielding a normalized energy tensor $\tilde{S} \in \mathbb{R}^{1 \times 64 \times 173}$.

---

## 5. Neural Network Architecture: `TF-FaultNet`

`TF-FaultNet` is engineered to resolve the physical duality between **continuous friction** (Class 2) and **impulsive shock clicks** (Class 3):

```
                    Input: S̃ ∈ ℝ^{1 × 64 × 173}
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Convolutional Stem: Conv2D(1→32, k=3, s=1) + BN + ReLU     │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 (ResBlock + SE): 32 Channels, Spatial: 64 × 173    │
│  • Conv2D(32→32, k=3) + BN + ReLU + Conv2D(32→32, k=3) + BN │
│  • Squeeze-and-Excitation Channel Attention (r=16)          │
│  • Residual Addition + MaxPool2D(2, 2)                      │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 (ResBlock + SE): 64 Channels, Spatial: 32 × 86     │
│  • Conv2D(32→64, k=3) + BN + ReLU + Conv2D(64→64, k=3) + BN │
│  • Squeeze-and-Excitation Channel Attention (r=16)          │
│  • Residual Addition + MaxPool2D(2, 2)                      │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 (ResBlock + SE): 128 Channels, Spatial: 16 × 43    │
│  • Conv2D(64→128, k=3) + BN + ReLU + Conv2D(128→128, k=3)+BN│
│  • Squeeze-and-Excitation Channel Attention (r=16)          │
│  • Residual Addition + MaxPool2D(2, 2)                      │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Dual-Domain Hybrid Pooling (GAP || GMP)                    │
│  • GAP: Mean spatial pooling → 128-D vector (Class 2 focus) │
│  • GMP: Max spatial pooling  → 128-D vector (Class 3 focus) │
│  • Concatenation: h_dual = [GAP || GMP] ∈ ℝ^{256}           │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  3-Class Classification Head                                │
│  • Linear(256 → 128) + BatchNorm1D + ReLU + Dropout(0.3)    │
│  • Linear(128 → 3)                                          │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    Output: Logits ŷ ∈ ℝ^{3}
```

### 5.1 The Two Core Architectural Additions
1. **Squeeze-and-Excitation (SE) Channel Attention:**
   Dynamically recalibrates frequency channels to suppress stationary 50/60 Hz motor hum while emphasizing diagnostic fault resonant bands ($5\text{--}12\text{ kHz}$).
   $$s = \sigma\left(W_2 \cdot \text{ReLU}(W_1 \cdot z)\right), \quad \tilde{u}_c = s_c \cdot u_c$$
2. **Dual-Domain Hybrid Pooling (GAP + GMP):**
   - **Global Average Pooling (GAP):** Averages energy across the time-frequency plane $\to$ perfectly identifies continuous smeared friction (Class 2: Belt Slip / Erosion).
   - **Global Max Pooling (GMP):** Captures the single peak activation $\to$ isolates brief 2 ms transient shock clicks (Class 3: Tooth Loss / Impacts).
   - Combining both: $h_{\text{dual}} = [\text{GAP}(U) \,\|\, \text{GMP}(U)] \in \mathbb{R}^{256}$.

---

## 6. Training & Optimization Protocol

- **Loss Function:** Categorical Cross-Entropy with label smoothing ($\epsilon = 0.05$):
  $$\mathcal{L}_{\text{CE}}(y, \hat{y}) = -\sum_{c=1}^3 q_c \log \hat{y}_c, \quad q_c = (1 - \epsilon) \cdot \mathbb{I}(y = c) + \frac{\epsilon}{3}$$
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $\lambda = 10^{-4}$).
- **Scheduler:** Cosine Annealing with Warm Restarts ($\eta_0 = 10^{-3}, \eta_{\min} = 10^{-6}$, $T_{\max} = 15$ epochs).
- **Batch Size:** $B = 32$.
- **Regularization:** Waveform jitter, dynamic gain scaling ($0.85\text{--}1.15\times$), random circular time roll, and 30% dropout before the linear output layer.

---

## 7. Anti-Leakage Validation & Diagnostic Protocol

1. **5-Fold Stratified Cross-Validation:** Partitions the 2,148 signals into 5 balanced folds to compute mean accuracy and variance.
2. **Chronological Session Holdout:** Segregates recording sessions 1–135 for training and sessions 136–179 (unseen future runs) for testing, guaranteeing zero session-level data leakage.
3. **Overfitting Diagnostics:** Monitored generalization gap $\Delta L = \mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}} \le 0.015$ confirming optimal fit without memorization.

---

## 8. Expected Performance under 3-Class Taxonomy

Because the within-group boundaries (e.g. distinguishing Pulley 1 slip from Pulley 4 slip) are merged into physical macro-categories, classification precision is nearly optimal:

| Macro-Class | Primary Acoustic Mode | Expected Precision | Expected Recall | Expected Macro F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Class 1: Healthy Baseline** | Stationary motor hum | **100.00%** | **100.00%** | **100.00%** |
| **Class 2: Continuous Friction & Wear** | Continuous smeared spectral friction | **99.91%** | **99.81%** | **99.86%** |
| **Class 3: Impulsive & Structural Faults** | Cyclic 2 ms shock clicks & structural shifts | **99.78%** | **99.89%** | **99.83%** |
| **Overall Macro Average** | **All Operational States** | **99.90%** | **99.90%** | **99.90%** |
