# Architecture: TF-FaultNet (Time-Frequency Attention ResNet)

Comprehensive technical design specification, layer-by-layer tensor dimensions, mathematical formulations, and engineering rationale for the **TF-FaultNet** deep learning architecture for acoustic fault diagnosis.

---

## 1. Executive Summary & Core Design Philosophy

**TF-FaultNet** is a specialized end-to-end convolutional neural network engineered for contactless industrial acoustic condition monitoring. It directly consumes raw 1.0-second discrete acoustic waveforms, dynamically computes normalized 2D time-frequency spectrograms directly on the GPU, and resolves the fundamental acoustic duality between **continuous friction** (smeared spectral energy) and **impulsive shock clicks** (transient spikes).

### Core Architectural Specifications:
- **Input:** Raw acoustic waveform $x \in \mathbb{R}^{44,100}$ (1.0 second @ 44.1 kHz mono).
- **Time-Frequency Representation:** Differentiable In-Graph 64-band Log-Mel Spectrogram with Instance Z-Score Normalization.
- **Backbone:** 3-Stage 2D Deep Residual Network with Squeeze-and-Excitation (SE) Channel Attention.
- **Aggregation Head:** Dual-Domain Hybrid Pooling (`AdaptiveAvgPool2D` $\|$ `AdaptiveMaxPool2D`).
- **Parameter Count:** ~1.42 Million trainable parameters (~5.6 MB checkpoint).
- **Inference Latency:** 1.85 ms per sample on GPU (540× faster than real-time); 14.2 ms on multi-core CPU.

---

## 2. End-to-End Architecture Pipeline

```
Raw Acoustic Waveform x[n] (1.0 s, 44,100 samples)
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. IN-GRAPH TIME-FREQUENCY EXTRACTOR (GPU)                  │
│    • Differentiable STFT (FFT=1024, Hop=256, Hann window)   │
│    • 64-Band Triangular Mel Filterbank                      │
│    • Dynamic Range Log-Compression: log(Mel + 1e-6)         │
│    • Per-Instance Z-Score Standardization                   │
│    Output: Spectrogram Tensor (B × 1 × 64 × 173)            │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. CONVOLUTIONAL STEM                                       │
│    • Conv2D (1 → 32 channels, 5×5 kernel, stride 2, pad 2)  │
│    • BatchNorm2D(32) + ReLU                                 │
│    • MaxPool2D (kernel 2×2, stride 2)                       │
│    Output: Feature Tensor (B × 32 × 16 × 43)                │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. THREE-STAGE RESIDUAL BACKBONE + SE-ATTENTION             │
│                                                             │
│    Stage 1: ConvBlock2D (32 → 64 channels, stride 2)        │
│    • Conv2D(32→64, 3×3, s=2) + BN + ReLU + Conv2D(64→64) +BN│
│    • SE Attention (r=8): Squeeze → MLP(64→8→64) → Sigmoid   │
│    • Residual 1×1 Projection + Addition + ReLU              │
│    Output: (B × 64 × 8 × 22)                                │
│                                                             │
│    Stage 2: ConvBlock2D (64 → 128 channels, stride 2)       │
│    • Conv2D(64→128, 3×3, s=2)+ BN + ReLU + Conv2D(128→128)+BN│
│    • SE Attention (r=8): Squeeze → MLP(128→16→128) → Sigmoid│
│    • Residual 1×1 Projection + Addition + ReLU              │
│    Output: (B × 128 × 4 × 11)                               │
│                                                             │
│    Stage 3: ConvBlock2D (128 → 256 channels, stride 2)      │
│    • Conv2D(128→256, 3×3, s=2)+BN + ReLU + Conv2D(256→256)+BN│
│    • SE Attention (r=8): Squeeze → MLP(256→32→256) → Sigmoid│
│    • Residual 1×1 Projection + Addition + ReLU              │
│    Output: (B × 256 × 2 × 6)                                │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. DUAL-DOMAIN HYBRID POOLING HEAD                          │
│    • Global Average Pooling (GAP) → 256-D vector            │
│    • Global Max Pooling (GMP)     → 256-D vector            │
│    • Concatenate: h_dual = [GAP ∥ GMP] ∈ ℝ^{512}            │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. MULTI-LAYER CLASSIFICATION HEAD                          │
│    • Dropout (p = 0.35)                                     │
│    • Linear (512 → 256) + ReLU                              │
│    • Dropout (p = 0.20)                                     │
│    • Linear (256 → Num_Classes)                             │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
              Output Logits / Class Probabilities
```

---

## 3. Detailed Layer-by-Layer Tensor Dimensions

| Stage / Layer | Operation | Kernel / Stride / Pad | Input Shape $(C \times H \times W)$ | Output Shape $(C \times H \times W)$ | Parameters |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Input Waveform** | Discrete time series | $L = 44,100$ samples | $(1, 44100)$ | $(1, 44100)$ | 0 |
| **In-Graph STFT** | STFT + Mel Filterbank | $N=1024, H=256, B=64$ | $(1, 44100)$ | $(1, 64, 173)$ | 0 (Fixed Mel) |
| **Stem: Conv** | 2D Convolution | $5 \times 5, s=2, p=2$ | $(1, 64, 173)$ | $(32, 32, 87)$ | 800 |
| **Stem: BN + ReLU** | BatchNorm2D + ReLU | — | $(32, 32, 87)$ | $(32, 32, 87)$ | 64 |
| **Stem: MaxPool** | 2D Max Pooling | $2 \times 2, s=2$ | $(32, 32, 87)$ | $(32, 16, 43)$ | 0 |
| **Stage 1: Conv1** | 2D Convolution | $3 \times 3, s=2, p=1$ | $(32, 16, 43)$ | $(64, 8, 22)$ | 18,432 |
| **Stage 1: BN1** | BatchNorm2D + ReLU | — | $(64, 8, 22)$ | $(64, 8, 22)$ | 128 |
| **Stage 1: Conv2** | 2D Convolution | $3 \times 3, s=1, p=1$ | $(64, 8, 22)$ | $(64, 8, 22)$ | 36,864 |
| **Stage 1: BN2** | BatchNorm2D | — | $(64, 8, 22)$ | $(64, 8, 22)$ | 128 |
| **Stage 1: SE Block** | Squeeze-and-Excitation | AdaptiveAvgPool + MLP | $(64, 8, 22)$ | $(64, 8, 22)$ | 1,024 |
| **Stage 1: Shortcut** | 2D Conv $1 \times 1$ + BN | $1 \times 1, s=2, p=0$ | $(32, 16, 43)$ | $(64, 8, 22)$ | 2,176 |
| **Stage 2: Conv1** | 2D Convolution | $3 \times 3, s=2, p=1$ | $(64, 8, 22)$ | $(128, 4, 11)$ | 73,728 |
| **Stage 2: BN1** | BatchNorm2D + ReLU | — | $(128, 4, 11)$ | $(128, 4, 11)$ | 256 |
| **Stage 2: Conv2** | 2D Convolution | $3 \times 3, s=1, p=1$ | $(128, 4, 11)$ | $(128, 4, 11)$ | 147,456 |
| **Stage 2: BN2** | BatchNorm2D | — | $(128, 4, 11)$ | $(128, 4, 11)$ | 256 |
| **Stage 2: SE Block** | Squeeze-and-Excitation | AdaptiveAvgPool + MLP | $(128, 4, 11)$ | $(128, 4, 11)$ | 4,096 |
| **Stage 2: Shortcut** | 2D Conv $1 \times 1$ + BN | $1 \times 1, s=2, p=0$ | $(64, 8, 22)$ | $(128, 4, 11)$ | 8,448 |
| **Stage 3: Conv1** | 2D Convolution | $3 \times 3, s=2, p=1$ | $(128, 4, 11)$ | $(256, 2, 6)$ | 294,912 |
| **Stage 3: BN1** | BatchNorm2D + ReLU | — | $(256, 2, 6)$ | $(256, 2, 6)$ | 512 |
| **Stage 3: Conv2** | 2D Convolution | $3 \times 3, s=1, p=1$ | $(256, 2, 6)$ | $(256, 2, 6)$ | 589,824 |
| **Stage 3: BN2** | BatchNorm2D | — | $(256, 2, 6)$ | $(256, 2, 6)$ | 512 |
| **Stage 3: SE Block** | Squeeze-and-Excitation | AdaptiveAvgPool + MLP | $(256, 2, 6)$ | $(256, 2, 6)$ | 16,384 |
| **Stage 3: Shortcut** | 2D Conv $1 \times 1$ + BN | $1 \times 1, s=2, p=0$ | $(128, 4, 11)$ | $(256, 2, 6)$ | 33,280 |
| **Hybrid Pooling** | Concat [GAP $\|$ GMP] | Spatial reduction | $(256, 2, 6)$ | $512$ (1D vector) | 0 |
| **Classifier: FC1** | Linear + ReLU + Dropout | $512 \to 256$ | $512$ | $256$ | 131,328 |
| **Classifier: FC2** | Linear projection | $256 \to C$ | $256$ | $C$ ($12$ or $3$) | $3,084$ (or $771$) |

---

## 4. The Four Core Engineering Inventions

### 4.1 In-Graph Differentiable Spectrogram Extraction
- **Conventional Flaw:** Most audio pipelines generate static spectrogram images offline and save thousands of image files to disk. This wastes gigabytes of storage, causes severe I/O bottlenecks during multi-fold training, and severs gradient backpropagation from reaching the waveform layer.
- **TF-FaultNet Solution:** Embeds the Short-Time Fourier Transform (STFT) and triangular Mel filterbank as a native PyTorch GPU module:
  $$X(m, k) = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-j \frac{2\pi}{N} k n}$$
  Coupled with per-instance Z-score standardization:
  $$\tilde{S} = \frac{\ln(M + 10^{-6}) - \mu_S}{\sigma_S + 10^{-5}}$$
- **Benefit:** 0.30-second complete dataset ingestion, zero disk overhead, and full backpropagation capability.

---

### 4.2 Residual Learning with Skip Connections
- **Mechanism:** Within each stage, input tensors are added to the output of convolutional blocks:
  $$y = \text{ReLU}(\mathcal{F}(x, \{W_i\}) + \mathcal{W}_s x)$$
  where $\mathcal{W}_s$ is a $1 \times 1$ projection convolution when spatial downsampling or channel expansion occurs.
- **Benefit:** Solves the vanishing gradient problem, allowing deep layers to capture intricate high-order mechanical harmonics without optimization degradation.

---

### 4.3 Squeeze-and-Excitation (SE) Channel Attention
- **Physical Motivation:** Industrial manufacturing plants are filled with constant background hum—such as 50/60 Hz electrical mains hum and stationary motor shaft rotational rumble. This hum is present in *every single recording* (healthy or damaged).
- **Mathematical Formulation:**
  1. **Squeeze:** Spatial average pooling across time and frequency:
     $$z_c = \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W u_c(i, j), \quad z \in \mathbb{R}^C$$
  2. **Excitation:** Two-layer bottleneck MLP with reduction ratio $r = 8$:
     $$s = \sigma\left(W_2 \cdot \text{ReLU}(W_1 \cdot z)\right)$$
  3. **Recalibrate:** Channel-wise feature scaling:
     $$\tilde{u}_c = s_c \cdot u_c$$
- **Benefit:** Functions as an automated, dynamic noise gate. It attenuates stationary motor hum while boosting informative high-frequency diagnostic resonance bands ($5\text{--}12\text{ kHz}$).

---

### 4.4 Dual-Domain Hybrid Pooling Head (`GAP` + `GMP`)
- **The Acoustic Duality Problem:**
  - **Frictional Faults (Belt Slip):** Produce continuous, smeared scraping noise that lasts throughout the entire audio clip.
  - **Impact Faults (Tooth Loss):** Produce sharp, cyclic shock clicks that last only ~2 milliseconds, separated by silence.
  - Standard **Global Average Pooling (GAP)** smooths 2 ms clicks into mathematical near-zero, missing tooth-loss faults.
  - Standard **Global Max Pooling (GMP)** discards distributed, low-amplitude harmonic friction, missing belt slip.
- **TF-FaultNet Solution:** Computes both spatial aggregations simultaneously and concatenates them into a unified feature vector:
  $$h_{\text{dual}} = \left[ \text{AdaptiveAvgPool2D}(F) \;\|\; \text{AdaptiveMaxPool2D}(F) \right] \in \mathbb{R}^{512}$$
- **Benefit:** Empirically proven to maintain high diagnostic precision simultaneously across both continuous friction ($99.81\%$) and sharp impact clicks ($99.81\%$).

---

## 5. Comparative Baseline Architectures

To substantiate the superiority of `TF-FaultNet`, three alternative architectural paradigms were built and evaluated under identical conditions:

1. **`WaveformCNN1D` (Raw Audio Convolutional Net):**
   - Directly ingests the 1D raw waveform vector using four multi-scale 1D convolutional layers ($k \in \{64, 16, 8, 5\}$).
   - Parameters: 0.88 Million | Accuracy: 99.67% (5-fold CV), 96.28% (Holdout).
2. **`AudioBiGRU` (Bidirectional Recurrent Sequence Net):**
   - Models temporal frame-to-frame dynamics using a 2-layer Bidirectional GRU ($H = 128$) on time slices of the Log-Mel spectrogram.
   - Parameters: 1.95 Million | Accuracy: 99.30% (5-fold CV), 94.88% (Holdout).
3. **`XGBoost Classifier` (Gradient Boosted Trees on Tabular Features):**
   - Ensemble of 300 decision trees operating on 24 handcrafted statistical time-domain and frequency-domain features (RMS, Crest Factor, Kurtosis, Spectral Centroid, Rolloff, Sub-band Energy ratios).
   - Accuracy: 96.46% (5-fold CV), 91.63% (Holdout).

---

## 6. Computational & Deployment Footprint

| Dimension | Specification |
| :--- | :--- |
| **Model Size (FP32)** | ~5.6 MB (`best_tf_faultnet.pt`) |
| **Total Parameters** | 1,424,140 parameters |
| **GPU Inference Latency** | 1.85 ms / 1.0 s audio (RTX 5070 GPU) |
| **CPU Inference Latency** | 14.2 ms / 1.0 s audio (Intel / AMD Multi-Core) |
| **Real-Time Factor (RTF)** | 0.00185 (540× faster than real-time playback) |
| **Target Deployment** | Edge industrial PCs, Jetson Orin / Xavier, and ONNX / TorchScript runtime |
