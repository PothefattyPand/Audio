# Limitations, Deployment Scope, and Ethical Considerations
## TF-FaultNet Acoustic Fault Diagnosis System

> Required by CLAUDE.md §2 and ML Research Best Practices.

---

## 1. Technical Limitations & Scope

### 1.1 Single-Site Laboratory Test Rig
- **Environmental Homogeneity**: All acoustic recordings were acquired on a single laboratory test rig under controlled mechanical operating conditions. The background acoustic transfer function, room reverberation, and transducer placement remained stationary throughout the campaign.
- **Cross-Rig Transferability**: Real-world industrial installations vary in housing geometry, motor types, ambient acoustic noise (nearby machinery, HVAC, foot traffic), and foundation stiffness. Direct zero-shot deployment onto non-identical machinery without domain adaptation or fine-tuning will likely experience performance degradation.

### 1.2 Unimodal Sensory Input (Audio-Only)
- TF-FaultNet processes monaural 1-second 44.1 kHz acoustic pressure signals. In industrial condition-monitoring standards (e.g., ISO 10816 / ISO 20816), vibration accelerometers, motor current signature analysis (MCSA), and infrared thermography are frequently combined. Relying solely on acoustics makes the system susceptible to high-intensity acoustic interference if not isolated or denoised.

### 1.3 Stationary Rotational Speed
- Data were gathered under nominal, constant rotational speeds. Non-stationary speed variations (acceleration/deceleration ramps) introduce frequency smearing in the Short-Time Fourier Transform (STFT), which may require order tracking or wavelet packet transforms for optimal resolution under varying RPM.

---

## 2. Safety & Industrial Deployment Risks

### 2.1 False Negative vs False Positive Trade-off
- In mechanical health monitoring, **False Negatives (Type II errors)**—failing to detect an active structural fault—carry the highest risk, potentially resulting in catastrophic mechanical breakdown, factory downtime, or human injury.
- TF-FaultNet achieves **100% precision on the Healthy_Baseline class**, meaning zero faulty test samples were misdiagnosed as normal. All 6 misclassifications were confined to adjacent fault categories (Continuous Friction vs Impulsive Shocks).
- **Recommendation**: In life-critical machinery, TF-FaultNet should operate in a **Human-in-the-Loop (HITL)** triage framework, flagging suspect segments for vibration analyst inspection rather than executing autonomous shutdown without verification.

### 2.2 Transducer Failure Modes
- Microphone saturation, clipping, cable disconnection, or sensor degradation could present anomalous acoustic spectra. An out-of-distribution (OOD) detector or signal integrity sanity check (e.g., clipping detection, SNR thresholding) must precede inference in production.

---

## 3. Ethical and Privacy Considerations

| Dimension | Assessment |
|-----------|------------|
| **Human Subjects & PII** | Zero human subjects or personally identifiable information (PII). Audio data consists solely of rotating mechanical machinery sounds. |
| **Acoustic Surveillance Privacy** | If acoustic sensors are installed on factory floors where human operators communicate, there is an incidental privacy risk of capturing human speech. To mitigate this: (1) Transducers should be directional contact/near-field microphones focused on the bearing housing; (2) Edge inference via the exported ONNX backbone discards raw audio immediately after computing log-mel frames in volatile memory. |
| **Dual Use** | Low risk. The application is strictly industrial predictive maintenance and equipment safety. |
| **Environmental Footprint** | Low computational footprint. TF-FaultNet has only 1.25M parameters and infers in 1.48 ms per 1-second sample on CPU, minimizing edge energy consumption. |

---

## 4. Reproducibility Statement

- **Frozen Split**: All experiments are strictly deterministic and reproducible via the frozen split manifest `data/splits.csv` (Seed 42).
- **Environment & Checkpoints**: Exact model weights (`results/checkpoints/best_tf_faultnet_3class.pt`), ONNX graphs, TorchScript JIT models, and JSON configuration files (`configs/experiment_3class.json`) are preserved in the repository.
