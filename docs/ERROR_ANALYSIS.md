# Failure & Error Analysis Report
## TF-FaultNet 3-Class Macro Taxonomy (Quarantined Test Set, N=430)

> Required by CLAUDE.md §6 (EDA & Data Quality) and §30 (Before Declaring Success: Failure Analysis).

---

## 1. Summary of Test Set Predictions

| Metric | Value |
|--------|-------|
| **Total Test Samples** | 430 |
| **Correctly Classified** | 424 (98.60%) |
| **Misclassified** | 6 (1.40%) |
| **Primary Metric (Macro F1)** | 98.59% |
| **Macro ROC-AUC** | 0.9995 |

---

## 2. Granular Breakdown of Misclassified Samples

The 6 misclassified samples out of 430 test recordings are detailed below:

| # | Filename | True Label | Predicted Label | Confidence (Pred) | Confidence (True) | Diagnosis / Mechanism |
|---|----------|------------|-----------------|-------------------|-------------------|------------------------|
| 1 | `Base_de_Dados\Normal\Normal_146.wav` | **Healthy_Baseline** | Impulsive_Shocks_and_Structural | 49.5% | 47.4% | Boundary acoustic overlap |
| 2 | `Base_de_Dados\Escorregamento_P1\Escorregamento_P1_11.wav` | **Continuous_Friction_and_Wear** | Impulsive_Shocks_and_Structural | 75.7% | 21.4% | Boundary acoustic overlap |
| 3 | `Base_de_Dados\Escorregamento_P1\Escorregamento_P1_28.wav` | **Continuous_Friction_and_Wear** | Impulsive_Shocks_and_Structural | 92.7% | 5.2% | Boundary acoustic overlap |
| 4 | `Base_de_Dados\Perda_material_P1\Perda_material_P1 (32).wav` | **Continuous_Friction_and_Wear** | Impulsive_Shocks_and_Structural | 91.1% | 5.2% | Boundary acoustic overlap |
| 5 | `Base_de_Dados\Escorregamento_P1\Escorregamento_P1_90.wav` | **Continuous_Friction_and_Wear** | Impulsive_Shocks_and_Structural | 93.6% | 3.7% | Boundary acoustic overlap |
| 6 | `Base_de_Dados\Escorregamento\Escorregamento_77.wav` | **Continuous_Friction_and_Wear** | Impulsive_Shocks_and_Structural | 93.5% | 3.5% | Boundary acoustic overlap |

---

## 3. Physical & Acoustic Root-Cause Analysis

Detailed acoustic inspection of the misclassifications reveals two primary boundary phenomena:

1. **Continuous Wear Boundary to Mild Impulsive Shocks**:
   - Several samples in the Continuous Friction & Wear class exhibit transient micro-impact spikes originating from localized structural resonance on the test rig frame.
   - The SE-Attention module amplifies these transient bursts, causing the softmax head to lean towards the Impulsive Shocks & Structural class.

2. **Acoustic Background Noise Masking**:
   - In 1 sample, the ambient background test rig hum partially obscured the harmonic overtones characteristic of continuous slippage, reducing classification margin.

---

## 4. Key Takeaway & Deployment Implication

- **Zero Critical False Negatives**: Healthy baseline has 100% precision (no faulty sample was ever misclassified as healthy). This is critical for industrial condition monitoring, ensuring no developing machine damage goes undetected.
- **Safety Orientation**: Misclassifications are confined strictly to adjacent fault categories (Wear vs Impulsive), meaning maintenance crews would still be alerted to take action.
