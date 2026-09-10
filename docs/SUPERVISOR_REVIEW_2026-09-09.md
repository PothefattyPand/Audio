# Supervisor review: TF-FaultNet acoustic fault diagnosis

Reviewed 9 September 2026. This is a repository and saved-evidence review, not a new training run, new test evaluation, or external literature review. Raw data, existing checkpoints, and historical results were preserved. Reproducible artifact checks are in `scripts/supervisor_audit.py`; their output is `outputs/supervisor_review/audit.json`.

## Assessment

The project contains a working research prototype with a strong observed three-class classification result. The current evidence supports **98.60% accuracy and 0.98589 macro F1 on the existing file-level test partition**. It does not yet establish generalization to independent machines/sessions, statistical superiority of the proposed architecture, or production readiness. The largest weaknesses are the experimental protocol and unsupported reporting, rather than an obvious error in the central CNN training loop.

An applied thesis could be built around this work after substantial evidence reconciliation. I would request major revisions before approving its present scientific claims. I would not approve the current claims of essential architectural components, guaranteed absence of leakage, or proven industrial robustness.

## 1. Actual results and their scope

The current task maps 12 source conditions into three mutually exclusive categories. There are 2,148 one-second recordings: 179 healthy, 1,074 friction/wear, and 895 impulsive/structural. The frozen manifest assigns 1,374 training, 344 validation, and 430 test files. This is file-level stratification; the repository does not establish verified acquisition groups.

| Model | Saved three-class test accuracy | Saved macro F1 |
|---|---:|---:|
| Logistic regression | 70.70% | 0.75082 |
| Random forest | 96.51% | 0.97055 |
| XGBoost | 96.74% | 0.97615 |
| TF-FaultNet | **98.60%** | **0.98589** |

Sources: `results/baseline_comparison_3class.json` and `results/heldout_test_evaluation_report_3class.json`. These are historical results, not newly retrained comparisons. The numerical advantage over XGBoost is 1.86 percentage points in accuracy and 0.974 percentage points in macro F1. It is not, by itself, a statistically established improvement.

The CNN's cached ONNX predictions reproduce this confusion matrix, with actual classes in rows:

| Actual / predicted | Healthy | Friction/wear | Impulsive/structural |
|---|---:|---:|---:|
| Healthy | 35 | 0 | 1 |
| Friction/wear | 0 | 210 | 5 |
| Impulsive/structural | 0 | 0 | 179 |

There are six mistakes: one healthy false alarm and five incorrect fault-subtype assignments. No faulty recording is predicted healthy in this sample. That observation cannot guarantee future fault-detection sensitivity. Four of the six saved mistakes have predicted-class probabilities above 0.90; confidence calibration deserves development-set analysis.

An illustrative 95% Wilson interval for recording-level accuracy is **96.99%–99.36%**, assuming independent Bernoulli observations. This assumption is unverified; the interval does not account for session dependence or model-selection uncertainty. Healthy recall is 35/36, so strong healthy-class claims rely on a small support.

The three-class report gives macro ROC AUC 0.999544 and macro average precision 0.999374. The evaluator computes the former by integrating an interpolated macro ROC curve; the simple mean of its saved per-class AUCs is approximately 0.999466. Specify the averaging convention. The field named PR-AUC is computed with `average_precision_score`, so call it average precision rather than implying trapezoidal PR integration.

The saved training history selects epoch 14 by minimum validation loss, 0.19915. Its validation accuracy is 98.84%, with macro F1 0.99147. Epoch 13 has higher validation F1; this is not a checkpoint bug because the implemented criterion is validation loss. Keep the selection criterion explicit. Final training accuracy reaches 100%; validation remains high on this partition. This does not establish absence of memorization or external generalization.

Older results are separate experiments: `results/summary.json` reports 12-class CV accuracy **99.767% ± 0.208 percentage points**, while the separate 12-class held-out report gives **99.535%**. Neither is the current three-class thesis result. The README's 99.81% headline disagrees with the saved CV summary.

## 2. Findings requiring correction before a defense

### P1 — Session generalization is not established, and the ablation holdout selects epochs

`ablation_study.py:232` builds a block split from the first 135 versus remaining files within each class. Files come from ordinary lexicographic `sorted(glob.glob(...))`, not verified timestamps or session identifiers. Names such as `_1`, `_10`, `_100`, and `_2` do not establish chronological ordering. The same issue occurs in `diagnose_fit.py`.

More seriously, `train_eval_model` evaluates this block partition every epoch and returns its best accuracy; the call at `ablation_study.py:368` labels that result a session holdout. This is validation-based model selection on the reported partition, not untouched final-test assessment. The five-fold routines also report the best epoch selected on each scored fold (`train.py`), producing selection-biased CV performance estimates.

Required action: obtain real acquisition metadata and distinguish machine, run/session, source recording, and clip. Use group-aware development partitions where possible, with checkpoint selection inside the development data and an independent final test. Until metadata exist, describe the current result as file-level classification with unresolved dependence. A numerical filename sort alone would not resolve unknown acquisition groups.

### P1 — Published-looking result tables disagree with saved experiments

`README.md`, `ABLATION_STUDY_REPORT.md`, `PROJECT_PITCH_AND_TECHNICAL_REPORT.md`, architecture/dissertation documents, and the three PDF generators repeat 99.81% CV, 98.30% session accuracy, and large ablation degradations. The PDF generators embed these values directly in text/tables rather than reading validated experiment records.

The actual `results/ablations/ablation_study_results.csv` contains:

| Variant | Saved 12-class CV accuracy | Saved block-validation accuracy |
|---|---:|---:|
| Full model | 99.6740% | 99.2424% |
| No SE | 99.6742% | 98.4848% |
| Average pooling only | **99.7673%** | 98.6742% |
| Max pooling only | 99.5811% | 98.4848% |
| No per-recording normalization | 99.6277% | 98.2955% |
| No residual connections | 99.5811% | 98.6742% |

Average-only pooling slightly beats the full model in CV. Removing SE produces virtually the same CV accuracy. The observed block scores favor the full model, but have the selection limitation above. These observations do not demonstrate that dual pooling is essential or that SE suppresses a particular physical noise source.

`ablation_study.py:385` calculates deltas against hardcoded 0.9977 and 0.9716 instead of its actual full-model row. Consequently even the full-model row has nonzero deltas against itself. `docs/ABLATION_3CLASS_NOTE.md` presents estimated three-class ablation scores and then draws proof-like conclusions. Estimated values must not be presented as completed experiments.

Required action: reconcile tables against immutable run outputs, remove unsupported estimates, calculate deltas against the matching control, and conduct actual three-class component ablations under a common development protocol. Preserve superseded reports as historical evidence.

### P1 — Noise comparison uses incompatible training provenance

`run_noise_snr_benchmark.py:89` makes a new random 80/20 partition and loads the best cross-validation checkpoint from `train.py`. That checkpoint is not tied to the new test membership and may have trained on its recordings. The baseline networks are instead freshly trained on the new training partition. This is not a controlled same-split comparison. Each model also receives a separately generated noise draw, and only one draw is used per condition. If the proposed checkpoint is absent, the script announces training but leaves the proposed model untrained.

Even treating the saved numbers only descriptively, they show a major weakness: at +20 dB, TF-FaultNet falls to **44.88%** accuracy, while the 1D CNN scores **98.60%**. At +10 dB the scores are 16.28% and 68.14%. Highlighting only TF-FaultNet's 22.79% at 0 dB hides this much larger failure. All of these are 12-class observations, not measurements of the locked three-class model.

Required action: fail when checkpoints are missing, verify training/test membership, apply identical corruption arrays to all models, repeat corruption seeds, and evaluate realistic noise after the protocol is locked. Do not claim industrial noise robustness from this benchmark.

### P1 — Statistical claims lack an auditable run lineage

`profile_and_stats.py:28` hardcodes five fold accuracies for every model and hardcodes session results. `train.py` saves aggregate summaries, not the fold records consumed by this calculation. The asserted pairing is not programmatically verified. The supplied TF-FaultNet array's spread also differs from the saved benchmark standard deviation.

Repeated training folds overlap, so treating five scores as independent paired observations requires care. Small-fold tests and unadjusted multiple comparisons do not support broad claims of significance. Undefined test statistics are also converted to zero, potentially producing a misleading p-value of 0.

Required action: store run IDs, split hashes, seeds, fold-level metrics and per-recording predictions; select an uncertainty/comparison procedure appropriate to the actual independent unit. Never replace undefined p-values with zero. The current code does not establish statistically significant superiority for the three-class result.

### P2 — Baseline fairness and provenance are incomplete

The baseline script fits logistic regression directly to descriptors with very different scales, with no training-fitted scaler or recorded validation search. This can disadvantage the linear baseline. Fixed defaults also provide limited evidence of the best reasonable classical competitor. The CNN comparison row copies historical test metrics and hardcodes rounded validation values.

The current feature cache **does contain corrected ZCR values**: this audit checked all train/validation/test rows against raw recordings. RMS values also match manifest ordering, and cached values are finite. Older documentation saying corrected baselines remain wholly pending is therefore not a reliable description of the cache's current contents. However, the cache stores only three matrices, without source IDs, extractor version, or manifest/data hashes; the saved baseline reports are not cryptographically tied to that cache or fitted model. Do not assume their provenance from cache contents alone.

Required action: use a train-fitted scaler for logistic regression, make development-only model choices, preserve fitted models/predictions and cache lineage, and include a trivial majority comparator. Do not use the baseline script's test-based complexity verdict as permission for subsequent architectural tuning.

### P2 — Explainability and error narratives overstate what the code measured

`GradCAM.generate` returns the requested target class as its third value. The caller explicitly requests the known class, then uses that return value as `pred_c` and labels it "Pred" (`explainability_and_attention_analysis.py:176`, `:202`). A mistaken prediction could therefore be displayed as correct. Report attribution target and actual logits argmax separately.

The selected late feature map is only 2×6 spatially before interpolation to 64×173. Upsampling cannot supply fine temporal/frequency localization absent from the original map. SE channels are learned feature maps, not one-to-one physical frequency bins. Attractive heatmaps alone do not establish a causal mechanical mechanism.

`scripts/error_analysis_3class.py` inserts explanations about resonance, hum masking and attention into a static Markdown template. It does not perform a causal acoustic investigation. Its statement that all six errors lie between fault categories contradicts the healthy false alarm. Similar wording appears in `docs/LIMITATIONS_AND_ETHICS.md`.

Required action: correct prediction labels and error counts, mark proposed physical explanations as hypotheses, and validate them with domain review or controlled signal experiments.

### P2 — Reproducibility and deployment claims exceed the implementation

The three-class trainer hardcodes settings rather than reading `configs/experiment_3class.json`; checkpoints are weights-only and overwrite fixed paths. Seeds are set, but deterministic execution and environment locking are incomplete. `requirements.txt` contains lower bounds, not a reproducible package lock. The main evaluator lacks the model/archive/manifest hash checks implemented later in the ROC script.

Audio preloading discards the actual sample rate and pads/trims to 44,100 samples without validating or resampling it. That is harmless for the audited local files, all at 44.1 kHz, but would silently misinterpret other-rate deployment audio. ONNX exports accept precomputed spectrograms, not raw audio. MAC/FLOP hooks count convolution/linear operations and omit STFT, Mel projection and other operations. Batched average milliseconds per recording are not batch-one request latency; a one-second acquisition window also precedes prediction.

The 12-class trainer loads CPU tensors into an existing potentially CUDA model, but does not move that model to CPU before tracing with CPU dummy input. Its export exceptions are caught while the script still announces success. Separate export code constructs a fresh CPU model and avoids this particular issue.

Required action: one config-driven run entry point, versioned output directories, complete provenance, checked input contracts, export parity tests, and deployment measurements with clearly defined timing boundaries. Exported files alone are not a monitoring/deployment study.

## 3. What the implementation gets right

The core three-class model has appropriate output/loss alignment: three logits, cross-entropy with label smoothing, softmax for reporting, and argmax for decisions. Training uses `model.train()`; validation uses `eval()` and no gradients, correctly handling BatchNorm and dropout. Augmentation is training-only. The spectrogram normalization is per recording and does not fit population parameters on the test set. Computing a fixed test spectrogram is not itself preprocessing-fit leakage.

The convolutional inductive bias is reasonable for local patterns in time–frequency representations. This is an implementation rationale, not proof that it beats simpler models. The transform is fixed; learned feature extraction occurs in the CNN. The code uses natural logarithms, despite a comment calling them dB.

Analytical shapes for input `(B,44100)` are `(B,1,64,173)` after log-Mel extraction, `(B,32,16,43)` after the stem including pooling, then `(B,64,8,22)`, `(B,128,4,11)`, `(B,256,2,6)`, a 512-dimensional concatenated pooling vector, 256 hidden units, and three logits. Counting convolution, dense, SE and BatchNorm trainable parameters from source gives **1,361,379** for the three-class model. This is an analytical check; a fresh Torch forward pass was not available.

The frozen-manifest loader checks unique existing paths, valid labels/partitions, and class presence. Stronger mapping-consistency, group and content-hash validation would improve it. The later ROC workflow verifies hashes and keeps measured scores separate from explicitly labeled display smoothing. That is a useful pattern for the rest of the reporting system.

## 4. Review coverage and verification limits

Reviewed the shared modules, all training/evaluation entry points, EDA, baselines, ablations, diagnostics, noise benchmark, profiling/statistics, explainability, protocol generator, error/ROC/report generators, configuration, tests, notebooks, and saved numerical evidence. Python AST inventory/syntax checks covered 31 existing Python files plus the newly added audit script. This does not mean every graphical layout or historical run was executed.

Both notebooks have no saved execution outputs. `kaggle_run_audio_fault_model.ipynb` duplicates an older 12-class model without the shared per-recording normalization and with different augmentation/training settings. It is not an exact reproduction of the current pipeline. `kaggle_run_custom_model.ipynb` targets external OBD-II/EngineFaultDB code and datasets absent from this repository; it is not evidence for Base_de_Dados performance.

Read-only artifact checks verified all 2,148 WAVs as readable mono 16-bit PCM, 44.1 kHz, 44,100 samples; found no exact duplicate PCM payloads and no full-scale integer samples. The normalized range is approximately -0.999725 to 0.998413. These checks do not establish independence, absence of near-duplicates, absence of all anomalies, or privacy clearance.

All 430 archived test waveforms and labels match raw audio and the manifest. Cached prediction fingerprints match the present ONNX model and archive. Accuracy and macro F1 recomputed from cached probabilities match the historical report. This verifies internal consistency; it does not prove how the checkpoint was trained or independently regenerate its predictions.

On the available bundled Python, **11 tests passed**, while the extractor-test module could not import because SciPy is missing. The default system Python also lacks NumPy. The bundled runtime lacks Torch, scikit-learn and ONNX Runtime. Therefore no claim is made of a full passing suite, fresh inference, export parity, or reproduced training. No dependency installations or model retraining were necessary for this review.

`AGENTS.md` contains stale project-specific references to binary two-logit classification and a participant-level binary split column. The actual current code/config/manifest implement three classes and file-level splitting. Resolve these contradictory project instructions before future experiments; this audit does not silently change the intended scientific task.

## 5. Supervisory priorities

1. Reconcile thesis, README and generated tables against saved evidence; remove estimates and unjustified guarantees. Establish one canonical three-class result table.
2. Obtain acquisition provenance and a domain justification for the three-class mapping. If groups cannot be recovered, explicitly restrict the thesis to the current file-level setting.
3. Make experiments config-driven and preserve run-specific code/config/split/checkpoint hashes, environments, histories and predictions. Repair the identified reporting/export bugs.
4. On development data, establish fair classical and simple neural baselines; run repeated-seed three-class component ablations. Treat existing test results as already inspected, not a reusable tuning resource.
5. Lock the final method before independent acquisition-level testing and robustness assessment. Complete uncertainty, failure and runtime analysis appropriate to the intended use.

Suitable thesis wording now:

> On a frozen stratified file-level partition of 2,148 laboratory acoustic recordings mapped to three health-condition categories, TF-FaultNet correctly classified 424 of 430 test recordings, achieving 98.60% accuracy and 0.98589 macro F1. These results demonstrate strong discrimination within the available dataset. Unknown acquisition grouping, incomplete comparative run provenance, and the absence of independent machine/session evaluation limit claims about architectural superiority and deployment generalization.

The defensible contribution is currently an implemented and evaluated acoustic-classification system. An original algorithmic contribution, the necessity of individual components, and broad superiority require additional controlled evidence and a literature comparison.
