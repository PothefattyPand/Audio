# TF-FaultNet: three-class acoustic fault classification

This research project classifies one-second machine recordings into healthy baseline, continuous friction/wear, and impulsive/structural conditions. The 2,148 WAV recordings come from 12 source-condition folders. The current experiment uses a frozen **file-level** split: 1,374 training, 344 validation, and 430 test recordings.

The saved three-class model correctly classifies **424/430 test recordings: 98.60% accuracy and 0.98589 macro F1**. These metrics were checked against cached predictions and original audio. Acquisition/session independence is unverified; architectural superiority and deployment readiness are not established.

## Start here

- [Research methodology](docs/THREE_CLASS_RESEARCH_METHODOLOGY.md): current task, architecture, training and evaluation protocol.
- [Supervisor review](docs/SUPERVISOR_REVIEW_2026-09-09.md): verified results, methodological issues and research priorities.
- [Problem specification](docs/PROBLEM_SPECIFICATION.md), [EDA record](docs/EDA_PREPROCESSING.md), [error analysis](docs/ERROR_ANALYSIS.md), and [limitations](docs/LIMITATIONS_AND_ETHICS.md).
- [Cleanup record](docs/CLEANUP_2026-09-10.md): removed files and recovery information.

## Active files

| Path | Purpose |
|---|---|
| `Base_de_Dados/` | Immutable original WAV files |
| `data/splits.csv` | Frozen `split_multiclass_3class` membership |
| `metadata/class_mapping_3class.json` | Source-folder mapping |
| `configs/experiment_3class.json` | Recorded settings; trainer still hardcodes them |
| `src/` | Models, audio transforms, dataset and manifest validation |
| `train_3class_pipeline.py` | Three-class training and export |
| `train.py` | Shared epoch helpers and historical 12-class CV entry point |
| `evaluate_3class_test_set.py` | Locked three-class model evaluation |
| `scripts/baseline_comparison_3class.py` | Classical baseline pipeline |
| `scripts/supervisor_audit.py` | Artifact verification without training or fresh inference |
| `results/heldout_test_evaluation_report_3class.json` | Saved primary result |
| `results/checkpoints/best_tf_faultnet_3class*` | Three-class weights and exports |
| `tests/` | Manifest and signal-statistic tests |

## Environment and checks

Install project dependencies into your chosen environment:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/supervisor_audit.py
```

The requirements specify minimum versions, not a fully locked environment. The latest review ran 11 tests successfully; extractor tests were blocked by missing SciPy. Full training and inference were not reproduced during review.

The existing test set has already been inspected. Further model selection belongs on development data. Training/evaluation scripts write fixed output paths, so preserve prior artifacts before an intentional new experiment.

## Historical experiments

The 12-class training, held-out evaluation, ablation, fit-diagnostic, noise and explainability scripts remain for provenance, alongside their numerical results and checkpoints. They use different experimental protocols and contain known limitations documented in the supervisor review. Their results must not be substituted for the three-class result or treated as verified independent-session performance.

Use empirical ROC plots under `results/visualizations/`; redundant smoothed display variants were removed. ONNX files contain the spectrogram backbone, while TorchScript exports include waveform preprocessing. Exported files alone do not establish deployment readiness.
