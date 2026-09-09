# General ML Research Pipeline and Debugging Guide

This guide generalizes the techniques used to build, debug, validate, and
report the eye-tracking classification pipelines. It is intended to be reused
for other supervised machine-learning projects, especially small datasets.

## 1. Start with a problem specification

Before writing model code, record:

- Task: what must be predicted or discovered?
- Input `X`: which measurements are available at prediction time?
- Target `y`: what is the label, and how is it defined?
- Unit of analysis: person, patient, image, session, trial, window, or row?
- Task type: regression, binary, multiclass, multilabel, or unsupervised?
- Primary metric and secondary metrics.
- Baseline that a simple method must beat.
- Data source and provenance.
- Split strategy and grouping variable.
- Leakage risks, ethical risks, and practical constraints.

The unit of analysis controls the entire pipeline. If the final prediction is
one result per participant, every window, trial, and row from that participant
must remain in the same split.

## 2. Audit the data before modeling

Create a machine-readable inventory before training:

1. List every file, extension, size, and source.
2. Count samples and unique entities.
3. Inspect feature names, data types, ranges, missingness, duplicates, and
   constant columns.
4. Inspect target counts and class imbalance.
5. Identify subject, site, session, batch, and time structure.
6. Check whether the claimed modality is actually present. Do not assume that
   `.png`, `.npy`, CSV, or a column named `image` means image data.
7. Inspect representative records and array shapes.
8. Record exclusions and why each exclusion is valid.

For arrays, verify shape, dtype, channel order, value range, and whether the
first dimension is time, samples, channels, height, or width. For tabular data,
verify that each row represents the intended unit.

Keep raw data immutable. Write audits and derived data to separate locations.

## 3. Prevent leakage at the split boundary

Choose the split before fitting anything learned:

- IID rows: random split, stratified when appropriate.
- Repeated entities: group split by person, patient, session, site, or object.
- Time-dependent data: chronological split.
- Small development sets: cross-validation on development data only.

Create a deterministic split manifest containing the entity ID and split name.
Treat it as the single source of truth. Never hardcode IDs in model scripts.

Assert all of the following:

- train, validation, and test entity sets are pairwise disjoint;
- no duplicate or near-duplicate entity crosses a boundary;
- all derived windows inherit the parent entity split;
- every included row has exactly one split;
- the saved manifest still matches the manifest used during evaluation.

The test set is sealed. Do not use it for preprocessing fit, feature
selection, threshold selection, hyperparameter tuning, architecture choice,
early stopping, or debugging decisions.

## 4. Build a defensible preprocessing path

Separate operations into two categories:

- Stateless operations: renaming, fixed unit conversion, and deterministic
  parsing may be applied consistently where justified.
- Learned operations: scaling, imputation statistics, feature selection,
  dimensionality reduction, vocabulary construction, and learned transforms
  must be fitted on training data only.

For each fold or split:

1. Fit the preprocessing object on the training entities.
2. Transform training and validation data with that fitted object.
3. Save the fitted object, feature names, feature order, and configuration.
4. During final evaluation, load the object and call `transform()` only.

Keep feature names attached through preprocessing. Assert the saved feature
order exactly matches the model input order.

## 5. Establish simple baselines first

Use a baseline ladder:

1. Majority or prevalence baseline.
2. Simple statistical baseline, such as logistic or linear regression.
3. Established classical models appropriate for the data.
4. Small neural model or domain-specific model.
5. More complex model only when the simpler models show a specific limitation.

Use the same split, preprocessing rules, target definition, and evaluation
metrics for every model. A complex model is not justified by its complexity;
it is justified by reproducible evidence that it addresses an observed need.

## 6. Select features and hyperparameters correctly

Feature selection is model selection. Perform it inside the training portion
of each fold. A safe cross-validation pattern is:

```text
for each development fold:
    rank/select features using fold-training entities only
    fit preprocessing using fold-training entities only
    fit the candidate model
    evaluate on fold-validation entities
select configuration using development validation results
refit the selected configuration on all development entities
evaluate once on the sealed test set
```

Choose metrics before looking at test results. Use validation performance for
hyperparameters, early stopping, dropout, regularization, architecture, and
classification thresholds.

Prefer a small, interpretable search when data are scarce. Save every trial's
configuration, seed, validation metrics, best epoch, and checkpoint identity.

## 7. Match model, output, loss, and metric

Check the full chain:

- Regression: linear output; MSE or MAE as justified; RMSE, MAE, and R2.
- Binary classification: two logits with cross-entropy, or one logit with
  binary cross-entropy; make the choice explicit.
- Multiclass classification: one logit per mutually exclusive class and
  cross-entropy; prediction is argmax.
- Multilabel classification: independent sigmoid outputs and binary
  cross-entropy per label; do not use softmax.

Verify tensor shapes, class order, label encoding, probability meaning, and
threshold policy with a tiny hand-checked example before a long run.

## 8. Diagnose training from raw evidence

Log, at minimum, raw per-epoch train loss, validation loss, validation metric,
learning rate, and the selected/best epoch. Never fabricate, redraw, or
hand-tune a curve or metric. A smoothed trace may be an overlay, but the raw
trace must remain visible and authoritative.

Use these symptoms to guide the next experiment:

| Observation | Likely causes | Checks or fixes |
|---|---|---|
| Train and validation both poor | underfitting, bad features, wrong labels, optimization failure | verify labels/shapes, try a simpler baseline, inspect scaling and learning rate |
| Train improves while validation worsens | overfitting, excessive capacity, correlated samples | reduce capacity, increase regularization, remove duplicate/overlapping samples, use entity-level selection |
| Both losses oscillate | learning rate too high, unstable scaling, small batches | lower learning rate, inspect ranges, test gradient norms and batch size |
| Loss becomes NaN | invalid values, exploding gradients, unsafe operations | check finite values, normalize inputs, clip gradients, lower learning rate |
| Validation metric jumps in large steps | validation set too small | report uncertainty; do not over-interpret one epoch or one child |
| Window metric looks strong but entity metric is weak | correlated windows or entity imbalance | select and report at the entity level |
| A feature is unusually predictive | confound, label leakage, acquisition artifact | audit its meaning, timing, availability, and subgroup behavior |

For sequence data, overlapping windows can inflate the apparent sample count and
encourage memorization. Keep the entity split fixed, make stride explicit, and
consider non-overlapping or otherwise justified sampling. Model selection and
early stopping should use the final unit of analysis when possible.

## 9. Use domain and ethics checks

Separate predictive signal from undesirable shortcuts. A quality or compliance
feature may predict the label without representing the intended phenomenon.
Train a comparison model when useful, document the confound, and decide
whether the feature is acceptable before exporting the primary model.

Check for direct leakage such as post-outcome variables, clinician scores that
exist only for one class, metadata proxies, duplicated records, and collection
artifacts. Report limitations rather than hiding inconvenient findings.

For health, education, or child data, document privacy, consent, subgroup
performance, potential harms, and whether the model is suitable for research
only or any operational use.

## 10. Evaluate once on the sealed test set

After the model, preprocessing, threshold, and reporting plan are locked:

1. Load the exported checkpoint and preprocessing state.
2. Assert test entities are absent from train and validation.
3. Assert feature order and split manifest match the training metadata.
4. Fit nothing.
5. Run inference in evaluation mode and disable gradients where appropriate.
6. Save per-entity predictions before calculating summaries.
7. Report the primary metric, secondary metrics, uncertainty, confusion matrix
   or residual analysis, and representative errors.

For small test sets, accuracy is quantized and confidence intervals are wide.
Report ROC-AUC or another appropriate ranking metric when justified, but never
use a metric simply because it gives a better-looking number.

## 11. Make every result reproducible

Save:

- code version or commit identifier;
- environment and dependency versions;
- random seed and deterministic settings;
- split manifest;
- preprocessing object and feature order;
- model architecture and class order;
- hyperparameters and selection rule;
- raw loss histories;
- predictions and metrics;
- data and exclusion audit;
- command used to reproduce the run.

Use separate directories for raw data, derived data, checkpoints, predictions,
plots, metrics, and reports. Make evaluation a separate process that consumes
exports rather than silently refitting them.

## 12. A reusable debugging loop

When a pipeline fails or behaves suspiciously:

1. Reproduce the failure with the smallest command or fixture.
2. Identify the exact stage: data, split, preprocessing, model, optimization,
   evaluation, or reporting.
3. Form one falsifiable hypothesis.
4. Run the cheapest check that can disprove it.
5. Make one focused change.
6. Rerun the same check.
7. Compare against the baseline and preserve the evidence.
8. Only then widen the experiment.

Do not respond to every bad metric by changing the model. First verify the
target, unit of analysis, split, feature meaning, data ranges, and metric code.

## 13. Final sign-off checklist

Before declaring success, confirm:

- [ ] Problem specification is explicit.
- [ ] Data inventory and modality check are complete.
- [ ] Unit of analysis is correct.
- [ ] Split is deterministic and entity-disjoint.
- [ ] Raw data remain unchanged.
- [ ] Learned preprocessing and feature selection use training data only.
- [ ] A simple baseline is reported.
- [ ] Model, output, loss, labels, and metrics agree.
- [ ] Test data stayed sealed until the final evaluation.
- [ ] Curves and metrics come from real runs.
- [ ] Overfitting and optimization symptoms were investigated from raw traces.
- [ ] Confounds, leakage risks, ethics, and limitations are documented.
- [ ] Predictions, uncertainty, artifacts, environment, and commands are saved.
- [ ] Results are reported at the correct unit of analysis.

If any item is incomplete, report the result as incomplete rather than claiming
the pipeline is finished.