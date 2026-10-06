# RQ2 final Beta calibrator

Fit one frozen Beta mapping after review of the completed five-fold calibrator
comparison. This operation does not repeat comparison, select another method,
perform inference or access final-test data. Real execution belongs on DICC.

## Frozen population and implementation

Input: `runs/rq2/calibration_analysis/labelled_predictions.csv`.
Use all **7,131** retained predictions from the frozen **821-image** development
manifest, with raw confidence **>= 0.01** and primary same-class one-to-one
**IoU >= 0.50** correctness. No refiltering occurs. Development images without
retained predictions remain part of the manifest inventory; they contribute no
prediction rows. IoU75 is copied as an opaque source column, never used for fitting
or metrics. Original identifying columns and row order are preserved.

The implementation calls these existing comparison functions directly:

- `fit_calibrator("Beta", confidence, labels)` exactly once on the full population;
- `apply_calibrator(model, confidence)`;
- `metric_summary(probabilities, labels)` for raw and fitted probabilities.

The unchanged mapping is:

```text
p_cal = sigmoid(a * log(p) - b * log(1-p) + c)
a >= 0, b >= 0, c unrestricted
```

Logarithms clip their input to `[1e-15, 1-1e-15]`. The mapping is non-decreasing.
Unregularised mean binary NLL is optimised deterministically with L-BFGS-B,
initial parameters `[1,1,0]`, analytic gradients, `maxiter=2000`, `ftol=1e-12`,
`gtol=1e-8`, `maxls=50`. No random initialisation or hyperparameter search occurs.
Optimiser failure or a single-class fitting population stops execution.

## Validation and reviewed selection

Reuse the comparison input validator to verify the frozen development manifest,
checkpoint identity in source provenance, population size, confidence floor,
primary labels, row identities, table hash and confirmed-link boundaries. No
image, annotation, checkpoint or final-test manifest is read.

Require completed comparison provenance with the frozen protocol, matching input
hashes and `final_fit_performed == false`. Verify its selection-summary and
method-metrics hashes. The selection must name **Beta** and have the existing
`comparison_complete_pending_review` status. Do not rewrite that historical record.

The explicit `--approve-beta-selection` flag records acceptance of the completed
selection. Without it, even an otherwise valid pending-review result is rejected.
This confirms the user's review; it does not rerun or alter the selection rule.
An incomplete comparison, scientific-review-required result, non-Beta selection,
or prior final fit is rejected. The calibration-analysis summary hash is also
verified. Input hashes and execution Git state are checked again before completion.

Both complete and incomplete existing output directories block execution before
fitting. Archive a prior directory explicitly if a retry is approved; there is no
overwrite or implicit restart option. Treat artifacts as frozen only when their
provenance status is `complete` and recorded output hashes match.

## Future DICC command

After reviewing/committing the implementation, use the project's environment with
NumPy, SciPy and PyYAML installed. From the repository root:

```bash
PYTHONPATH=src python -B -m trustpcb.rq2_final_calibrator --dicc --approve-beta-selection
```

This is CPU-only. No GPU, PyTorch or Ultralytics is needed. The CLI rejects Windows
and requires `--dicc`; do not process real development predictions locally.

Output: `runs/rq2/final_calibrator/`.

| File | Contents |
|---|---|
| `final_calibrator.json` | Beta coefficients/form, epsilon, monotonicity constraints, frozen population/target, source/comparison hashes, Git commit, optimiser settings/convergence, versions and timestamp |
| `calibrated_development_predictions.csv` | Original columns/order plus prediction ID, raw/calibrated confidence and primary correctness label |
| `fit_summary.json` | Coefficients, convergence/iterations, raw and fitted NLL/Brier/ECE and bin details |
| `provenance.json` | Review confirmation, Git identity, versions, input/output hashes and running/complete/incomplete state |

**All fit-summary metrics are in-sample fit diagnostics, not independent evidence
of generalisation.** NLL uses epsilon `1e-15`; Brier uses unclipped probabilities.
ECE uses the established ten equal-width bins `[0,.1)`, ..., `[.9,1]` and
prediction-weighted absolute confidence/correctness gaps. Method-selection evidence
remains the existing five-fold out-of-fold comparison.

Local tests use only synthetic text fixtures and tiny synthetic fits. The
7,131-row fixture checks inventory/order only; it does not fit a real or full-sized
scientific population. No completed scientific output is changed by preparation.
