# RQ2 calibrator comparison

This workflow prepares a development-only, out-of-fold comparison. It does not
train a detector, access final-test data, or fit a final calibrator. Review the
comparison before approving any later final fitting.

## Frozen inputs and boundaries

- Input: `runs/rq2/calibration_analysis/labelled_predictions.csv`, verified against
  its completed provenance and table hash.
- Population: all 7,131 retained predictions from the frozen 821-image development
  manifest. Confidence inclusion remains `>= 0.01`. Invalid rows cause failure;
  the workflow does not filter the population again.
- Target: `correct_at_iou50`, exported internally as `correct_iou50`. The IoU 0.75
  field is never read for fitting, folds, metrics or selection.
- Detector identity: epoch 72 and the frozen checkpoint SHA-256 are verified in
  the input provenance. No checkpoint, image or annotation file is opened.
- Confirmed similarity links come from
  `outputs/tables/cross_split_high_confidence_near_duplicates.csv`. Only this link
  metadata is read, including intermediate nodes needed for transitive closure.
  A component crossing the frozen development boundary causes failure. The final
  test manifest and test data are never opened.

## Five image-level folds

Seed: **24209199**. Form connected components of confirmed links, including
singleton images. Canonically ordered components are shuffled with Python's
seeded `random.Random`, then stably sorted by decreasing image count. Allocate
each entire component to the currently smallest fold, resolving ties by a fixed
seeded permutation of fold IDs. This balances image counts as group sizes permit.
No correctness label, prediction count or performance measure enters allocation.

Save `fold_manifest.csv` in source manifest order before fitting. Every image,
including an image with no retained predictions, appears once. All four methods
use these same folds. Each fit uses only four folds and predicts the fifth.
The concatenated results contain one out-of-fold value per prediction per method.

## Mathematical mappings and fitting

Let `p` be raw confidence, `sigmoid(z) = 1/(1+exp(-z))`, and
`logit(p) = log(p)-log(1-p)`.

| Method | Mapping | Constraints |
|---|---|---|
| Temperature | `sigmoid(logit(p)/T)` | Fit inverse temperature `a=1/T >= 1e-8` |
| Platt | `sigmoid(a*logit(p)+c)` | `a >= 1e-8`; intercept `c` free |
| Beta | `sigmoid(a*log(p)-b*log(1-p)+c)` | `a,b >= 0`; intercept `c` free |
| Isotonic | Weighted pool-adjacent-violators fit | Non-decreasing fitted probabilities |

Temperature and Platt are increasing; Beta is non-decreasing under the stated
constraints. The small positive slope floor implements strict positivity
numerically (Temperature consequently has `T <= 1e8`). Parametric fits minimise
unregularised mean binary NLL with analytic gradients and bounded L-BFGS-B:
`maxiter=2000`, `ftol=1e-12`, `gtol=1e-8`, `maxls=50`. Initial values reproduce
the identity mapping. Failed optimisation or a single-class training fold stops
the workflow; there is no silent replacement model.

Isotonic fitting aggregates tied scores, using their prediction counts as weights
and correctness means as targets. PAVA merges decreasing adjacent blocks.
Held-out values use linear interpolation between fitted unique scores and constant
endpoint extension outside fitted support.

Only log/logit computations clip to `[1e-15, 1-1e-15]`. NLL uses the same epsilon
for logarithms; Brier, ECE, plots and stored probabilities remain unclipped.
See the [Beta calibration paper](https://proceedings.mlr.press/v54/kull17a.html)
and [SciPy optimiser documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).

## Metrics and paired uncertainty

Calculate global prediction-weighted NLL and Brier for Raw and the four methods
after joining held-out predictions. ECE and reliability use the existing ten
equal-width bins: `[0,.1)`, ..., `[.9,1]`. Internal boundaries belong to the bin
on their right; empty bins have null mean probability and correctness rate. ECE
is the prediction-count-weighted absolute gap. It never selects a method.

Use **10,000** paired image-cluster bootstrap draws with seed **24209199** and
NumPy `Generator(PCG64)`. Each draw samples 821 images with replacement. An image's
entire prediction group receives its sampled multiplicity; zero-prediction images
remain in the sampling population. Divide summed losses by the sampled prediction
count, not image count. All methods and metrics use exactly the same draws.
Per-image sufficient sums avoid materialising repeated prediction rows. Resampling
is batched in groups of 128; a hash of all sampled image indices is recorded.

Compute competitor-minus-reference differences and the 2.5th/97.5th percentiles
with linear interpolation. These are conditional on the fixed OOF predictions;
models are not refitted during bootstrap. A resample containing no predictions
causes failure rather than silently redrawing.

## Predefined selection rule

1. Find the numerical NLL leader among the four calibrators.
2. Its NLL-indistinguishable set contains that leader and each competitor whose
   paired delta-NLL 95% interval includes zero, endpoints inclusive.
3. Find the numerical Brier leader within that set. Retain it and competitors
   within that set whose paired delta-Brier interval against it includes zero.
4. Choose the first remaining method in the fixed simplicity order:
   **Temperature, Platt, Beta, Isotonic**. Exact numerical ties use that same order.

Raw is a benchmark, never a candidate. If no calibrator has strictly lower
point-estimate OOF NLL than Raw, record `scientific_review_required` and no selected
method. Otherwise report the selected method's paired NLL and Brier differences
from Raw. The uncertainty/simplicity rule can select a method other than the
numerical leader; no additional performance-based rule is introduced.

The selection record is for review. This module has no final-refit operation.
Saved parameters are exclusively the twenty fold-specific fits.

## DICC execution and outputs

Dependencies: Python, NumPy, SciPy, Matplotlib and the project's existing PyYAML.
Commit/review the execution inputs before running. From the repository root on
DICC, with the completed calibration-analysis inputs available:

```bash
PYTHONPATH=src python -B -m trustpcb.rq2_calibrator_comparison --dicc
```

This is CPU work; no GPU, PyTorch or Ultralytics is required. The CLI rejects
Windows and requires explicit `--dicc`. Do not run real comparison data locally.

Output directory: `runs/rq2/calibrator_comparison/`.

- `fold_manifest.csv`: 821 image/fold/component records.
- `oof_calibrated_predictions.csv`: 28,524 rows across four methods, preserving
  prediction identity, image, raw confidence, primary label and held-out fold.
- `method_metrics.csv`: Raw plus four OOF method summaries.
- `paired_comparisons.csv`: point differences and paired intervals for all
  ordered method pairs, NLL and Brier.
- `selection_summary.json`: decision, candidate sets and selected-versus-Raw CIs.
- `reliability_data.json`, `reliability_comparison.png`: descriptive diagnostics.
- `fold_fit_parameters.json`: fold fits only; no final fitted calibrator.
- `provenance.json`: Git identity, input/output hashes, versions, seeds, numerical
  settings, fold counts, bootstrap draw hash and completion state.

Inputs and Git state are rechecked after processing. Any existing output directory,
including an incomplete run, blocks execution. Explicitly archive it for review
before a retry; completed results are never silently overwritten.

Local tests use synthetic populations only, reduced bootstrap draws and temporary
output directories. They are not scientific comparison results.
