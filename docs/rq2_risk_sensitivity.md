# Secondary IoU>=0.75 sensitivity analysis

This development-only analysis checks robustness to the existing stricter
correctness criterion. **IoU >= 0.50 remains the primary research target.** Do not
replace its results, choose whichever IoU threshold looks better, or use this
analysis to change the detector, calibrator, transformations, signals, risk
formula, weights or referral thresholds.

## Frozen inputs and target

Input: `runs/rq2/transformation_stability/prediction_stability_scores.csv`.
Require the same 821-image development manifest and 7,131 unique reference rows
as the primary evaluation. Reuse its complete provenance/hash-chain, population,
identity/order, range, missing-value and development-membership checks. No images,
annotations, checkpoints or final-test manifest are opened.

Read the stored binary `correct_at_iou75` field (or `correct_iou75` alias). Reject
missing, nonbinary or conflicting alias values. Do not recompute ground-truth
matching. Define:

```text
incorrect_iou75 = 1 - frozen IoU75 correctness
```

Rows are copied in memory. The shared evaluator's `incorrect_prediction` argument
is set to this secondary target; stored IoU50 fields and input files remain
unchanged. IoU75 labels are never supplied to any fitting or selection routine.

## Unchanged risks and metrics

Call `rq2_risk_evaluation.risk_scores` directly. Let `r=1-raw_confidence`,
`c=1-calibrated_confidence`, `k=1-class_consistency`, `l=1-localisation_stability`.
The same eight scores are `r`, `c`, `k`, `l`, `(c+k)/2`, `(c+l)/2`, `(k+l)/2`,
and `(c+k+l)/3`, in the primary method order. No weights are tuned.

Reuse prediction-weighted AUROC with half credit for exact ties, and AUPRC defined
as non-interpolated average precision over distinct descending score groups.
Incorrect IoU75 predictions are positive. Report total, correct, incorrect and
incorrect prevalence; prevalence is the PR reference level.

## Identical paired bootstrap procedure

Call the primary bootstrap directly: **10,000** image-cluster replicates, seed
**24209199**, NumPy PCG64. Each draw samples 821 images with replacement, retaining
all predictions and repeated-image multiplicity, including zero-prediction images
in the sampling population. All methods share each draw. With the same manifest
order, seed and RNG implementation, draw indices match the primary procedure;
the target does not influence sampling. Record the draw-index SHA-256.

Reuse the primary undefined-metric policy without redrawing: AUROC is undefined
without both classes; average precision is undefined without positives, and is
1 for an all-positive nonempty sample. Both are undefined for an empty sample.
Report valid/invalid counts per method and metric and retain every replicate's
index, values and validity flags.

Repeat the same five AUROC comparisons (A minus B): full TrustPCB versus raw;
full versus calibrated; calibrated-plus-class versus calibrated;
calibrated-plus-localisation versus calibrated; class-plus-localisation versus
calibrated. Report observed delta, bootstrap mean, and 2.5th/97.5th percentiles
using linear interpolation over valid paired draws. No automatic interpretation
or model/threshold-selection rule is introduced.

## DICC command and outputs

After review and committing the implementation, from the repository root:

```bash
PYTHONPATH=src python -B -m trustpcb.rq2_risk_sensitivity --dicc
```

Dependencies are the existing NumPy, SciPy and PyYAML environment. This is CPU-only;
no GPU or detector library is needed. Real development processing must remain on
DICC. Windows execution is blocked and `--dicc` is required.

Output: `runs/rq2/risk_sensitivity_iou75/`.

- `method_metrics.csv`: eight secondary AUROC/AP results and bootstrap intervals.
- `pairwise_auroc_differences.csv`: the five fixed comparisons.
- `bootstrap_metrics.csv`: all replicate/method values and validity flags.
- `bootstrap_summary.json`: seed/RNG/draw hash and valid/invalid counts.
- `summary.json`: secondary counts/prevalence, metric definitions and scope.
- `provenance.json`: explicit secondary label, target, primary/secondary IoUs,
  unchanged formulas, Git, input/output hashes, versions and timestamps.

Every table and JSON output is labelled **Secondary IoU>=0.75 sensitivity analysis**.
The primary `runs/rq2/risk_evaluation/` directory is neither read nor written.
Existing complete or incomplete sensitivity output blocks execution. Exceptions
leave incomplete provenance; archive explicitly before an approved retry. Inputs
and source Git state are rechecked before marking completion.

Local tests use synthetic populations and reduced bootstrap counts only. They
cover target inversion, unchanged scores, paired image multiplicities and draw
identity, undefined metrics, difference signs, guards, provenance and primary
output immutability. No real sensitivity results are generated during preparation.
