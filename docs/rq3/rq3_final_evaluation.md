# Frozen RQ3 final evaluation

Apply the frozen development referral protocol to exactly 2,052 test crops and
17,840 existing retained predictions. Population description:
"held out from RQ2 training and development, previously used for RQ1 validation."
This is final evaluation, with no method or operating-point selection on test.

## Inputs and safeguards

Reuse `rq2_weighted_final_evaluation.load_inputs()` for text-only validation of:

- `configs/datasets/similarity_aware_val_images.txt`, its pinned identity and
  separation from the frozen detector-training/development manifests
- `runs/rq2/final_evaluation/prediction_stability_scores.csv`
- `runs/rq2/final_evaluation/labelled_predictions.csv`
- `runs/rq2/final_evaluation/primary_iou50/summary.json` and completed provenance
- `runs/rq2/weighted_fusion/selected_weights.json` and evidence commit
  `74510277056c72114dc421bd900ddb5dfbc91933`

Check source table hashes, identities, ordering, confidence inclusion >=0.01,
finite signals, primary correctness, checkpoint/Beta metadata and exact counts.
The inherited RQ2 validator also checks stored IoU75 labels for integrity, but RQ3
uses only the frozen primary IoU50 definition. No labels are recomputed.

Load exactly `runs/rq3/development_selective_review/development_thresholds.json`
and validate its SHA-256 against the completed sibling `provenance.json`. Require
the original 821-crop/7,131-prediction development metadata, the frozen weights,
all five methods and nominal 5/10/20 budgets, strict `>` semantics and disabled
test retuning. Validate development weight identity against the reused frozen
weight evidence. Record the threshold file hash, source provenance hash and exact
applied values. Missing inputs fail; nothing is regenerated.

All existing evidence remains read-only. No images, annotation files or checkpoint
bytes are accessed. No detector/transform inference, calibration fitting or
application, computational benchmark, weight selection or threshold derivation runs.
This workflow consumes stored calibrated confidences and stability signals.

## Frozen calculations

Reuse the development functions for five higher-is-riskier scores: 1-raw confidence,
1-calibrated confidence, 1-class consistency, 1-localisation stability and the frozen
weighted TrustPCB score. Reconstruct the latter using the existing integer-weight
function with `(89,1,10)`. Ignore the source `trustpcb_risk` column: it is the legacy
equal-weight score, never the final weighted score.

Reuse complete ascending-score, whole-tie prediction risk–coverage curves and the
right-endpoint step AURC. Selective error means existing primary incorrect predictions
among retained predictions. It does not measure recall of missed objects.

Crop risk is maximum prediction risk. Empty crops have blank/null exported risk,
are never referred and remain in the 2,052-crop workload denominator. Apply only
`crop_risk > frozen_threshold`. No interpolation, tie-breaking, quantile derivation
or workload forcing is permitted. Test workload can differ from nominal development
budgets. Report `development_nominal_crops`, `development_achieved_crops` and
`development_achieved_workload` separately from actual test `referred_crops` and
`workload`; the development counts are descriptive metadata, not test targets.

For each method/operating point report referred predictions, captured errors and
error-capture fraction; automatic predictions/errors; remaining automatic error
rate; no-referral error rate; and absolute error reduction (reference minus automatic
rate, in probability points). Undefined rates remain blank/null. Reuse the exact
random sanity baseline: uniform referral among nonempty crops at achieved workload,
with analytic expected capture equal to referred count / nonempty count.

Always run 10,000 paired whole-crop bootstrap replicates with seed 24209199 and
PCG64. Include empty crops in draws and move all predictions of each sampled crop
together. All methods use the same draws; saved thresholds stay fixed throughout.
Reuse development percentile intervals (linear 2.5/97.5 percentiles), validity
counts and paired differences. Invalid replicates are omitted without redrawing.
The no-referral rate uses each replicate's full prediction population.

Paired differences are weighted TrustPCB minus raw/calibrated reference, for AURC
and, at all three operating points, error capture and remaining automatic error.
Negative favours weighted for AURC/remaining error; positive favours it for capture.

## DICC execution and outputs

After reviewing and committing the execution inputs, from the repository root:

```bash
PYTHONPATH=src python -B -m trustpcb.rq3_final_evaluation --dicc
```

CPU-only DICC analysis of stored artifacts; GPU is unnecessary. Do not run the
real evaluation locally. The CLI rejects Windows and requires `--dicc`.
The development threshold file/provenance and full RQ2 prediction tables must exist
on DICC. This preparation does not establish their actual hashes locally or reveal
any final RQ3 result. Execution validates them before analysis.

Write only `runs/rq3/final_evaluation/`:

- `prediction_risk_coverage.csv`: all achievable retained-score boundaries, per method
- `aurc.csv`: secondary AURC summary, five methods
- `crop_risks.csv`: all test crops per method, including empty crops
- `crop_referral_metrics.csv`: 15 method/development-operating-point rows
- `bootstrap_intervals.csv`: per-method metric intervals and valid/invalid counts
- `paired_bootstrap_differences.csv`: 14 paired comparison rows
- `bootstrap_summary.json`: seed, replicates, shared-draw hash and paired summaries
- `summary.json`: population, definitions and execution flags
- `provenance.json`: current Git/cleanliness, all input hashes, exact threshold
  identity/values, weight identity, counts, correctness definition, versions,
  timestamps, execution flags and output hashes

Existing complete or partial output directories block execution. Failures leave
incomplete provenance; archive explicitly before a reviewed retry. Recheck frozen
inputs and source state at completion. Tests use synthetic data only, including
text-only population-count fixtures; no real final-test analysis runs locally.
