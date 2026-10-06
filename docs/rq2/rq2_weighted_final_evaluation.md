# Frozen weighted RQ2 final evaluation

Apply the permanently frozen development weights **0.89 confidence / 0.01 class /
0.10 localisation** to existing final-test predictions. This is not a search,
selection or optimisation workflow. Existing equal-weight and final outputs remain
unchanged. No RQ3 or referral threshold is computed.

## Evidence and population validation

Require `runs/rq2/weighted_fusion/selected_weights.json` to contain exactly those
fractional weights, integer percentages `(89,1,10)` and IoU75 exclusion from selection.
Compare the file with its Git blob at evidence commit
`74510277056c72114dc421bd900ddb5dfbc91933`. Only CRLF/LF differences are tolerated;
record both actual file and committed-blob SHA-256. Missing Git evidence is a hard
failure, not a reason to trust a replacement file. Ensure this commit is available
in the DICC checkout before running (a shallow checkout may lack it).

Validate the pinned final-test manifest and separation through the existing
text-only manifest validator. Require **2,052 images and 17,840 reference rows**,
unique identities, valid binary labels and finite signals in [0,1], with raw
confidence >=0.01. Read the following existing final-evaluation artifacts:

- `prediction_stability_scores.csv`;
- `labelled_predictions.csv` (verify source fields and exact row order);
- `primary_iou50/summary.json`;
- completed `provenance.json`, binding their hashes and frozen identities.

Record the role verbatim: **held out from RQ2 training and development, previously
used for RQ1 validation**. Do not describe the population as historically untouched.
No image, annotation or checkpoint bytes are opened by this workflow.

## Frozen calculations

Let `c=1-calibrated_confidence`, `k=1-class_consistency`, `l=1-localisation_stability`.
Report three methods: calibrated risk `c`, existing equal-weight risk `(c+k+l)/3`,
and weighted risk `0.89*c+0.01*k+0.10*l`. Reuse the development weighted-score
function with integer percentages `(89,1,10)` to preserve its arithmetic order;
reuse the existing equal-weight calculation. No selection function is called.

Primary positive target is `1-correct_iou50`. Secondary positive target is
`1-correct_iou75`, explicitly labelled **Secondary IoU>=0.75 sensitivity analysis**.
Scores and weights are identical for both targets.

Reuse AUROC with half credit for ties and AUPRC as non-interpolated average
precision. The existing bootstrap accepts an optional ordered method list; its
default eight-method behaviour and numerical procedure remain unchanged.

For each target, use **10,000 paired image-cluster replicates**, seed **24209199**,
NumPy PCG64. Sample 2,052 images with replacement and retain all predictions with
their image multiplicity. Zero-prediction images remain sampling units. All three
methods share each draw; primary/secondary draw hashes must match. No invalid
replicate is redrawn. AUROC needs both classes; AP needs positives and is 1 for a
nonempty all-positive sample. Undefined values are null/blank and validity counts
are reported.

Report weighted-minus-calibrated and weighted-minus-equal differences for both
AUROC and AUPRC. Each difference subtracts method B from method A within the
same already-generated bootstrap replicate; AUPRC never starts another bootstrap.
Use tidy pairwise CSVs with four rows per target (two comparisons times two metrics).
Columns are `analysis_label`, `method_A`, `method_B`, `metric` (`auroc` or `auprc`),
`observed_difference`, `bootstrap_mean`, `lower_95`, `upper_95`,
`valid_replicates`, and `invalid_replicates`. Method A is always weighted TrustPCB.
Intervals use linear-interpolated 2.5th/97.5th percentiles of paired differences.
A replicate is invalid for a difference if either method's metric is undefined;
exclude it without redrawing. If none are valid, mean/CI fields are null/blank.
IoU75 remains sensitivity only and cannot affect weights or selection.
No additional significance tests or
performance-driven model changes are introduced.

## DICC execution and outputs

After review and committing execution inputs, from the repository root:

```bash
PYTHONPATH=src python -B -m trustpcb.rq2_weighted_final_evaluation --dicc
```

CPU-only; existing NumPy, SciPy and PyYAML suffice. No GPU, detector inference,
calibrator fitting, transformation processing or real local analysis is needed.
The CLI rejects Windows and requires `--dicc`. Existing completed/partial output
directories block execution. Failures leave incomplete provenance; archive
explicitly before a reviewed retry. Recheck inputs and Git state at completion.

Output: `runs/rq2/weighted_final_evaluation/`.

- `primary_method_metrics.csv`
- `primary_pairwise_differences.csv`
- `sensitivity_iou75_method_metrics.csv`
- `sensitivity_iou75_pairwise_differences.csv`
- `summary.json`: population, weights, per-target bootstrap metadata/draw hashes,
  counts/prevalence and validity counts.
- `provenance.json`: source evidence commit/file hashes, final-artifact hashes,
  role, targets, weights/formula, seed/count, Git, package versions, timestamps and
  output hashes. Explicit false flags record that no selection, fitting, inference,
  transformation processing or referral selection occurred.

Bootstrap replicate arrays remain in memory; compact outputs plus fixed input
hashes, RNG/seed, versions and draw-index hashes support reproduction. Existing
scientific results are never overwritten. Synthetic tests use tiny populations
and reduced replicate counts; real final-test analysis remains unexecuted locally.
