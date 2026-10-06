# RQ2 development risk-ranking evaluation

Evaluate the eight frozen baselines/ablations using existing development results.
This operation does not fit a detector/calibrator, generate transformations,
optimise risk weights, select a referral threshold or choose a composite formula.
It is development analysis, not held-out final-test generalisation evidence.

## Inputs and safeguards

Primary input:
`runs/rq2/transformation_stability/prediction_stability_scores.csv`.

Validate **7,131** unique reference predictions against the frozen **821-image**
development manifest and final calibrated prediction table. Preserve original
identity, source fields and row ordering. Required raw/calibrated confidence,
class consistency and localisation stability must be present, finite and in
`[0,1]`. Reference raw confidence must remain `>= 0.01`; labels must be binary.
No rows are filtered, dropped or imputed.

Reuse the existing final-calibrator input validation and semantic artifact resolver
to verify source/comparison/final-calibrator hashes and completed provenance. Check
that completed transformation provenance binds the same input hashes and that its
score-table hash matches. The final method remains Beta. Read text artifacts only:
no image, annotation, checkpoint or final-test manifest is opened. A physical
`images/val/` filename is permitted only if it belongs to the frozen development
manifest; physical folder names do not determine scientific partition membership.

## Frozen target and eight formulas

The positive class is an **incorrect prediction**:
`incorrect_prediction = 1 - correct_iou50`. The existing same-class, one-to-one
IoU >= 0.50 correctness labels are unchanged. IoU75 never enters evaluation.

Let `r = 1 - raw_confidence`, `c = 1 - calibrated_confidence`,
`k = 1 - class_consistency`, `l = 1 - localisation_stability`.

| Method | Risk |
|---|---|
| `raw_confidence_risk` | `r` |
| `calibrated_confidence_risk` | `c` |
| `class_consistency_risk` | `k` |
| `localisation_risk` | `l` |
| `calibrated_plus_class_risk` | `(c+k)/2` |
| `calibrated_plus_localisation_risk` | `(c+l)/2` |
| `class_plus_localisation_risk` | `(k+l)/2` |
| `trustpcb_risk` | `(c+k+l)/3` |

These are fixed equal-weight arithmetic means. No optimisation or method-selection
rule is implemented. Higher risk is intended to rank incorrect predictions first.

## Metric definitions

**Primary AUROC:** probability that a randomly chosen incorrect prediction has
higher risk than a randomly chosen correct prediction, with half credit for ties.
Each prediction has equal weight in the observed result.

**Secondary AUPRC:** non-interpolated **average precision**,
`sum((recall_j - recall_(j-1)) * precision_j)` at distinct descending risk values.
Entire equal-score groups enter together. This is the step-integral definition,
not trapezoidal interpolation of the PR curve. Incorrect prevalence is reported
as the PR reference level alongside total/correct/incorrect counts.

AUROC is undefined when either target class is absent. AUPRC is undefined when
there are no positives; if every sampled prediction is positive, AUPRC is 1.
Both metrics are undefined for an empty sampled prediction population. Undefined
values appear as null in JSON and blank in CSV, with explicit validity flags in
the replicate audit table.

The implementation sorts each method's risks once, groups exact ties, and sums
frequency-weighted positive/negative counts within those groups. This is equivalent
to expanding sampled predictions, without allocating repeated prediction rows.

## Paired image-cluster bootstrap

- **10,000 replicates**, seed **24209199**.
- NumPy `Generator(PCG64)`, sampling 821 development images with replacement.
- Include all predictions of every sampled image, preserving repeated-image
  multiplicity. Images with zero retained predictions remain sampling units.
- Use exactly the same draw for all eight methods and both metrics.
- Work in batches of 64; each prediction receives its image's sampled multiplicity.
- Never redraw an undefined replicate. Retain its index and report valid/invalid
  counts separately for AUROC and AUPRC for each method.
- Percentile 95% intervals use the 2.5th and 97.5th percentiles of valid values,
  with linear interpolation. If no values are valid, bounds are null.
- Record the SHA-256 of all sampled indices as little-endian int64 bytes, package
  versions and RNG identity for reproducibility.

Report these paired AUROC differences as **A minus B**:

1. Full TrustPCB minus raw-confidence risk.
2. Full TrustPCB minus calibrated-confidence risk.
3. Calibrated-plus-class minus calibrated-confidence risk.
4. Calibrated-plus-localisation minus calibrated-confidence risk.
5. Class-plus-localisation minus calibrated-confidence risk.

Each comparison includes observed delta, bootstrap mean delta, percentile bounds
and valid/invalid replicate counts. Only draws where both metrics are defined
contribute to a paired interval. Intervals never trigger automatic model changes.

## Descriptive diagnostics

For calibrated confidence, class consistency and localisation stability, report
correct/incorrect groups separately: count, mean, population standard deviation
(`ddof=0`), median, Q1 and Q3. Quantiles use linear interpolation. Empty groups have
count zero and null statistics.

Spearman correlations use average ranks for ties and Pearson correlation of those
ranks. Constant signals or fewer than two predictions yield null correlations with
an explicit status. These diagnostics do not change weights.

If predicted class IDs are present for every row, produce class-level counts and
AUROC for calibrated-confidence, class-consistency, localisation and full TrustPCB
risks. A class without both correctness outcomes has null AUROC and an explicit
single-class status. No per-class threshold or formula is selected.

## Future DICC execution

After reviewing/committing the implementation, from the repository root with
NumPy, SciPy and the existing PyYAML dependency available:

```bash
PYTHONPATH=src python -B -m trustpcb.rq2_risk_evaluation --dicc
```

This is **CPU-only**; no GPU, PyTorch or Ultralytics import is required. Do not
run real development results locally. The CLI rejects Windows and requires
`--dicc`. Existing complete or incomplete output directories block execution;
archive explicitly before a reviewed retry. Inputs/Git state are checked again
before marking completion. Failures leave incomplete provenance.

Output directory: `runs/rq2/risk_evaluation/`.

- `method_metrics.csv`: eight methods, observed AUROC/AUPRC, bootstrap means,
  intervals and valid/invalid counts.
- `pairwise_auroc_differences.csv`: five fixed paired comparisons.
- `signal_descriptives.csv`, `signal_correlations.csv`: descriptive diagnostics.
- `per_class_diagnostics.csv`: optional predicted-class diagnostics.
- `bootstrap_metrics.csv`: all 10,000 × eight method results, replicate IDs and
  separate metric-validity flags; no invalid draw is discarded from this audit.
- `bootstrap_summary.json`: RNG/seed, draw hash, replicate count and validity counts.
- `summary.json`: population counts/prevalence, metric definitions and scope.
- `provenance.json`: Git commit/dirty status, input path/hashes, population,
  correctness definition, bootstrap settings, exact formulas, metric definitions,
  package/Python versions, timestamps and output hashes.

## Local verification

Tests use synthetic arrays, tiny text-artifact fixtures and reduced bootstrap
replicates. They cover all formulas, metric direction/ties, image multiplicity,
shared draws, undefined replicates, intervals, paired signs, descriptives,
correlations, provenance and input/overwrite safeguards. They do not generate or
claim scientific results. Real DICC evaluation remains unexecuted during preparation.
