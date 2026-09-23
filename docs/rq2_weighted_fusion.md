# Proposal-specified RQ2 weighted fusion

Development-only selection of the weighted composite specified by the submitted
P1 proposal. Existing equal-weight results and all individual/partial ablations
remain unchanged. This module never opens final-test artifacts, invokes detector
inference, refits calibration, reruns transformations or recomputes matching.

## Frozen inputs

Reuse `rq2_risk_evaluation.load_inputs` to validate the stored development stability
table, its upstream provenance/hash chain, 821-image manifest, 7,131 unique ordered
predictions, binary IoU50 labels and finite signals in [0,1]. The reference floor
remains 0.01. No alternative input-file or population option is exposed.

Reuse `rq2_calibrator_comparison.load_inputs` for confirmed development similarity
links; require identical manifest inventory and matching dependency hashes.
Only existing development text artifacts and similarity metadata are read, not
dataset images, annotations, model files or final-test results.

## Grid and score

For integer percentages `(a,b,c)`, enumerate a from 0 through 100, then b from 0
through `100-a`, with `c=100-a-b`. There are exactly **5,151** distinct triples,
including all simplex vertices and zero-weight candidates. Validate nonnegative
integers summing exactly to 100; floating-point enumeration is never used.

```text
weighted_risk = (a*(1-calibrated_confidence)
               + b*(1-class_consistency)
               + c*(1-localisation_stability)) / 100
```

The integer percentages are authoritative; JSON also provides weights divided by
100. No constraint forces every component to contribute. Risk calculation uses a
fixed arithmetic term order. Equal-weight reporting calls the existing risk
implementation directly, rather than approximating thirds with a grid point.

## Five-fold evaluation

Seed **24209199**. Reuse the established image-count-based fold allocator:
connected similarity components are indivisible, transitive links stay together,
singletons are allowed, and seeded tie-breaking is independent of labels and
performance. Persist all image assignments in source-manifest order before scoring.

For every candidate calculate scores, then AUROC and AUPRC separately within each
of five validation folds. AUPRC is the existing non-interpolated average precision
for **incorrect IoU50 predictions**, `1-correct_iou50`. Means give the five folds
equal weight, rather than pooling predictions or weighting folds by size.

Every fold must contain both target classes. An empty/single-class fold stops the
workflow; it is not silently removed, reassigned or selected using performance.
No component fitting takes place in these folds: all component signals are frozen.
This is development selection, not nested cross-validation or independent evidence
of generalisation. Full-development reporting after selection is descriptive.

## Predefined selection hierarchy

1. Find maximum mean fold AUPRC. Retain candidates within **absolute 1e-12** of
   that numerical maximum (inclusive; relative tolerance zero).
2. Within that set, find maximum mean fold AUROC and retain candidates within the
   same absolute tolerance of its maximum. Ties are anchored to the maximum, not
   chained through pairwise near-equality.
3. Minimise Euclidean distance to `(1/3,1/3,1/3)`. Compare the exact integer
   numerator `sum((3*percent-100)^2)` of squared distance; geometric ties require
   exact equality. Report distance as `sqrt(numerator)/300`.
4. Resolve remaining ties by larger confidence weight, then larger localisation
   weight, then larger class weight.

The candidate evaluator receives a primary-only projection containing image,
IoU50 label and three signals. It cannot access IoU75. Secondary labels are parsed
only after weights have been selected; the same selected weights and scores are
then used for secondary AUROC/AP. No secondary result can trigger reselection.

## Outputs and DICC command

After review and committing execution inputs, use the project environment with
NumPy, SciPy and PyYAML, from the repository root:

```bash
PYTHONPATH=src python -B -m trustpcb.rq2_weighted_fusion --dicc
```

This is CPU work. Do not run the real development grid locally; Windows and missing
`--dicc` are rejected. No GPU or detector-library import is needed.

Output: `runs/rq2/weighted_fusion/`.

- `candidate_weights.csv`: all integer triples, five fold AP/AUROC values,
  mean fold metrics and equal-weight distance.
- `fold_assignments.csv`: image, fold and connected-component ID.
- `selected_weights.json`: integer/fractional weights, selected CV metrics,
  hierarchy/tolerances and explicit IoU75 exclusion.
- `development_metrics.csv`: full-population calibrated-only, original equal-weight
  TrustPCB and selected weighted TrustPCB AUROC/AP.
- `sensitivity_iou75_metrics.csv`: the same three scores against secondary labels.
- `summary.json`: population, fold counts, candidate count and interpretation scope.
- `provenance.json`: proposal-specific purpose, source hashes, population, target,
  seed, folds/group safeguard, grid, hierarchy, selected weights, Git, versions,
  timestamps and output hashes.

Any existing output directory blocks complete/partial overwrite. Failures retain
incomplete provenance; archive explicitly before an approved retry. Only this
new directory is written. Inputs and Git state are checked again at completion.
No existing equal-weight or ablation output is replaced, and no referral threshold
is selected. Do not use final-test results to choose or revise these weights.

## Local verification

Tests enumerate the integer grid without executing its real-data search. Tiny
synthetic CV fixtures verify all hierarchy levels, group folds, label isolation
and secondary score reuse. Output tests use explicitly synthetic candidate records,
temporary directories and forbidden-fit/inference mocks. No scientific selected
weights or real evaluation results are generated during preparation.
