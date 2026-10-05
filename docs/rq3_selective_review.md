# RQ3 development selective review

This workflow evaluates the frozen 821-image development/calibration population
using existing RQ2 text artifacts only. Each manifest image is one reviewable crop.
It does not load the test manifest, test predictions, dataset images, annotations
or checkpoint bytes. It does not fit a calibrator, rerun inference or transformations.
Missing inputs cause failure; they are never regenerated.

## Frozen inputs and methods

Reuse `rq2_risk_evaluation.load_inputs()` to validate the development manifest,
7,131 retained predictions (raw confidence >= 0.01), original prediction order,
Beta-calibrated confidence, transformation stability and their provenance chain.
The upstream loader binds the epoch-72 detector hash through completed extraction
metadata. Correctness remains same-class one-to-one IoU >= 0.50. No matching is rerun.
The existing frozen-weight evidence validator compares `selected_weights.json`
against Git commit `74510277056c72114dc421bd900ddb5dfbc91933` using `git show`
with a repository working directory, compatible with DICC Git 1.8.3.1.
Only that evidence validator is called; no RQ2 final-test loader is called.

Five methods use higher = higher risk:

- `raw_confidence_risk = 1 - raw_confidence`
- `calibrated_confidence_risk = 1 - calibrated_confidence`
- `class_consistency_risk = 1 - class_consistency`
- `localisation_stability_risk = 1 - localisation_stability`
- `weighted_trustpcb_risk = 0.89*(1-C) + 0.01*(1-Sc) + 0.10*(1-Sl)`

The weighted arithmetic reuses the frozen integer-percentage implementation.
No weights or calibrators are selected or fitted.

## Prediction risk–coverage curve and AURC

Retain predictions from lowest to highest score, adding entire equal-score groups.
Export every achievable scalar-score boundary; do not split ties by correctness
or prediction identity. Coverage is retained predictions / all retained-source
predictions. Empirical selective risk is incorrect retained / retained. At zero
coverage risk is undefined, rather than assigned zero.

AURC is the right-endpoint step integral: sum of each increase in coverage times
the corresponding retained error rate. Lower is better. This convention preserves
whole ties and is recorded in provenance. Empty bootstrap populations have undefined
AURC. This curve concerns existing predictions, not missed ground-truth objects.

## Crop referral and development thresholds

Crop risk is the maximum risk among its retained predictions, separately per method.
An empty crop has internal risk `-inf`, exported as a blank/null and explicitly
marked ineligible. All 821 crops remain in the workload denominator. Empty crops
are never referred, including when a threshold is below every nonempty score.

The predeclared budgets 5%, 10%, 20% have floor counts 41, 82, 164. Referral is
strictly `crop_risk > threshold` for a nonempty crop. Sort nonempty scores descending
and use the first excluded score as the boundary. All equal boundary scores remain
non-referred. This attains the largest achievable scalar-threshold count that does
not exceed the nominal budget; there is no crop-ID tie-break or label-based choice.
If all nonempty crops fit within the budget, use -1 (all scores lie in [0,1]).
Record nominal counts, actual counts, workload and the exact threshold for each
method/budget. Empty populations still produce zero referrals.

The derived thresholds are written to `development_thresholds.json`. They are
development-derived protocol candidates for review and subsequent freezing. Future
test application must use the frozen scalar values unchanged: no workload forcing,
test quantiles or retuning. No test application is implemented here.

Error capture is the fraction of all incorrect predictions lying in referred crops.
Also report referred prediction count, captured incorrect count and error rate
among referred predictions. Error capture is undefined when there are no incorrect
predictions. These are not estimates of human correction effectiveness or recall
of undetected objects.

The random sanity baseline selects uniformly without replacement among nonempty
crops at the same achieved crop count. It obeys empty-crop ineligibility and retains
all crops in the workload denominator. Its exact expected error capture is
referred count / nonempty count, with expected captured errors and referred
predictions also reported. This analytic expectation avoids random Monte Carlo
noise; no separate random-referral experiment is run.

## Optional paired whole-image uncertainty

`--bootstrap` enables 10,000 whole-image replicates with seed 24209199 and NumPy
PCG64. Sample 821 crops with replacement; every prediction in a sampled crop carries
that crop's multiplicity. Include empty crops in the draws. All methods/budgets
share the same image draw sequence. Record its SHA-256 for reproduction.

Keep original development thresholds fixed in every replicate. Report percentile
2.5/97.5 intervals, means and valid/invalid counts for AURC, achieved workload,
error capture and random expected capture. Undefined replicate metrics are excluded
without redrawing. This is conditional development uncertainty, not independent
test evidence; weights and Beta calibration were developed on this population.
The bootstrap module is separate from point calculations. Full bootstrap curves
are not persisted; full observed curves are exported.

## DICC execution

After review and committing execution inputs, from the repository root:

```bash
PYTHONPATH=src python -B -m trustpcb.rq3_selective_review --dicc
```

To include the optional 10,000-replicate uncertainty calculation in that run:

```bash
PYTHONPATH=src python -B -m trustpcb.rq3_selective_review --dicc --bootstrap
```

Choose one command before execution. CPU only; no GPU required. The CLI rejects
Windows. Either command blocks an existing partial/completed output directory;
archive explicitly before a reviewed retry. It checks scientific source cleanliness,
input hashes before/after, package versions and output hashes. RQ1/RQ2 evidence is
read-only. Historical artifact aliases are handled by existing input resolvers.

Output directory: `runs/rq3/development_selective_review/`:

- `prediction_risk_coverage.csv`: every achievable prediction score boundary
- `aurc.csv`: secondary area summary for each method
- `crop_risks.csv`: all 821 crops per method, including empty crops
- `crop_referral_metrics.csv`: 15 method/budget rows, achieved workloads and capture
- `development_thresholds.json`: scalar boundaries, budgets and no-test-retuning rule
- `summary.json` and `provenance.json`
- optional `bootstrap_intervals.csv` and `bootstrap_summary.json`

Local verification uses tiny synthetic fixtures only. Real development evaluation
and bootstrap execution belong on DICC. The missing local RQ2 artifacts must be
provided by the existing completed DICC runs, never regenerated for this workflow.
