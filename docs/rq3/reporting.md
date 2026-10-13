# RQ3 reporting organization

Browse [RQ3 tables, figures and summaries](../../outputs/rq3/README.md). This is
an additive presentation layer over saved `runs/rq3/` evidence, not scientific
execution. The three existing RQ3 modules, compatibility shims, thresholds,
weights, identities, schemas and overwrite guards remain unchanged.

## Experiments and evidence boundaries

- `development_selective_review`: original development threshold/referral,
  risk–coverage, AURC and uncertainty evidence.
- `development_selective_review/reporting_extension`: historical supplementary
  automatic-error and paired-difference evidence. It remains separate and does
  not replace the original development schema or thresholds.
- `computational_efficiency`: saved five-repetition DICC timing evidence for the
  two pipelines on the fixed 821-crop development population.
- `final_evaluation`: fixed development thresholds applied to 2,052 test crops;
  achieved workload is retained, never forced to the nominal development budget.

Small CSV/JSON files and the development DICC-run note are byte-identical reporting
copies. Full `prediction_risk_coverage.csv` and `crop_risks.csv` files remain at their
original locations, with SHA-256 references in the reporting manifest. Figures read
the saved curve coordinates directly; no large prediction or crop tables are copied.

## Figure interpretation

The new reporting module performs presentation only. It does not call scientific
workflows, rank predictions, select thresholds, fit calibration, integrate AURC,
resample bootstrap draws, evaluate models or benchmark pipelines.

- Risk–coverage plots retain all saved boundaries and whole-tie behavior. Their
  right-endpoint steps use Matplotlib `where="pre"`; undefined risk at zero coverage
  remains undefined rather than being assigned a fabricated zero.
- AURC points and confidence endpoints come from saved CSVs. Interval plots use
  actual lower/upper values, not symmetric error approximations around a mean.
- Referral figures plot saved `workload` against saved `error_capture`. Separate
  panels identify nominal development budgets; they do not rescale test workloads.
  Horizontal and vertical intervals are marginal saved intervals, not joint regions.
- Automatic-error figures use categorical development-budget labels and small
  display offsets; they do not represent those categories as achieved test workload.
- Paired differences remain weighted-method-minus-reference. Negative differences
  favour weighted for AURC/remaining automatic error; positive differences favour
  weighted for capture. No new significance conclusions are calculated.
- Prediction-level AURC and crop-level referral are different outcomes. The displays
  must not be read as a general claim that weighted TrustPCB improves crop referral.
- Runtime plots use stored means, sample SD and raw repetition timings. One timing
  panel has a labelled logarithmic scale. Sample SD is not a confidence interval.
  The protocol excludes startup/model loading, provenance checks, hashing and output
  writes. These DICC measurements do not establish real-time or factory-ready use.

Final-test images were held out from RQ2 training/development but previously used
for RQ1 validation. Development results remain conditional development evidence.
The reporting extension and held-out evaluation are never pooled.

## Availability and provenance

The local RQ3 reporting sources are present. Full upstream RQ2 detector/checkpoints,
raw dataset, development predictions/calibration/stability, and detailed final
prediction/transform-match artifacts are unavailable locally. External locations
must be confirmed and original identities verified on retrieval; missing local
artifacts must not be regenerated for housekeeping. See
[the artifact inventory](../artifact_inventory.md) and
[RQ2 reporting availability](../../outputs/rq2/README.md).

Each reporting artifact records classification, original source paths/hashes,
reporting path/hash, original experiment identity, local availability, derivation
scripts and recipe. New figures do not inherit the identity of original DICC plots.
Copied provenance retains its original meaning. Hashes cover exact local bytes,
including line endings; they do not replace scientific LF-canonical identities.
Manifest verification does not certify the unavailable upstream scientific chain.

## Reporting and CPU-safe tests

With `PYTHONPATH=src` and existing CPU dependencies:

```text
python -B -m trustpcb.rq3.reporting
python -B -m trustpcb.rq3.reporting --verify-only
python -B -m unittest discover -s tests/rq3 -t tests -v
python -B -m unittest discover -s tests -v
```

Only `outputs/rq3/` is written by the reporting builder. Existing edited/unlisted
reports or changed source hashes block rebuilding. The new module reuses only
pure helpers in `trustpcb.rq2.reporting`; the helper source hash is recorded.
Neither reporting import triggers Matplotlib or scientific/ML workloads.

The three original test files now live in `tests/rq3/` with only repository-root
depth repairs. Assertions, mock targets and compatibility checks are preserved.
New reporting tests use tiny synthetic values and temporary files. Check pytest
collection only when pytest already exists in the environment.

RQ1/RQ2 reporting and the root README remain unchanged; the RQ1 manifest binds
the README's hash. RQ3 protocol pages link here for navigation instead.
