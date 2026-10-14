# RQ3 research reporting

RQ3 investigates selective crop review, prediction risk–coverage and the
computational cost of the frozen TrustPCB pipeline.

| Experiment | Reporting materials | Original scientific identity |
|---|---|---|
| [Original development selective review](development_selective_review/summaries/README.md) | [tables](development_selective_review/tables/) · [figures](development_selective_review/figures/) · [summaries](development_selective_review/summaries/) | `rq3_development_selective_review` |
| [Development reporting extension](development_selective_review/reporting_extension/summaries/README.md) | [tables](development_selective_review/reporting_extension/tables/) · [figures](development_selective_review/reporting_extension/figures/) · [summaries](development_selective_review/reporting_extension/summaries/) | `rq3_development_reporting_extension` |
| [DICC computational efficiency](computational_efficiency/summaries/README.md) | [tables](computational_efficiency/tables/) · [figures](computational_efficiency/figures/) · [summaries](computational_efficiency/summaries/) | `rq3_computational_efficiency` |
| [Frozen-threshold final evaluation](final_evaluation/summaries/README.md) | [tables](final_evaluation/tables/) · [figures](final_evaluation/figures/) · [summaries](final_evaluation/summaries/) | `rq3_final_evaluation` |

Each experiment summary links the canonical Python workflow, relocated tests,
protocol and original evidence. The three scientific modules and compatibility
shims remain unchanged. Original development and supplementary reporting-extension
schemas stay separate; final evaluation retains the frozen development thresholds.

## How to read these artifacts

- **Original evidence:** `runs/rq3/`, unchanged. Large curve/crop-risk tables stay
  there and are referenced with hashes instead of duplicated.
- **Reporting copies:** small CSV/JSON files here are byte-identical; original
  provenance, thresholds, scientific identities and execution metadata remain intact.
- **Newly rendered visualizations:** figures are new presentations of saved data,
  not original DICC figures. No inference, calibration, ranking, metric calculation,
  threshold selection, bootstrap or benchmark was performed to create them.
- **Development versus test:** final-test crops were held out from RQ2 training
  and development, but previously used for RQ1 validation. Development plots are
  conditional development diagnostics, not independent generalisation evidence.
- Referral plots use **achieved workloads** on their horizontal axes. Nominal
  development budgets identify frozen operating points; they are not enforced
  test workloads. The automatic-error figure uses nominal budget categories only.
- Whole-score-group risk–coverage boundaries are rendered as right-endpoint steps.
  Undefined zero-coverage risk is not replaced with zero. Saved AURC is not recalculated.
- Paired differences retain weighted-method-minus-reference direction. Negative
  differences favour weighted for AURC/remaining automatic error; positive favour
  weighted for capture. Prediction-level results do not imply crop-level gains.
- Runtime figures show the saved DICC protocol, including its exclusions. They
  do not establish real-time performance or factory readiness. Saved sample SD
  across repetitions is distinguished from bootstrap confidence intervals.

## Missing/external inputs

The local RQ3 reporting inputs listed in the manifest are present. Full upstream
RQ2 detector/checkpoints, raw images/labels, development confidence/calibration/
stability artifacts and detailed final prediction/transform-match files remain
absent locally. Some downstream RQ2 summaries/weights exist; they do not replace
that missing chain. External/DICC locations must be confirmed and identities
verified when retrieving original artifacts. Absence locally does not mean they
were never generated. See [artifact inventory](../../docs/artifact_inventory.md)
and [RQ2 availability](../rq2/README.md). No missing evidence was regenerated.

## Reproduction and traceability

[reporting.py](../../src/trustpcb/rq3/reporting.py) is new presentation code. It
reuses only pure [RQ2 reporting helpers](../../src/trustpcb/rq2/reporting.py),
whose hash is recorded alongside its own. It does not import scientific workflows.
With `PYTHONPATH=src` and the existing Matplotlib environment:

```text
python -B -m trustpcb.rq3.reporting
python -B -m trustpcb.rq3.reporting --verify-only
```

The [manifest](manifest.json) records original source paths/SHA-256, reporting
paths/SHA-256, experiment identities, availability, derivation scripts/recipes and
rendering versions. Copies, visualizations and authored navigation are classified
separately. Its base commit is the pre-batch reference, not an experiment execution
commit. Hashes cover exact local bytes/line endings and do not replace historical
scientific LF-canonical hashes. The manifest is the inventory and does not hash itself.
Hash verification certifies these reporting inputs/outputs, not unavailable upstream
evidence. Existing edited/unlisted reports or changed inputs block rebuilding.

[Reporting protocol and CPU test commands](../../docs/rq3/reporting.md) ·
[RQ1 reporting](../rq1/README.md) · [RQ2 reporting](../rq2/README.md)
