# RQ2 reporting and test organization

Browse [RQ2 tables, figures and summaries](../../outputs/rq2/README.md). This is
an additive presentation layer over completed evidence, not a research rerun.
The 13 existing canonical RQ2 modules, compatibility shims, writer paths and
scientific protocols are unchanged. No RQ1 or RQ3 files are changed.

## Local material and gaps

Five completed experiment directories under `runs/rq2/` supply saved evidence:
`risk_evaluation`, `risk_sensitivity_iou75`, `weighted_fusion`, `final_evaluation`
and `weighted_final_evaluation`. Small CSV/JSON files are copied byte-for-byte
into experiment-specific `tables/` and `summaries/`. Primary and secondary final
results keep distinct paths. The candidate-weight grid stays at its original
path and is referenced with its hash; fold assignments are a small reporting copy.
The frozen partition verification report is copied under
`outputs/rq2/data_partition/summaries/`, with links to the original membership files.

The original RQ2 figure files are not present locally. New plots directly display:

- Saved AUROC/AUPRC point estimates and available 95% bootstrap intervals.
- Saved paired differences, preserving method-A-minus-method-B signs and endpoints.
- Saved integer-percentage selected weights, without reselection or normalization.
- Saved final-test calibration bins and Brier/NLL/ECE values, without rebinning or fitting.

These displays do not recompute metrics, statistical intervals, calibration, risk
scores, ranking, predictions, folds or weights. Missing values stay undefined.
An aggregate AUROC/AUPRC value cannot reconstruct a ROC/precision–recall curve;
none is invented. Display labels wrap original method identifiers for readability.
Original `trustpcb_risk` is the equal-weight composite, distinct from the weighted
method. No statistical-significance conclusions are added by the reporting code.

Development selection is not independent generalisation evidence. The final-test
population was held out from RQ2 training/development but previously used for RQ1
validation. Existing original and weighted final evaluations remain separate.

Upstream detector-training, development confidence/calibration and transformation
stability outputs/checkpoints remain absent locally. Frozen partition manifests and
their verification report are local. Detailed final predictions, transform matches,
inference audit and some bootstrap tables are also absent; see the
[artifact inventory](../artifact_inventory.md) for exact names. Their external/DICC
locations need verification. Reporting copies do not certify the unavailable chain.
Retrieve and verify existing artifacts; do not regenerate them for organization.

## Reproduction

The new [reporting module](../../src/trustpcb/rq2/reporting.py) imports no scientific
workflow and loads Matplotlib only when rendering. From the repository root, set
`PYTHONPATH=src` and use the existing environment with Matplotlib:

```text
python -B -m trustpcb.rq2.reporting
python -B -m trustpcb.rq2.reporting --verify-only
```

Only `outputs/rq2/` is written. Existing edits/unlisted files, changed sources or
newly available upstream artifacts require explicit inventory review before a
rebuild. The manifest records exact local source/output hashes, references,
classification, derivation recipes and rendering versions. These byte hashes do
not replace scientific LF-canonical identities. Original provenance and its
embedded paths remain unchanged; new visualization metadata is separate.

The root README is deliberately unchanged: committed RQ1 reporting records its
exact hash. Existing RQ2 protocol pages link here for navigation instead.

## CPU-safe tests

All 13 existing RQ2 test files now live in `tests/rq2/`. Repository-root calculations
account for the extra directory; three cross-test fixture imports are package
relative. Scientific assertions, patch targets and canonical/legacy checks remain
unchanged. The package marker supports the existing full discovery command.

```text
python -B -m unittest discover -s tests/rq2 -t tests -v
python -B -m unittest discover -s tests -v
```

Use `PYTHONPATH=src` and existing CPU test dependencies. The explicit top-level
`-t tests` makes relative fixture imports work for targeted discovery. If pytest is
already available, `python -B -m pytest --collect-only -p no:cacheprovider tests`
checks collection without running experiments or writing a pytest cache.
