# Research artifact naming

Name files, folders, Python modules and experiment/provenance identifiers by their
scientific or computational purpose, rather than conversational workflow numbering.

Use purpose-based names such as:

- `rq2_data_partition`
- `rq2_detector_training`
- `rq2_confidence_distribution`
- `rq2_prediction_matching`
- `rq2_calibration_analysis`
- `rq2_transformation_consistency`
- `rq2_localisation_stability`
- `rq2_risk_model`
- `rq2_selective_review`

Avoid workflow-number labels and vague freshness markers such as final, latest,
new or old. Repeated runs may use identifiers such as `seed_24209199`, `test_01`
or `experiment_001` inside a parent directory that names the experiment's purpose.

## Immutable records and renamed paths

Renaming a file does not change its scientific identity. Frozen manifests,
checkpoint bytes, result tables and completed provenance must not be rewritten
to make their embedded paths or experiment identifiers look newer. Preserve
their original hashes, seeds, measurements and Git commit references.

Future provenance uses `experiment` and purpose-based identifiers; detector
provenance uses `partition_commit`. These metadata-name changes do not alter
scientific parameters or historical records.

The explicit aliases in `src/trustpcb/rq2_artifact_paths.py` are the only supported
historical path translations. They are used when reading a frozen checkpoint or
prediction directory and when comparing path fields in historical provenance.
The provenance files themselves stay byte-for-byte unchanged. Hash and membership
checks remain mandatory. If both artifact locations exist, reading stops rather
than choosing one silently.

Writers use semantic paths exclusively. Existing historical output directories
or detector sidecars block a new run under the renamed destination. No experiment
should be rerun because of this refactor. A historical completed detector receipt
may still fail the training runner's strict current-plan/Git check; use its frozen
selected checkpoint for downstream analysis, not a new training run.

Completed DICC directories can remain in their existing locations: downstream
readers support them. If moving one manually, preserve every byte and the adjacent
detector provenance sidecar, verify before/after hashes, and ensure that only one
supported location remains. Do not edit absolute paths embedded in the completed
receipt. Do not regenerate frozen outputs or rewrite Git history.

For inventory, rename details, and verification evidence, see
[the naming refactor record](naming_refactor.md).
