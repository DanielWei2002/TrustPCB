# RQ1 notebook conversion and recovery

All three historical notebooks are represented by static Python workflows under
`src/trustpcb/rq1/`. These are **new reconstructions**, not the exact programs
originally executed. No real dataset processing, split search, training, validation,
or scientific metric generation was performed during conversion. Frozen evidence
and its historical provenance remain authoritative and unchanged.

## Python entry points

- [audit_workflow.py](../../src/trustpcb/rq1/audit_workflow.py): inventory, image and
  YOLO-label validation, class/box diagnostics, exact duplicates, pHash screening,
  secondary checks, tiering, translation analysis, review pages and human decisions.
  Reuses [dataset_audit.py](../../src/trustpcb/rq1/dataset_audit.py) numerical kernels.
- [split_workflow.py](../../src/trustpcb/rq1/split_workflow.py): supplied inventory,
  confirmed-pair loading, group composition, class distributions, allocation,
  manifest generation and validation. Reuses
  [similarity_split.py](../../src/trustpcb/rq1/similarity_split.py) graph/search kernels.
- [preliminary_baseline.py](../../src/trustpcb/rq1/preliminary_baseline.py): environment,
  split/class checks, shared runtime configuration, explicit smoke/supplied/similarity
  training commands and saved-log comparison. Reuses `dataset_config.py` and
  [workflow.py](../../src/trustpcb/rq1/workflow.py) settings/first-best-epoch extraction.
  `workflow.py` remains the unchanged matched repeated-training implementation.
- [archived_sources.py](../../src/trustpcb/rq1/archived_sources.py): immutable Git-source
  verification for tests and reporting. It reads JSON as data, never executes cells.
- [reconstruction.py](../../src/trustpcb/rq1/reconstruction.py): shared stage wiring,
  sorted inventory, DICC gate and fresh-output guards.

Functions expose named inputs and outputs; no notebook execution engine or runtime
source extraction is required by research workflows. Heavy libraries are lazy.

## Cell coverage (zero-based indexes)

| Notebook | Cells | Representation |
|---|---|---|
| 01 audit | 0–3 | Shared local paths/inventory, printed examples and existing `check_pairs` |
| 01 audit | 4–7 | `inspect_images`, `validate_annotations`, `describe_boxes`, `exact_duplicates` |
| 01 audit | 8–10 | `compute_phash`, `screen_phash`, `verify_candidates`, `shortlist` |
| 01 audit | 11–13 | `show_tier_a`, `align_candidates`, `show_final_review`; PNG pages replace inline display |
| 01 audit | 14 | `record_manual_review`; filename-keyed A/B input replaces hard-coded row-number decisions |
| 01 audit | 15 | `write_audit_summary` accepts explicitly reviewed metric/value rows; fixed historical values/conclusion remain archived evidence |
| 01 audit | 16 | Empty cell; no executable content |
| 02 split | 2, 4, 6 | Shared paths, `supplied_inventory`, `load_confirmed_pairs` |
| 02 split | 8, 10, 12 | `construct_groups`, `group_classes`, `supplied_distribution` |
| 02 split | 14, 16, 18 | `feasible_allocations`, `allocate`, `compare_distribution` |
| 02 split | 20, 22 | `write_manifests`, `validate_split` |
| 02 split | 24 | Empty cell |
| 03 baseline | 2, 4, 6 | `verify_environment`, `verify_split_inputs`, `verify_class_mapping` |
| 03 baseline | 8, 9 | Existing `generate_runtime_configs`, `validate_runtime_configs` |
| 03 baseline | 12 | Frozen existing configuration, `seed_environment`, `training_options`; no rewrite of the frozen YAML |
| 03 baseline | 14, 16, 18 | `train` with explicit smoke/supplied/similarity stage; original settings and output names retained |
| 03 baseline | 20 | `compare_results`, using unchanged `workflow.read_metrics`; first maximal logged mAP50–95 epoch |
| 03 baseline | 22 | Empty cell |
| 02/03 | All markdown | [Historical explanatory notes](historical_notebook_notes.md), retained verbatim with historical labeling |

## Deliberate boundaries and preserved conventions

The audit keeps the 32x32 grayscale DCT pHash, non-DC median, 64-bit packing,
64-image comparison batches, thresholds 0/2/4/6/8/10, normalized 128x128 MAE/Pearson,
original A/B/C filters, ±10 translation search, 0.85 overlap, first strict-MAE tie,
0.015/0.985 final shortlist, six review pairs per page and fivefold displayed
differences. Annotation boundary overflow remains diagnostic rather than silently
clipped or rejected. Class/area summaries retain original formulas and thresholds.

Manual A/B judgments cannot be inferred from numerical similarity. The automated
audit stops with blank decision fields. Review input must identify every exact
filename pair; the historical 24 row-number labels are never reapplied to a new
ordering. Original decisions, all inline PNGs, printed outputs and execution counts
remain recoverable in Git. Two representative existing PNGs are already available
in [RQ1 reporting](../../docs/rq1/RQ1_Research_Journal.docx); additional panels are not regenerated.
The fixed cell-15 summary and its prose conclusion are historical records, not an
algorithm that may declare a new dataset valid. Explicitly reviewed summary rows
can be serialized without inventing a replacement aggregation or conclusion.

The split code preserves nine classes, sorted graph traversal, the historical
`.jpg` label exclusion, exhaustive 9/10-group choices and mean/max percentage-change
ranking. Its runner rejects populations other than 19 disjoint train/val JPG pairs
because the original pair-based accounting does not generalize safely. Validation
still requires 10,259 images and 19 changed assignments. Unaffected membership is
preserved; data files are never moved or copied. Outputs use a fresh destination.

Baseline settings remain seed 24209199, YOLOv8n, 640 pixels, batch 32, 100 epochs,
workers 2 and the original deterministic/AMP settings; smoke is two epochs and is
not baseline evidence. Set `PYTHONHASHSEED=24209199` before launching Python for an
effective interpreter hash seed (the old cell set it after startup). Comparison
uses the shared strict complete-100-epoch log validator; incomplete or malformed
logs fail instead of being presented as a complete baseline. No per-metric maxima
are combined. The introductory notebook mentions efficiency/prediction objectives
but contains no separate implementations of those stages; none is fabricated.

The unchanged checkpoint resolver still accepts a legacy DICC `notebooks/yolov8n.pt`
location as well as the repository-root checkpoint. This is compatibility with an
external checkpoint path, not a dependency on any `.ipynb` or a local notebook folder.

## Original archive and hash verification

[notebook_archive.json](notebook_archive.json) records all original Git paths,
commit `1937995fb78d979a66e237b7ab7facbe20ca0513` (existing main), blob object IDs,
Git-blob SHA-256, exact historical Windows-checkout SHA-256 and newline transforms.
All three blobs also exist unchanged at pre-conversion HEAD
`ef86c125b07c5b875254d1916eca65963110fb6d`.

Git stores LF bytes, while the recorded Windows copies have CRLF (notebook 03 has
mixed endings). Verification first checks the immutable Git blob hash, then applies
the explicit recorded newline transform and checks the **unchanged old source
hash**. Neither validation is skipped. Full history is required; missing Git objects
fail closed. To inspect an original without restoring the directory:

```text
git show 1937995fb78d979a66e237b7ab7facbe20ca0513:notebooks/01_dataset_structure_audit.ipynb
git show 1937995fb78d979a66e237b7ab7facbe20ca0513:notebooks/02_similarity_aware_split.ipynb
git show 1937995fb78d979a66e237b7ab7facbe20ca0513:notebooks/03_yolo_baseline.ipynb
```

Use binary subprocess output when restoring byte-exact Git blobs; PowerShell text
redirection can change encoding/newlines. The recorded Windows representation can
be obtained through `archived_sources.read_source`. No restoration is needed for
ordinary Python workflows. Tests compare selected pure historical algorithms on
tiny fixtures using the verified Git archive.

RQ1 reporting copies/figures remain byte-identical. Its manifest now distinguishes
archived sources from current files, retaining the generator version that actually
created each existing figure. Updated authored navigation is explicitly recorded
as a conversion edit, not a rerender. RQ2/RQ3 reporting and all frozen scientific
provenance remain unchanged. The [artifact inventory](../artifact_inventory.md) is
the preserved pre-conversion snapshot; its notebook-location statement is historical.
The unidentified producer of RQ1 paired-summary/difference tables remains unresolved.

## DICC handoff (prepare only; not executed locally)

Dependencies: NumPy, pandas, Pillow, tqdm, OpenCV and Matplotlib for audit; NumPy
and pandas for splitting; PyYAML, PyTorch and Ultralytics for baseline execution.
Use the existing DICC research environment. Local tests use only synthetic data.
All research CLIs reject Windows execution, even with `--dicc`.
Set `PYTHONPATH=src` in the DICC checkout. Replace the explicit example paths:

```text
python -m trustpcb.rq1.audit_workflow audit --project-root /path/to/TrustPCB --output /scratch/rq1-new-audit --dicc
python -m trustpcb.rq1.audit_workflow review --project-root /path/to/TrustPCB --shortlist /scratch/rq1-new-audit/cross_split_final_manual_review.csv --decisions /scratch/reviewed-decisions.csv --output /scratch/rq1-new-review --dicc
python -m trustpcb.rq1.split_workflow --project-root /path/to/TrustPCB --audit-tables /scratch/rq1-new-review --output /scratch/rq1-new-splits --dicc
python -m trustpcb.rq1.preliminary_baseline verify --project-root /path/to/TrustPCB --dicc
python -m trustpcb.rq1.preliminary_baseline smoke_test --project-root /path/to/TrustPCB --version-run-dir /scratch/rq1-reconstruction/v1 --dicc
python -m trustpcb.rq1.preliminary_baseline supplied --project-root /path/to/TrustPCB --version-run-dir /scratch/rq1-reconstruction/v1 --dicc
python -m trustpcb.rq1.preliminary_baseline similarity_aware --project-root /path/to/TrustPCB --version-run-dir /scratch/rq1-reconstruction/v1 --dicc
python -m trustpcb.rq1.preliminary_baseline compare --project-root /path/to/TrustPCB --version-run-dir /scratch/rq1-reconstruction/v1 --output /scratch/rq1-new-comparison --dicc
```

These are separate prospective runs, not instructions to refresh frozen evidence.
Baseline uses the committed frozen split templates through shared runtime config;
it does not silently adopt newly generated split candidates. New audit/split outputs
require independent scientific review before any future protocol adopts them.

Tests: [tests/rq1](../../tests/rq1/). Reporting verification remains
`python -B -m trustpcb.rq1.reporting --verify-only`. No scientific result is recalculated.
