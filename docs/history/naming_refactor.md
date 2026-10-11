# Semantic naming refactor record

## Proposed and actual rename map

The following proposal was reported before editing and then applied. Module names
also rename their matching `tests/test_<module>.py` files and all active imports.

| Previous name | Current name | Action |
| --- | --- | --- |
| `src/trustpcb/rq2_split.py` | `src/trustpcb/rq2_data_partition.py` | Tracked move |
| `src/trustpcb/rq2_train.py` | `src/trustpcb/rq2_detector_training.py` | Tracked move |
| `src/trustpcb/rq2_confidence.py` | `src/trustpcb/rq2_confidence_distribution.py` | Tracked move |
| `docs/rq2_stage1.md` | `docs/rq2_data_partition.md` | Tracked move |
| `docs/rq2_stage2.md` | `docs/rq2_detector_training.md` | Tracked move |
| `docs/rq2_stage3b1.md` | `docs/rq2_confidence_distribution.md` | Tracked move |
| `data/splits/rq2_stage1_v2/` | `data/splits/rq2_train_development_split/` | Three tracked files moved without byte changes |
| `runs/rq2/stage2/` | `runs/rq2/detector_training/` | Future output definition; no local run directory present |
| `runs/rq2/stage3b1/raw_confidence/` | `runs/rq2/confidence_distribution/` | Future output definition; no local run directory present |
| `configs/local/datasets/rq2_stage2/` | `configs/local/datasets/rq2_detector_training/` | Future runtime definition; no local runtime directory present |

The optional rejected partition uses `data/splits/rq2_rejected_train_development_split/`
for future naming. Its historical location remains a read-only alias. There are no
rejected partition files in this checkout.

Tracked moves used `git mv`. The complete naming-only changes, including reference
edits, are staged for review. No commit or push was performed.

## Occurrence classification

**A — active references:** Python modules/imports, test aliases/classes, CLI examples,
dataset-template manifest references, output/runtime destinations and future
provenance keys/identifiers were updated. README links point to semantic documents.

**B — frozen records:** the partition verification report is moved unchanged,
including its original protocol, rejected-partition identity and scope fields.
The four historical aliases in the path-compatibility module intentionally retain
original identifiers to read immutable DICC artifacts and block accidental reruns.
They are not names for future outputs. Tests exercise these aliases via that mapping.
No completed RQ2 run files/checkpoints are present locally to rename or rehash.

**C — Git history:** no history or existing commit message was changed. The accepted
partition commit remains `aaed8091323cac90ae5ae565719ce4b48a926335`.

The previous-name column above is a migration record, not an active command or
destination. This document, the explicit alias mapping and the immutable report
are the only intentional workflow-number references in the current text inventory.

### Complete remaining-occurrence inventory

| File | Lines | Why retained |
| --- | --- | --- |
| `src/trustpcb/rq2_artifact_paths.py` | 11, 12, 13, 14 | Four explicit aliases for unchanged historical artifact locations; required for reading and no-rerun protection |
| `data/splits/rq2_train_development_split/verification_report.json` | 532, 658, 666 | Original protocol, rejected-partition identity and scope in frozen scientific provenance; byte-identical preservation |
| `docs/naming_refactor.md` | 13-19 | Seven previous-name entries in the rename map above; migration documentation only |
| `notebooks/01_dataset_structure_audit.ipynb` | 1145 | Generic audit comment, not a workflow-number identifier, path or experiment name; existing notebook left unchanged |
| `notebooks/02_similarity_aware_split.ipynb` | 312 | Generic explanatory prose, not a workflow-number identifier, path or experiment name; existing notebook left unchanged |

Git's ordinary staged/unstaged terminology in RQ1 is unrelated to research naming
and is unchanged. Historical Git messages and ignored local audit snapshots are
not active repository identifiers.

## Scientific preservation

The two moved manifests and their frozen verification report preserve their exact
pre-refactor working-tree bytes (including line endings). All other split files,
the baseline experiment configuration, RQ1 code and historical runs are untouched.
LF-canonical manifest identities remain:

| Partition | Images | SHA-256 |
| --- | ---: | --- |
| Detector training | 7,386 | `ec3f484e81364a1cd8619852578020aed9b9ce928eccf39cebc2613477155c06` |
| Development/calibration | 821 | `1985f0d4097812462069be9c97cb672c4f6b3257ba8cf6a2d55243d3a7f3a6e1` |

Split/training seed 24209199, reserved-test membership (2,052), selected epoch 72,
and confidence extraction settings remain unchanged. The separate uncommitted
calibration workflow retains its frozen inclusion, matching and metric definitions. The selected checkpoint hash constraint is still
`793992d25ef671c45c820dc1b5a61bca5837af0d3726cd3830b2f2c656a6a85f`.
The checkpoint itself is absent locally; its actual bytes cannot be independently
rehash-verified here. The DICC reader still verifies the frozen hash before use.

## Current DICC entry points

These names replace previous module commands. They are documentation, not commands
executed during the refactor. Do not regenerate completed experiments.

```sh
PYTHONPATH=src python -B -m trustpcb.rq2_data_partition --dicc
PYTHONPATH=src python -B -m trustpcb.rq2_detector_training plan
PYTHONPATH=src python -B -m trustpcb.rq2_detector_training train --dicc
PYTHONPATH=src python -B -m trustpcb.rq2_confidence_distribution plan
PYTHONPATH=src python -B -m trustpcb.rq2_confidence_distribution extract --dicc
```

## Commit boundary and verification

This first change set contains only renaming and compatibility for the previously
committed data-partition, detector-training and confidence-distribution workflows.
The calibration implementation, test and document remain separate and untracked:

- `src/trustpcb/rq2_calibration_analysis.py`
- `tests/test_rq2_calibration_analysis.py`
- `docs/rq2_calibration_analysis.md`

Its future output remains `runs/rq2/calibration_analysis/`. No scientific matching
or calibration implementation is included in the naming-only index. README links
and executable command examples in this change set resolve without these files.
The calibration-specific historical-output guard lives with that uncommitted code.

Validation covers both the complete working-tree CPU-safe suite and an isolated
copy of the index's source, tests and text fixtures. The latter excludes the three
uncommitted calibration files and proves the first change set stands on its own.
Frozen manifests and the verification report are compared byte-for-byte against
both the original Git blobs and the pre-refactor local hash inventory. No actual
checkpoint is present locally; the pinned checkpoint hash remains unchanged.

No real experiment, inference, model loading, dataset processing or GPU work runs
as part of this validation. The index is prepared for review, not committed.

Recommended commit message: `Refactor RQ2 workflow to semantic research names`.
