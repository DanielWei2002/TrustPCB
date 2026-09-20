# RQ2 Stage 2: fresh detector and development checkpoint selection

This prepares one DICC training run. Do not execute training on the local Windows
machine. No calibration, transformations, risk scoring, test inference or final
test evaluation is part of this stage. Do not retrain on train + development.

## Frozen inputs

Stage 1 commit: `aaed8091323cac90ae5ae565719ce4b48a926335`.

| Role | Manifest under `data/splits/rq2_stage1_v2/` | Images | SHA-256 (LF canonical bytes) |
| --- | --- | ---: | --- |
| Fitting | `detector_train_images.txt` | 7,386 | `ec3f484e81364a1cd8619852578020aed9b9ce928eccf39cebc2613477155c06` |
| Development selection | `development_calibration_images.txt` | 821 | `1985f0d4097812462069be9c97cb672c4f6b3257ba8cf6a2d55243d3a7f3a6e1` |

The portable YAML template references exactly these two manifests. The runner
checks their complete ordered contents by hash, counts, uniqueness and disjointness.
CRLF checkout conversion is normalized to LF for these hashes. Physical
`images/train/` and `images/val/` folders do not determine scientific membership.
There is no `test` key or selectable dataset/split CLI argument. The reserved
2,052-image RQ2 test manifest is never passed to Ultralytics.

The existing baseline YAML supplies the frozen training settings:
`model=yolov8n.pt`, `seed=24209199`, `imgsz=640`, `epochs=100`, `batch=32`,
`deterministic=true`, `device=0`, `workers=2`, `patience=100`, `pretrained=true`,
`amp=true`, `save=true`, `val=true`. Existing `plots=true` is retained.
The runner adds fixed runtime paths, `exist_ok=false`, `resume=false` and
`split=val`; it accepts no hyperparameter overrides.

## DICC preparation and execution

Review and commit the preparation changes before DICC execution (the runner
requires clean committed execution/scientific inputs). Configure the existing
ignored `configs/local/paths.yaml` with the DICC project and dataset roots.

Keep exactly one original `yolov8n.pt` in either the repository root or
`notebooks/`. Its absolute path, SHA-256 and byte size must equal the existing
`runs/rq1/pretrained_model.json` binding. Missing/ambiguous weights, a missing
RQ1 binding, or a changed identity stop execution. Do not substitute an RQ1-trained
checkpoint or create a new binding to bypass this check. No automatic pretrained
weight download is used. Do not commit weights or local runtime configuration.

From the repository root, with the intended DICC Python environment activated:

```sh
# CPU-safe, text-only plan; does not create runtime files or load models.
PYTHONPATH=src python -B -m trustpcb.rq2_train plan

# DICC ONLY: launches the fresh GPU training worker.
PYTHONPATH=src python -B -m trustpcb.rq2_train train --dicc
```

The launcher sets `PYTHONHASHSEED=24209199` before the fresh worker starts.
Absolute train/dev manifests and YAML are generated under the ignored
`configs/local/datasets/rq2_stage2/`. It only constructs path strings at this
step; it does not scan images or labels.

Expected output:

```text
runs/rq2/stage2/
  seed_24209199.provenance.json
  seed_24209199/
    args.yaml
    results.csv
    weights/
      last.pt
      best.pt
      selection_candidate.pt
      selected.pt
```

## Exact checkpoint selection

After each epoch's CSV and `last.pt` are saved, `on_model_save` compares the stored
`metrics/mAP50-95(B)` decimal strings. A strictly higher score copies that epoch's
`last.pt` to `selection_candidate.pt`. Exact ties retain the earlier candidate.
This does not rely on Ultralytics' `best.pt`, whose internal fitness/tie behavior
can differ from the stored-score rule.

After normal return, all 100 contiguous epoch records and checkpoint callbacks
must be present. The runner independently selects the earliest maximum from the
complete CSV, checks the retained candidate identity, and exclusively creates
`weights/selected.pt`. This is the checkpoint frozen for later RQ2 stages.
The copy retains the epoch's serialized checkpoint, including its EMA weights;
no additional model loading or evaluation is used to select it. Normal training
validation (including Ultralytics' end-of-training validation) uses development
only. Selection does not use metrics from that end-of-training revalidation.

The callback ordering was inspected in the official
[Ultralytics v8.4.117 trainer source](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.117/ultralytics/engine/trainer.py).
The adapter checks callback order, CSV completeness, actual training arguments and
output paths and fails closed on mismatches. Actual DICC integration has not been
executed locally; record and review the installed package versions in provenance.

Ultralytics 8.4.117 calls `parse_device` before the pre-fit callback and before
saving `args.yaml`: the requested integer `0` becomes the string `"0"`. The
pre-fit and saved-argument checks accept exactly these two representations of
device 0; other devices, automatic selection, booleans and floating-point zero
are rejected. Other argument comparisons are unchanged. The requested config
remains integer `0`, and provenance preserves the actual serialized string.

## Provenance and restart policy

The sidecar records current Git commit, frozen Stage 1 commit/manifests/hashes,
source/config hashes, seed, requested and actual full training configuration,
runtime input hashes, original pretrained path/hash/size, Python/package versions,
timestamps, selected epoch/stored score/checkpoint path/hash/size, and CSV/args
hashes. Original pretrained and runtime input identities are checked before and
after training and before reuse.

An exclusive runner lock and sidecar reservation prevent concurrent/overwriting
runs. Incomplete runs, orphan output directories, altered runtime files or stale
locks require explicit manual inspection; no automatic resume, deletion or
overwrite is provided. A completed run is reused only if the plan, input identities,
selected checkpoint, CSV selection and artifact hashes all match. A different Git
commit also blocks reuse. A partial or failed run is not an approved detector.

### Pre-fit failure with only `args.yaml`

A directory containing only `args.yaml` has no checkpoint to resume. It is still
an occupied run directory and is **not reused automatically**. The runner neither
overwrites it nor automatically archives/renames it.

After confirming the failed process has exited and inspecting the artifacts,
manually archive `runs/rq2/stage2/seed_24209199/` to a distinct unused location.
Also archive the adjacent `runs/rq2/stage2/seed_24209199.provenance.json`, if present:
it is outside the run directory and independently blocks retry. Keep both for
diagnosis. Manual deletion of a verified failed attempt is an alternative, but
is not required if it has been archived out of these original paths. Never clear
a completed run to bypass its identity checks. Inspect any stale `.runner.lock`
and remove it only after confirming no launcher/worker remains active.

Fix the underlying pre-fit error before starting a fresh attempt with the same
documented DICC command. The `args.yaml`-only state by itself does not identify
that error. `git status` showing `?? runs/rq2/` is expected for untracked runtime
artifacts and does not dirty the scoped execution/scientific input check.

## CPU-safe tests

```sh
PYTHONPATH=src python -B -m unittest discover -s tests -v
```

Stage 2 tests use empty temporary dataset roots, committed manifest text and tiny
synthetic checkpoint bytes. They do not import PyTorch/Ultralytics or run models.
