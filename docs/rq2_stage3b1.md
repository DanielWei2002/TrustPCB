# RQ2 Stage 3B1: raw development confidence

Preparation only; execute the real 821-image extraction on DICC. No calibration,
correctness matching, test evaluation, threshold recommendation, transformations,
stability/risk scoring or referral analysis is implemented here.

## Frozen inputs and scope

- Epoch 72: `runs/rq2/stage2/seed_24209199/weights/selected.pt`.
- SHA-256: `793992d25ef671c45c820dc1b5a61bca5837af0d3726cd3830b2f2c656a6a85f`.
- Development: `data/splits/rq2_stage1_v2/development_calibration_images.txt`,
  821 images, LF-canonical SHA-256
  `1985f0d4097812462069be9c97cb672c4f6b3257ba8cf6a2d55243d3a7f3a6e1`.

Exact hash, membership order and uniqueness are checked. There are no CLI options
to substitute another checkpoint, source list or partition. Only the development
paths are passed individually to prediction, in manifest order. Physical
`images/val/` storage does not denote scientific test membership. The test manifest,
training manifest and annotation files are not opened. No correctness labels are
assigned: future same-class one-to-one matching at IoU 0.50/0.75 is out of scope.

## Reproducible prediction settings

Use the detector environment with Ultralytics **8.4.117** and Matplotlib installed.
The version guard refuses other Ultralytics versions. Frozen prediction options:
confidence **0.001**, NMS IoU **0.7**, maximum detections **300**, image size **640**,
device **0**, batch **1**, rectangular padding, class-aware NMS, all classes,
no test-time augmentation, default full precision (`quantize=None`). Image overlays
and Ultralytics file exports are disabled; this workflow writes its own tables.

IoU/max_det/class-aware NMS and precision follow the
[versioned Ultralytics defaults](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.117/ultralytics/cfg/default.yaml).
The minimum confidence changes only for this exploratory extraction. **0.001 is
not a selected calibration inclusion threshold.** Scores are raw, uncalibrated
post-NMS detection confidences, not all dense network candidates. NMS and max_det
truncate the population. The report flags images reaching max_det without changing
it. Counts cannot characterize detections suppressed by NMS or below the floor.

## Commands (repository root on DICC)

Review/commit these execution inputs and transfer them to DICC first. The runner
requires clean execution/scientific sources. Configure ignored
`configs/local/paths.yaml` for this checkout and dataset. No weights are downloaded.

```sh
# Text-only plan, safe locally; no checkpoint load/hash or image access.
PYTHONPATH=src python -B -m trustpcb.rq2_confidence plan

# DICC ONLY; GPU 0 is recommended and fixed for reproducibility.
PYTHONPATH=src python -B -m trustpcb.rq2_confidence extract --dicc
```

## Outputs

Under `runs/rq2/stage3b1/raw_confidence/`:

- `predictions.csv`: portable image ID, predicted class ID/name, raw confidence,
  x1/y1/x2/y2 in original-image pixel coordinates. No rounding of confidence and
  no correct/incorrect target. Rows preserve image and detector output order.
- `images.csv`: all 821 IDs and prediction counts, including zero detections.
- `summary.json`: total detections, images with detections, min/median/mean/max
  predictions per image and confidence, per-class counts, seven requested interval
  counts/percentages, and retained counts/percentages at 0.001/0.01/0.05/0.10/0.25.
  Interval upper bounds are exclusive except the final bound 1.00. Percentages
  use total extracted predictions. Empty populations give zero percentages and
  null confidence statistics. No final threshold is chosen.
- `confidence_histogram.png`: one 100-bin confidence histogram, not reliability.
- `provenance.json`: completion status, Git commit, package versions, requested
  and effective settings, pinned checkpoint/epoch/hash/size and development hash,
  timestamps and output hashes. Checkpoint/input identities are rechecked after
  extraction. Only a `complete` report is usable.

The output directory is created exclusively. Both partial and completed output
directories block retry; nothing is automatically deleted, resumed or overwritten.
Inspect failures and manually archive partial artifacts before a fresh attempt.
Stage 2 files are never modified. The checkpoint hash is the binding evidence for
the accepted epoch; this stage does not load optimizer state or rerun selection.

Local tests use tiny synthetic predictions and inert checkpoint bytes. Actual
Ultralytics inference and the real confidence population remain unverified until
the DICC command runs. No threshold recommendation will be made automatically.
