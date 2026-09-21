# RQ2 Calibration analysis: correctness labels and raw calibration

This analysis analyzes existing development predictions on CPU. It does not load the
detector, rerun inference, fit any calibrator, select a threshold/method, evaluate
test images, or perform transformation, stability, risk or referral analysis.

## Frozen boundaries and inputs

Only the 821-image development manifest from Data partition v2 is accepted, with its
existing LF-canonical SHA-256. Confidence distribution must be complete under
`runs/rq2/confidence_distribution/`, and its provenance must identify epoch 72 and
checkpoint SHA-256
`793992d25ef671c45c820dc1b5a61bca5837af0d3726cd3830b2f2c656a6a85f`.
The prediction/inventory CSV hashes must match that provenance, and the inventory
must cover all 821 development images with consistent counts. Unknown images are
rejected even if their predictions would be excluded by the inclusion threshold.
No test manifest, test labels or test images are opened. Physical `images/val/`
paths are not an indication of scientific test membership.

**Raw confidence >= 0.01 is fixed before correctness analysis.** There is no CLI
override or code path that chooses a different threshold from the results.
Predictions below 0.01 are excluded before either matching pass. Data partition, Detector training
and Confidence distribution files are read-only inputs and are never rewritten.

## Matching and coordinates

For each image, predictions are sorted by descending raw confidence. Exact ties
retain original Confidence distribution CSV row order. Each pass considers unused ground-truth
boxes of the predicted class and chooses the highest IoU. Exact IoU ties retain
label-file line order. A match consumes its ground-truth box only if IoU meets
the pass's inclusive threshold. The primary pass is IoU >= 0.50; IoU >= 0.75 is
an independent sensitivity pass with a fresh used-ground-truth set.

An unmatched prediction is marked `duplicate` if a qualifying same-class box has
already been consumed in that pass; otherwise it is `unmatched`. A qualifying
unused box takes precedence over duplicate status. Neither a wrong-class box nor
an already-used box can become a match. No ground truth is consumed on failure.

Predictions use original-image xyxy pixel coordinates. YOLO five-column label
files are resolved only from development IDs (`images/...` to `labels/...txt`).
Normalized label centers/widths/heights are converted using image-header dimensions.
IoU uses continuous geometry, without a +1 pixel convention. Missing or malformed
labels fail; an existing empty label file is valid. Nontrivial EXIF orientation
fails for explicit coordinate review rather than guessing how boxes were oriented.
Ground-truth IDs are portable label paths plus original one-based line numbers.

## Raw numerical diagnostics

The binary target is whether a retained prediction matches under the indicated
IoU rule. Brier score is mean `(confidence - correct)^2`. NLL is mean Bernoulli
negative log-likelihood using natural logs. Only NLL clips probabilities to
`[1e-15, 1 - 1e-15]` to keep endpoints finite; matching, bins, Brier and ECE use
the original confidence.

Reliability/ECE use **10 fixed equal-width bins**:
`[0,0.1), [0.1,0.2), ..., [0.9,1.0]`. Every bin is lower-inclusive and upper-exclusive
except the final bin, which includes 1.0. Inclusion at 0.01 means the first bin
contains no scores below 0.01. ECE is the prediction-count-weighted mean absolute
difference between each bin's mean confidence and empirical correctness rate.
Empty bins have null means/rates, contribute zero to ECE and are omitted from the
reliability curve. A completely empty retained population has null metrics/rate.

These are prediction-level raw calibration diagnostics conditional on inclusion
and Confidence distribution NMS/max_det settings. Missed ground-truth objects are not additional
negative prediction rows. These metrics are not detector recall or test performance.
No results automatically choose a calibrator, threshold or analysis method.

## DICC execution

Use the configured `configs/local/paths.yaml` and a reviewed, committed, clean
checkout. Pillow and Matplotlib are required; PyTorch/Ultralytics are not imported
or required to execute this analysis. GPU is not needed.

From the repository root on DICC:

```sh
PYTHONPATH=src python -B -m trustpcb.rq2_calibration_analysis analyze --dicc
```

Local execution of the real development analysis is blocked. Synthetic unit tests
remain CPU-safe:

```sh
python -B -m unittest discover -s tests -v
```

## Expected outputs

Under `runs/rq2/calibration_analysis/`:

- `labelled_predictions.csv`: image, class ID/name, raw confidence, xyxy box,
  original source row, `correct_at_iou50` and `correct_at_iou75`, and for each pass
  matched ground-truth ID/IoU, best unused same-class IoU, best same-class IoU
  regardless of availability, and matched/duplicate/unmatched status. Unmatched
  IDs/matched IoUs are blank; best IoU is zero when no candidates remain.
- `summary.json`: extracted/excluded/retained counts, correct/incorrect counts
  and rates, raw Brier/NLL/ECE plus reliability-bin data for both IoU definitions,
  and corresponding counts/metrics grouped by predicted class.
- `raw_reliability_iou50.png`: primary raw reliability diagram; points show mean
  confidence versus correctness within each nonempty fixed bin.
- `retained_confidence_histogram.png`: raw confidence histogram after fixed inclusion.
- `ground_truth_provenance.json`: portable label paths/hashes, dimensions and
  object counts for every development image.
- `provenance.json`: completion status, Git commit, frozen identities, fixed
  analysis thresholds, Confidence distribution input hashes, package versions, timestamps and
  output hashes.

The destination is reserved exclusively. Partial and completed directories both
block reruns; no overwrite, resume or automatic deletion is provided. Inspect and
manually archive a failed attempt before retry. Only `status: complete` denotes a
finished analysis. Actual development statistics/plots await DICC execution.
