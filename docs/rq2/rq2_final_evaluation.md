# Frozen RQ2 final-test evaluation

The 2,052-image similarity-aware test partition was **held out from RQ2 training
and development, previously used for RQ1 validation**. It is not historically
untouched. This workflow evaluates frozen development decisions; it cannot fit,
select or tune any model, calibrator, transformation, risk formula, weight or
referral threshold. The equal-weight full TrustPCB risk remains primary regardless
of the observed ordering of test metrics.

## Frozen identities and preflight

The portable manifest is `configs/datasets/similarity_aware_val_images.txt`:

```text
SHA-256 (LF canonical):
a2f246bf5be60cad5fac7f985ed91888bddb43601b73331f39d02e180075bc8e
```

Check the pinned manifest hash/count/uniqueness against the immutable partition
`verification_report.json`, whose LF-canonical hash is also pinned. Validate both
RQ2 train/development manifest identities and explicitly require disjoint membership.
Only membership text is read from these partitions, never their dataset images.
Use the existing semantic/historical resolver for recorded artifacts; ambiguous
current/historical checkpoint locations fail rather than silently choosing.

Selected detector: epoch 72, checkpoint SHA-256:

```text
793992d25ef671c45c820dc1b5a61bca5837af0d3726cd3830b2f2c656a6a85f
```

Require a completed, reviewed final Beta artifact with matching output and input
provenance hashes and exactly these coefficients:

```text
p_cal = sigmoid(a*log(p) - b*log(1-p) + c)
a = 0.9671379615387139
b = 2.2912177337533874
c = -0.6283462684200526
epsilon = 1e-15
```

The source artifact must identify Beta fitted on 7,131 development predictions
from 821 images, using the frozen 0.01 inclusion and primary IoU50 target. Existing
development artifacts are read-only. No fitter, comparison selector or training
function is called. Exact transformation strengths/order and critical inference
settings are checked before dataset execution.

## Reused implementation

| Operation | Existing implementation |
|---|---|
| Original inference | `rq2_confidence_distribution._predict`, optional settings and image-hash capture |
| Checkpoint identity | `rq2_confidence_distribution.checkpoint_identity` and artifact resolver |
| YOLO labels/coordinates | `rq2_calibration_analysis.read_ground_truth` |
| Independent IoU50/IoU75 correctness | `rq2_calibration_analysis.label_predictions` |
| Frozen Beta probabilities / calibration metrics | `rq2_calibrator_comparison.apply_calibrator`, `metric_summary` |
| Eleven transformed views/inference | `rq2_transformation_stability._predict` / `transform` |
| Inverse polygons and class-agnostic assignment | `rq2_transformation_stability.match_transform` |
| Family-balanced signals | `rq2_transformation_stability.aggregate` |
| Eight frozen risks / ranking / bootstrap / diagnostics | `rq2_risk_evaluation` pure functions |
| Secondary target adapter | `rq2_risk_sensitivity.sensitivity_rows` |

The small shared-adapter extensions preserve all development defaults: original
extraction can receive an explicit settings dictionary and capture source hashes;
the transformation adapter can label runtime provenance as final-test instead of
development. No transformation, matching or aggregation algorithm is rewritten.

## Execution order and frozen settings

1. Validate manifests, checkpoint, Beta artifact, clean scientific source state,
   local paths, required package versions and exclusive output reservation.
2. Read only the validated test image headers and corresponding YOLO labels.
   Missing labels fail; empty label files are permitted. Nontrivial EXIF orientation
   fails for explicit coordinate review, following existing correctness handling.
3. Infer all 2,052 original images with `conf=0.01`, `imgsz=640`, `iou=0.7`,
   `max_det=300`, `device=0`, `batch=1`, `rect=True`, `augment=False`, class-aware NMS.
4. Retain references at the fixed 0.01 floor. Assign stable image/source-row IDs,
   apply the frozen Beta mapping, and independently label IoU50 and IoU75 using
   descending raw confidence, unused same-class GT and highest eligible IoU.
   Store matched GT IDs, IoUs and duplicate/unmatched statuses.
5. Evaluate raw/Beta NLL, Brier and ten-bin ECE against primary IoU50 labels.
   Save all bin statistics. This is final-test evaluation of an already-fitted
   calibrator, not an in-sample refit or new calibration-method comparison.
6. Run the frozen eleven transformations per test image: **22,572 transformed
   passes**. Reuse brightness/contrast ±10%, sigma-0.6 blur, rotations ±2° and
   translations ±2%, per-channel median affine borders and unchanged dimensions.
   Use `conf=0.001`, `imgsz=640`, `max_det=300` and the other frozen settings.
7. Reuse inverse four-corner polygons, polygon IoU >=0.50, class-agnostic maximum
   match cardinality followed by maximum summed IoU, and deterministic tie ordering.
   Exclude non-evaluable pairs; assign zero to evaluable unmatched pairs. Aggregate
   within each family, then equally across evaluable families.
8. Compute all eight risks once, unchanged. Evaluate primary incorrect IoU50 and
   secondary incorrect IoU75 using these same score arrays.

Ultralytics must be **8.4.117**. Requested/effective inference settings, including
NMS IoU, device and package versions are recorded. Original and transformed
max-det saturation are explicit in the audit. Image hashes must agree between
original and transformed inference; image/annotation and frozen input hashes are
rechecked before completion.

## Ranking and uncertainty

Use unchanged AUROC (half credit for ties) and non-interpolated average precision
as AUPRC. Incorrect predictions are positive. Report counts and prevalence.
Each target uses **10,000 paired image-level replicates**, seed **24209199**,
sampling 2,052 images with replacement, including zero-prediction images.
Preserve image multiplicity and all predictions per sampled image. All methods
share each draw; primary/secondary draw hashes must also agree.

Undefined replicates are recorded without redrawing. The inherited policy is:
AUROC undefined without both classes; AP undefined without positives, AP=1 for
nonempty all-positive samples. Percentile 95% bounds use linear 2.5/97.5 quantiles
over valid values. Save every replicate's validity and metric values.

Repeat the five pre-specified AUROC differences, A minus B: full versus raw;
full versus calibrated; calibrated+class versus calibrated; calibrated+localisation
versus calibrated; class+localisation versus calibrated. Include observed delta,
bootstrap mean, bounds and valid/invalid counts. Never promote another combination
based on the test results.

Primary diagnostics reuse correct/incorrect group statistics (population standard
deviation, linear quartiles) and average-tied-rank Spearman correlations. Secondary
tables are labelled **Secondary IoU>=0.75 sensitivity analysis** and cannot override
the primary result. RQ3 referral/risk-coverage operating points are not computed.

## Future DICC command

After review and committing execution inputs, configure `configs/local/paths.yaml`
and use the frozen detector environment with NumPy, SciPy and Pillow installed.
From the repository root:

```bash
PYTHONPATH=src python -B -m trustpcb.rq2_final_evaluation --dicc
```

This performs GPU inference on **device 0**: 2,052 original passes plus 22,572
transformed passes, followed by CPU bootstrap analysis. Do not run it locally.
Windows and missing `--dicc` are rejected before dataset access. Unit tests inject
synthetic adapters; they never call real inference.

Output root: `runs/rq2/final_evaluation/`.

```text
original_predictions.csv
labelled_predictions.csv
calibration_metrics.json
calibration_bins.csv
transformed_predictions.csv
prediction_transform_matches.csv
prediction_stability_scores.csv
transformation_manifest.json
inference_audit.json
primary_iou50/
    method_metrics.csv
    pairwise_auroc_differences.csv
    bootstrap_metrics.csv
    bootstrap_summary.json
    signal_descriptives.csv
    signal_correlations.csv
    summary.json
sensitivity_iou75/
    method_metrics.csv
    pairwise_auroc_differences.csv
    bootstrap_metrics.csv
    bootstrap_summary.json
    summary.json
provenance.json
```

Prediction rows carry checkpoint identity and stable prediction IDs; the stability
table additionally includes all eight risk values. The root provenance contains
the test role/history, manifests, checkpoint/Beta identities, exact settings,
definitions, Git, versions, timestamps and recursive output SHA-256 hashes.

Any existing output directory blocks both completed-run overwrite and automatic
partial-run reuse. Failures retain incomplete provenance. Archive explicitly before
an approved retry. An empty retained prediction population stops with a clear
undefined-evaluation error. No development outputs are modified.

## Preparation validation and limitation

Local tests use synthetic manifest text, checkpoint bytes, labels, small arrays,
fake predictions and reduced bootstrap counts. They exercise both targets and
the complete orchestration without a fitting path. Real test images have not been
processed during implementation. GPU adapter integration and resource requirements
remain to be verified by the user on DICC; no local performance estimate is claimed.
