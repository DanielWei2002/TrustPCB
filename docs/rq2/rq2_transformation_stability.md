# RQ2 transformation stability

Development-only preparation for class consistency and localisation stability.
No retraining, calibration fitting, risk weighting or referral selection is part
of this workflow. Execute real image processing only on DICC.

## Frozen inputs

- Frozen 821-image development manifest at
  `data/splits/rq2_train_development_split/development_calibration_images.txt`.
- All 7,131 reference rows, in their original order, from
  `runs/rq2/final_calibrator/calibrated_development_predictions.csv`.
- Completed final Beta artifact/provenance under `runs/rq2/final_calibrator/`.
- Selected epoch-72 detector via the existing semantic/historical path resolver:
  `runs/rq2/detector_training/seed_24209199/weights/selected.pt`.
- Checkpoint SHA-256:
  `793992d25ef671c45c820dc1b5a61bca5837af0d3726cd3830b2f2c656a6a85f`.

Validation reuses the existing final-fit/comparison input validators, without
calling any fitting or selection operation. Require source population, manifest,
completed calibrator provenance and all recorded artifact hashes to match.
Verify finite nonnegative Beta a/b, unchanged form/epsilon/threshold, and confirm
stored calibrated values match the frozen mapping (absolute tolerance 1e-12).
Original row fields, identity and order must match the upstream labelled table.
The mapping is applied only for artifact validation, never refitted.

Only development membership is opened; physical `images/val/` paths can legitimately
belong to development membership. No final-test membership file, image or annotation
is opened. Existing correctness columns are carried forward, not used for matching.
Exactly the validated development paths are passed to the image adapter, with
resolved-path containment checks. All 821 images receive eleven views, including
images with no reference predictions.

## Eleven deterministic views

| Family | Variants | Implementation |
|---|---|---|
| Brightness | 0.90, 1.10 | Multiply intensities by factor |
| Contrast | 0.90, 1.10 | `mean + factor*(pixel-mean)`; scalar original intensity mean over pixels and channels |
| Blur | sigma 0.6 pixels | SciPy Gaussian filter, spatial axes only, reflect boundary, truncate 4.0 |
| Rotation | -2°, +2° | About `(width/2,height/2)`, positive visually counterclockwise |
| Translation | x ±2% width; y ±2% height | Exact fractional pixel displacement; no rounding of displacement |

Images are decoded with Pillow, EXIF orientation applied, converted to RGB, then
passed as BGR arrays to Ultralytics. Every view starts from the original image;
transforms are never composed. All operations use float64, followed by clipping
to [0,255], nearest-even rounding and uint8 conversion. Original dimensions remain
unchanged; transformed images are not saved.

### Affine rasterisation and geometry

Use xy edge coordinates with bounds `[0,width] x [0,height]`. Pixel centres are
`(column+0.5,row+0.5)`. Forward 3x3 homogeneous matrices define both rasterisation
and geometry. SciPy inverse resampling uses bilinear interpolation (`order=1`,
`prefilter=False`, `mode="grid-constant"`). The outside constant is the **original
image's per-channel median**, with interpolation across the border. Per-image
medians and forward matrices are recorded for every view.

For rotation/translation, transform all four original reference-box corners.
The reference is evaluable only when every corner remains within the inclusive
image bounds. No clipping or tolerance expands these bounds. Photometric/blur
views are always evaluable. Non-evaluable references do not participate in that
view's assignment and cannot consume transformed detections.

Inverse-map all four corners of each transformed detector box using the exact
inverse matrix. Keep the resulting polygon, including portions outside the original
canvas; never substitute an enclosing axis-aligned rectangle. The same centre
convention is used for resampling and geometric mapping.

## Frozen inference settings

Reuse `rq2_confidence_distribution.SETTINGS`, including:

- `conf=0.001`, `iou=0.7`, `max_det=300`, `imgsz=640`, `device=0`;
- `batch=1`, `rect=True`, `augment=False`, class-aware NMS;
- no class filter, saved crops, detector labels or visualisation outputs.

Require Ultralytics **8.4.117** and the frozen detector class map. Check the effective
settings after each call and record the full effective predictor arguments in
provenance (replace the source array with its descriptive type). Also record the
resolved device and NumPy/SciPy/Pillow/Ultralytics/torch versions.

The 0.01 inclusion rule applies exclusively to references. Transformed predictions
at 0.001 through below 0.01 remain eligible. A view returning exactly 300 detections
is flagged in its prediction rows, matching rows and per-view summary, including
views without reference predictions. Counts above 300 fail validation.

## Polygon IoU and assignment

Polygon area uses the shoelace formula. Deterministic Sutherland-Hodgman clipping
intersects convex polygons, accepting either winding direction. IoU is intersection
area divided by union area; degenerate polygons have zero IoU. Use this same
routine for all views; photometric polygons remain axis-aligned rectangles.

Assignment is class-agnostic. Only polygon-IoU edges **>= 0.50** are eligible.
For n evaluable references and m detections, give each eligible edge reward
`min(n,m)+1+IoU`; give invalid edges -1 and each dummy unmatched column zero.
SciPy linear assignment maximises reward. One additional valid match dominates
every possible total-IoU difference; within a fixed cardinality, total IoU decides.
This does not maximise unrestricted IoU and discard invalid pairs afterward.

Keep reference rows in source order. Sort transformed detections by xyxy, then
original detector index to stabilise exact ties. Class and confidence do not
enter ordering, edge eligibility or either optimisation objective. Use SciPy's deterministic
assignment tie handling for that fixed ordering; record its version for reproduction.
No epsilon perturbation changes the secondary IoU objective.

## Signals and equal family weighting

For each reference and view:

- Evaluable and matched: class value is 1 for the same predicted class, otherwise
  0; localisation value is mapped polygon IoU.
- Evaluable and unmatched: both values are 0.
- Non-evaluable: both values are missing, excluded from denominators.

For each signal `s` and family `f`, calculate
`family_score[f] = sum(s over evaluable variants) / n_evaluable_variants`.
Then `overall_score = mean(family_score[f] over families with evaluable variants)`.
A family with no evaluable variants is missing. Translation has the same family
weight as blur despite having four variants. Report the ten family scores, two
overall scores and counts of evaluable views, matched views and evaluable families.

## DICC execution

After review and committing the execution inputs, configure the existing
`configs/local/paths.yaml`. Use the detector environment plus NumPy, SciPy and
Pillow. From the repository root:

```bash
PYTHONPATH=src python -B -m trustpcb.rq2_transformation_stability --dicc
```

GPU device **0** is used for inference; transformations/geometry use CPU. There
are 9,031 view-inference calls. Do not execute this command locally. The CLI and
real inference adapter reject Windows. Synthetic tests inject a fake predictor.

Output: `runs/rq2/transformation_stability/`.

| File | Audit contents |
|---|---|
| `transformation_manifest.json` | Exact eleven specs and implementation rules |
| `transformed_predictions.csv` | View/detection IDs, class, confidence, transformed xyxy and inverse-mapped four-corner polygon; saturation flag |
| `prediction_transform_matches.csv` | Every reference × eleven views: evaluability, match ID, mapped IoU, both signal values and saturation flag |
| `prediction_stability_scores.csv` | Original reference rows/order plus family and aggregate scores/counts |
| `summary.json` | View coverage/counts/saturation, dimensions, forward matrices, median borders, source-image hashes and match/evaluability totals |
| `provenance.json` | Git, input/output hashes, checkpoint identity, rules, package versions, requested/effective inference settings and completion status |

Existing complete or incomplete output directories always block reuse. Failures
leave incomplete provenance; archive explicitly before a reviewed retry. Source
tables, calibrator and checkpoint identity are rechecked before completion;
source image bytes are hashed before/after each image's views. No completed
scientific output is rewritten. Only consume outputs with complete provenance
and matching recorded hashes.

## Local validation and limitations

Tests use tiny synthetic pixels/polygons, synthetic text provenance fixtures and
fake detector results only. They cover transformations, geometry, cardinality-first
matching, family denominators, saturation and overwrite protection. Real detector
integration/performance remains to be verified on DICC. Exact GPU outputs can
depend on the recorded hardware/software environment; no claim of cross-platform
bitwise GPU reproducibility is made.

Implementation references: [SciPy affine resampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html),
[Gaussian filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html),
[linear assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).
