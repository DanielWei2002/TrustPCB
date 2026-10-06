# RQ2 Data partition: detector-training/development membership

Data partition partitions only the 8,207-image similarity-aware training pool, with
seed **24209199** and a target of 821 development/calibration images (7,386
detector-training images). Existing similarity-aware validation membership
(2,052 images) is reserved for RQ2 test use and is never used for balancing.
Only its membership is read to check disjointness and existing group boundaries.
No reserved-test image, label, or model-performance information is read.

## Inputs and method

The generator cross-checks `configs/datasets/similarity_aware_*_images.txt`
against `data/splits/similarity_aware_*.txt` and the scientific split CSV.
It reconstructs connected components from
`outputs/tables/cross_split_high_confidence_near_duplicates.csv`, the confirmed
edge records used by notebook 02. Transitive links are joined before allocating
groups; unknown endpoints or components crossing the existing pool/test boundary
are errors. It does not rerun similarity detection or notebook 02 allocation.

Per-image annotation counts come only from labels referenced by the source
manifest. Physical `images/train` versus `images/val` storage is retained; an
image moved scientifically may still live in its original physical folder.
Images are never decoded. Label reading is dataset-wide source preprocessing,
so real generation belongs on DICC under the repository compute boundary.

## Revised allocator (v2)

The v1 rare-class-first allocation was rejected for class imbalance despite
valid membership. V2 has no rare-class-first rule. Every component has a vector
of its image count, per-class image-presence counts, and annotation counts.
Singleton images remain components. Canonical components get seeded tie ranks;
identical feature vectors are bucketed for efficiency without merging membership.

The class objective is mean squared deviation, in **percentage points squared**,
of each development/source class ratio from **10%**. Image-presence and annotation
features have equal weight; absent source classes are excluded, not counted as
zero-error observations. Count proximity to 821 is a separate constraint.

1. At each greedy step, compare all available component feature types and choose
   the lowest resulting class error that fits the remaining development quota.
2. If additions cannot hit the quota, allow a closer whole-component overshoot;
   never split a component. No cardinality-optimality claim is made.
3. Search all available one-component-for-one-component exchanges, accepting
   only exchanges that do not worsen size distance and improve class error
   (or improve size while preserving class error). Choose the best deterministic
   exchange, repeat up to 200 accepted swaps, and record whether the search
   reached a local optimum or its cap.

This is deterministic multilabel optimisation using source counts only, with
no model-performance input. It is not globally optimal; the local optimum refers
only to that exchange neighbourhood. Source feature combinations or indivisible
groups can prevent exact ratios. Inspect the resulting deviations before approval.

Output order is the original source manifest order restricted to each partition.
The same source, labels, confirmed edges, implementation and seed reproduce the
same membership/manifest bytes. Generation timestamps may differ.

## Generation on DICC only

Configure `configs/local/paths.yaml` for the existing DICC checkout and dataset.
Then, from the checkout root:

```bash
PYTHONPATH=src python -B -m trustpcb.rq2_data_partition --dicc
```

This command does not import YOLO/PyTorch or perform training, inference,
calibration, transformations, scoring or evaluation. It reads source labels,
generates membership and freezes these new files:

- `data/splits/rq2_train_development_split/detector_train_images.txt`
- `data/splits/rq2_train_development_split/development_calibration_images.txt`
- `data/splits/rq2_train_development_split/verification_report.json`

The report includes exact counts, per-class image and annotation counts for the
source and both partitions, integrity assertions, group counts, source/group
input hashes, a source-label inventory digest, generator hash, seed, Git commit,
UTC timestamp, and both output manifest hashes. Paths in committed artifacts
are portable. The original manifests, dataset and RQ1 outputs are not modified.
The rejected `data/splits/rq2_rejected_train_development_split/` files are preserved byte-for-byte. When
present, their membership is validated and scored with the same current source
counts and error definition; the revised report includes the rejected per-class
table and error reduction. It does not read test labels or performance results.
An existing **v2** directory is an error; there is no overwrite option.
If output writing is interrupted, inspect the partial directory explicitly.

The `class_balance` table reports every class's source/dev image and annotation
counts, development percentages of source, and absolute deviations from 10% in
percentage points. The summary reports MAE, RMSE, MSE and maximum absolute
deviation across nonzero class features. The optimisation block records initial
and final MSE, accepted swaps, error history and convergence/cap status. These
are split-balance statistics, not detector metrics. Counts/hashes and all prior
integrity checks remain in the report.

## Local verification and current availability

```bash
PYTHONPATH=src python -B -m unittest discover -s tests -v
```

Tests use synthetic counts/labels, including a synthetic 8,207-image manifest;
synthetic counts are not real RQ2 results.

At implementation time this checkout had no configured dataset root or complete
per-image class-count inventory. Consequently the real partition counts,
per-class statistics and output hashes cannot yet be reported. Run the DICC
generation command and return its three artifacts for review. Do not begin
detector training until the membership has been reviewed/frozen and training
is explicitly approved.
