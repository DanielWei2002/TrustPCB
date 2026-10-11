# RQ3 computational efficiency

The canonical module is `trustpcb.rq3.computational_efficiency`.
`trustpcb.rq3_computational_efficiency` remains a compatibility import/CLI shim;
both import paths resolve to the same live module. RQ2 dependencies use canonical
package paths, retaining the directly bound `selected_identity` validator.
The definition-time clock, adapter and image-count defaults are unchanged.
Completed benchmark evidence and the historical experiment identifier
`rq3_computational_efficiency` remain unchanged. This source relocation does not
require rerunning the benchmark.

Benchmark only the pinned 821 development/calibration crops in manifest order,
using the frozen epoch-72 YOLOv8n selected checkpoint, existing Beta parameters and
89/1/10 weighted reliability score. This is a DICC GPU workload; never run locally.
No test manifest, test image, test prediction or test result is read. Existing
RQ1/RQ2/selective-review evidence is read-only. Missing artifacts fail rather than
being regenerated. No fitting, matching-method changes or threshold selection occurs.

## Frozen pipelines

Baseline: original crop YOLO inference, including file loading/decode, detector
preprocessing, inference, NMS and detection extraction to CPU. One detector pass/crop.

TrustPCB: the same original inference plus eleven frozen transformations:
brightness 0.90/1.10, contrast 0.90/1.10, blur sigma 0.6, rotation -2/+2 degrees,
translation x/y each -0.02/+0.02 of the relevant dimension. Every transformed view
starts from the original decoded BGR image. Reuse frozen EXIF-aware RGB decoding,
quantisation, geometric mapping, class-agnostic matching and evaluable-family
aggregation. Apply the existing Beta model to original references and compute
`0.89*(1-C)+0.01*(1-Sc)+0.10*(1-Sl)` using the original integer-weight arithmetic.
Twelve detector passes/crop, including crops without retained original predictions.

Both pipelines use unchanged `trustpcb.rq2.confidence_distribution.SETTINGS`, including
batch=1, device=0, imgsz=640, conf=0.001, iou=0.7, max_det=300, rect=True,
augment=False and class-aware NMS. Original references retain confidence >=0.01;
transformed detections use the frozen extraction floor. The full pipeline loads
the original through the same detector path as baseline and additionally decodes
it through the frozen Pillow/EXIF path for transformation generation. Both decoding
costs are included in full timing, as are matching, calibration and risk calculation.
Prediction settings/dimensions are checked during processing.

## Warm-up and timing

Load one detector before warm-up, validate identities/settings and record input
hashes outside timing. Warm baseline on the first 20 manifest crops, synchronize
CUDA, then warm full TrustPCB on those same 20 crops and synchronize CUDA again.
This also excludes predictor initialization and internal detector warm-up from
reported time. No warm-up inference is counted in timed pass totals.

Run five repetitions. Each repetition times baseline over all 821 crops followed
by TrustPCB over all 821 crops, always in the same manifest order. Each timed region:
CUDA synchronize(0), start `perf_counter`, process the full population, CUDA
synchronize(0), stop `perf_counter`. This includes transfer/completion waits and
prevents asynchronous under-reporting. No decoded-image or transformed-view cache
is used by the adapter. OS filesystem caching remains a normal warm-run influence.

Exclude model loading, environment startup, provenance/input hashing and all output
file writing. Include crop loading, preprocessing, transformation generation,
detector inference/postprocessing, CPU matching/aggregation, calibration and risk
computation as appropriate. Synchronization is at population repetition boundaries,
not artificially added to every crop.

Per repetition: baseline 821 passes; TrustPCB 9,852 passes. Across five timed
repetitions: baseline 4,105 passes; TrustPCB 49,260 passes. Warm-up adds 20 baseline
and 240 full-pipeline passes, outside reported timing. A pass means one explicit
batch-1 prediction call; internal library initialization is warmed up separately.

Report raw total seconds for each repetition and pipeline, mean, sample standard
deviation (ddof=1), mean milliseconds/crop = mean seconds*1000/821, throughput =
821/mean seconds, and detector passes/crop. Relative overhead = TrustPCB mean /
baseline mean; percentage increase = (relative overhead-1)*100. No confidence
interval, performance optimisation or scientific method selection is introduced.

## DICC execution and outputs

After review and committing execution inputs, configure local paths and run from
the repository root on a DICC GPU allocation:

```bash
PYTHONPATH=src python -B -m trustpcb.rq3.computational_efficiency --dicc
```

Requires frozen Ultralytics 8.4.117 and the existing CUDA/PyTorch environment.
Windows is rejected before ML imports; unavailable CUDA causes failure, never a
CPU fallback. Checkpoint hash, selected-weight evidence commit and completed
development Beta/stability provenance are validated before execution. All 821
development file hashes and source/artifact identities are rechecked afterwards.

New output only: `runs/rq3/computational_efficiency/`:

- `repetition_timings.csv`: ten rows with pipeline, repetition, crops, seconds,
  detector passes, milliseconds/crop and throughput
- `summary.json`: per-pipeline repetition totals/statistics, overhead, frozen protocol
- `provenance.json`: checkpoint/Beta/weight identities, manifest/development artifact
  hashes, development order and first-20 warm-up inventory, image hashes, transforms
  and rules, requested/effective settings, Git state, package/Python/CUDA/cuDNN
  versions, GPU name/device, timestamps, status and output hashes

An existing partial/completed directory blocks execution; archive explicitly before
a reviewed retry. Failure after output reservation leaves incomplete provenance.
No completed research output is overwritten.

Tests use fake clocks, callbacks, mocked ML modules and tiny synthetic pixel arrays
only. They verify warm-up exclusion, synchronization/order, five repetitions,
pass totals, frozen transforms/settings, empty crops, score reuse, summary formulas,
output isolation and execution protection. No real benchmark has run locally.
