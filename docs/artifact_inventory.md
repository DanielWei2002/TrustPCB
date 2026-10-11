# Artifact inventory

## Evidence-location convention

RQ1, RQ2 and RQ3 experimental work is complete. This inventory describes the
local checkout preserved at commit `94da758e57dae0cce6f5ccda922c27733fd7180e`,
tagged `pre-reorganization-rq1-rq2-rq3-complete`. It does not relocate evidence.

- `outputs/`: early dataset-audit and preliminary output evidence.
- `results/rq1/`: curated portable RQ1 evidence.
- `runs/`: execution records and completed RQ2/RQ3 evidence.

These differing conventions reflect the project's evolution. Completed paths
remain unchanged because code and provenance reference them. Recorded scientific
values, memberships, hashes and historical provenance must remain immutable.

## Evidence present locally

Paths below are relative to the repository root.

| Path | Purpose and contents | Classification | Immutable? |
|---|---|---|---|
| `outputs/tables/` | Dataset-audit summary, similarity candidates, alignment/manual review, 19 confirmed similarity pairs and preliminary baseline comparison | Frozen early scientific input/evidence | Yes |
| `runs/preliminary_baseline/v1/smoke_test/` | Original two-epoch real-data smoke-run arguments, epoch log and saved figures | Historical preliminary evidence | Yes |
| `runs/preliminary_baseline/v1/supplied/` | Supplied-split preliminary run arguments, 100-epoch log and saved figures | Historical preliminary evidence | Yes |
| `runs/preliminary_baseline/v1/similarity_aware/` | Similarity-aware preliminary run arguments, 100-epoch log and saved figures | Historical preliminary evidence | Yes |
| `results/rq1/` | Per-seed metrics, aggregate metrics, paired differences/summaries and pretrained-model identity | Canonical curated frozen RQ1 evidence | Yes |
| `results/rq1/run_records/` | Ten matched-run records across two splits and five seeds; arguments, 100-epoch logs, provenance and completion receipts | Canonical curated frozen RQ1 execution evidence | Yes |
| `runs/rq2/risk_evaluation/` | Development risk ablations, ranking metrics, bootstrap summaries and signal/class diagnostics | Frozen development evidence | Yes |
| `runs/rq2/risk_sensitivity_iou75/` | Secondary correctness-definition sensitivity analysis | Frozen secondary evidence | Yes |
| `runs/rq2/weighted_fusion/` | 5,151 weight candidates, development fold assignments, selected weights and primary/secondary metrics | Frozen development-only selection evidence | Yes |
| `runs/rq2/final_evaluation/` | Original final-test calibration/ranking summaries, transformation specification and provenance; some detailed outputs are external | Frozen original final-test evidence | Yes |
| `runs/rq2/weighted_final_evaluation/` | Frozen weighted-method metrics and paired AUROC/AUPRC differences for primary and secondary correctness definitions | Frozen weighted final-test evidence | Yes |
| `runs/rq3/development_selective_review/` | Development thresholds, crop risks/referral metrics, full prediction risk-coverage curves, AURC and bootstrap evidence | Frozen development protocol/evidence | Yes |
| `runs/rq3/development_selective_review/reporting_extension/` | Remaining automatic-error metrics and paired intervals added without rewriting the original development outputs | Frozen supplementary reporting evidence | Yes |
| `runs/rq3/computational_efficiency/` | Raw five-repetition timings for both pipelines, summary and provenance | Frozen DICC benchmark evidence | Yes |
| `runs/rq3/final_evaluation/` | Fixed-threshold test referral metrics, crop risks, prediction risk-coverage curves, AURC and paired uncertainty | Frozen final-test evidence | Yes |

`data/splits/` and the active files in `configs/datasets/` are also frozen scientific
inputs. The RQ2 detector-training/development manifests remain under
`data/splits/rq2_train_development_split/`. The three original notebooks remain
under `notebooks/`; they preserve early audit, split-generation and preliminary
training context and are not reporting-only executables.

The duplicate RQ1 provenance/completion receipt contents serve distinct record
roles and are retained. The RQ3 reporting extension supplements its parent;
it does not replace or rewrite the frozen thresholds or original evidence.

## Known external / DICC-only artifacts

The audit established the following absent local artifacts or categories.
Recorded references establish dependencies, not local availability. Their
external locations must be confirmed when retrieving them.

- Raw dataset images and annotation files, configured through local dataset roots.
- Pretrained and trained checkpoint files, including the selected RQ2 detector;
  checkpoints are excluded from Git. Completed provenance records their identities.
- Raw RQ1 execution/checkpoint artifacts under `runs/rq1/`; the curated
  `results/rq1/README.md` explains the separation.
- Upstream RQ2 detector-training, confidence-distribution, calibration-analysis,
  calibrator-comparison, final-calibrator and development transformation-stability
  artifacts. Downstream code/provenance references these workflows, but their
  directories are not present in this local checkout.
- Under `runs/rq2/final_evaluation/`, these provenance-listed outputs are absent:
  `inference_audit.json`, `original_predictions.csv`, `labelled_predictions.csv`,
  `prediction_stability_scores.csv`, `prediction_transform_matches.csv`,
  `transformed_predictions.csv`, `primary_iou50/bootstrap_metrics.csv` and
  `sensitivity_iou75/bootstrap_metrics.csv`.
- `bootstrap_metrics.csv` under each of `runs/rq2/risk_evaluation/` and
  `runs/rq2/risk_sensitivity_iou75/`, as listed in their completed provenance.

Missing artifacts must be **located and verified against their recorded identities**,
not regenerated for housekeeping. Do not rerun experiments to make the local tree
appear complete, and do not rewrite provenance to conceal missing local files.

## Preservation and verification

The preservation tag protects the pre-reorganization state. Documentation moves
do not change artifact paths or scientific identities. Windows checkouts can have
CRLF line endings while scientific manifests use recorded LF-canonical hashes;
verification must follow the existing workflow's hash convention.

This inventory is a location guide, not a new experimental result or a claim that
the complete external artifact chain has been independently verified locally.
