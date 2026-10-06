# TrustPCB

TrustPCB is a research project for reliability-aware PCB defect detection using deep learning and automated optical inspection.

## Project Aim

The project investigates a model-agnostic reliability framework for lightweight PCB defect detection by combining:

- calibrated confidence
- class consistency
- localisation stability
- composite prediction-risk scoring
- selective human-review referral

## Dataset

This project uses the **DsPCBSD+ PCB surface-defect dataset** introduced by Lv et al. (2024).

**Reference**  
Lv, S., Ouyang, B., Deng, Z., et al. (2024). *A dataset for deep learning based detection of printed circuit board surface defect*. Scientific Data, 11, 811.  
DOI: https://doi.org/10.1038/s41597-024-03656-8  
Figshare dataset: https://doi.org/10.6084/m9.figshare.24970329

The DsPCBSD+ dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** licence.

Raw and processed dataset files are not stored in this GitHub repository. Some generated training visualisations, validation examples and model-output figures in this repository are derived from DsPCBSD+ for research and evaluation purposes and should be interpreted with attribution to the original dataset authors.

## Completed research

RQ1, RQ2 and RQ3 experimental work is complete:

- **RQ1:** matched repeated-training comparison of supplied and similarity-aware splits.
- **RQ2:** detector reliability through confidence calibration, transformation stability and frozen weighted prediction-risk scoring.
- **RQ3:** selective crop review, computational-efficiency benchmarking and held-out referral evaluation.

The completed pre-reorganization state is preserved by the Git tag
`pre-reorganization-rq1-rq2-rq3-complete`.

## Repository structure

- `src/` contains the reproducible Python implementation.
- `tests/` contains automated verification, including CPU-safe synthetic tests.
- `notebooks/` contains the original exploratory, dataset-audit and preliminary research notebooks. They retain their existing paths and can include expensive execution cells.
- `configs/` and `data/splits/` contain scientific configuration and frozen membership manifests. Machine-specific runtime configuration belongs in ignored `configs/local/`.
- `outputs/` contains early dataset-audit and preliminary output evidence.
- `results/rq1/` contains curated portable RQ1 evidence.
- `runs/` contains execution records and completed RQ2/RQ3 evidence.
- `docs/` contains experiment protocols, documentation grouped by research question and historical migration records.

Completed evidence paths are intentionally preserved because provenance and
downstream code refer to them. See the [artifact inventory](docs/artifact_inventory.md)
for local evidence and known external dependencies. The local checkout does not
contain the raw dataset, model checkpoints or every upstream prediction/calibration
artifact referenced by completed provenance. Missing artifacts must be located
and verified rather than regenerated for housekeeping.

## Protocol documentation

- [RQ1 matched repeated training](docs/rq1/rq1_workflow.md)
- [RQ2 data partition](docs/rq2/rq2_data_partition.md)
- [RQ2 detector training](docs/rq2/rq2_detector_training.md)
- [RQ2 confidence distribution](docs/rq2/rq2_confidence_distribution.md)
- [RQ2 calibration analysis](docs/rq2/rq2_calibration_analysis.md)
- [RQ2 calibrator comparison](docs/rq2/rq2_calibrator_comparison.md)
- [RQ2 final Beta calibrator](docs/rq2/rq2_final_calibrator.md)
- [RQ2 transformation stability](docs/rq2/rq2_transformation_stability.md)
- [RQ2 development risk evaluation](docs/rq2/rq2_risk_evaluation.md)
- [RQ2 sensitivity analysis](docs/rq2/rq2_risk_sensitivity.md)
- [RQ2 weighted fusion](docs/rq2/rq2_weighted_fusion.md)
- [RQ2 original final evaluation](docs/rq2/rq2_final_evaluation.md)
- [RQ2 weighted final evaluation](docs/rq2/rq2_weighted_final_evaluation.md)
- [RQ3 development selective review](docs/rq3/rq3_selective_review.md)
- [RQ3 computational efficiency](docs/rq3/rq3_computational_efficiency.md)
- [RQ3 final evaluation](docs/rq3/rq3_final_evaluation.md)

Name files, folders and experiment identifiers by scientific or computational
purpose. See [research naming](docs/research_naming.md) and the
[historical naming-refactor record](docs/history/naming_refactor.md).
Historical protocol and migration documents retain their original context;
their preparation-time statements are not claims that all artifacts exist locally today.

## Execution environment

Research execution uses Python 3.11, PyTorch, CUDA and Ultralytics YOLO on DICC.
Computationally intensive training, inference, evaluation and benchmarking belong
on DICC. Local work is limited to implementation, inspection and CPU-safe tests
with synthetic or deliberately tiny inputs. Do not run the original notebooks
wholesale locally.

Current module entry points remain unchanged and use `PYTHONPATH=src`; package
restructuring is deferred. Dataset roots are configured using
[the local-path example](configs/paths.example.yaml).
