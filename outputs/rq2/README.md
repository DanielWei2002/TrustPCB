# RQ2 research reporting

RQ2 investigates confidence calibration, transformation consistency/localisation
stability and frozen prediction-risk fusion for PCB defect detection.

| Experiment | Purpose | Reporting materials |
|---|---|---|
| [risk_evaluation](risk_evaluation/summaries/README.md) | Development risk ablations and incorrect-prediction ranking. | [tables](risk_evaluation/tables/) · [figures](risk_evaluation/figures/) · [summaries](risk_evaluation/summaries/) |
| [risk_sensitivity_iou75](risk_sensitivity_iou75/summaries/README.md) | Secondary correctness-definition sensitivity analysis on development evidence. | [tables](risk_sensitivity_iou75/tables/) · [figures](risk_sensitivity_iou75/figures/) · [summaries](risk_sensitivity_iou75/summaries/) |
| [weighted_fusion](weighted_fusion/summaries/README.md) | Frozen development-only selection of confidence/class/localisation fusion weights. | [tables](weighted_fusion/tables/) · [figures](weighted_fusion/figures/) · [summaries](weighted_fusion/summaries/) |
| [final_evaluation](final_evaluation/summaries/README.md) | Frozen Beta calibration and original equal-weight/ablation ranking on final-test evidence. | [tables](final_evaluation/tables/) · [figures](final_evaluation/figures/) · [summaries](final_evaluation/summaries/) |
| [weighted_final_evaluation](weighted_final_evaluation/summaries/README.md) | Previously selected weights applied to final-test signals; saved primary/secondary ranking and paired differences. | [tables](weighted_final_evaluation/tables/) · [figures](weighted_final_evaluation/figures/) · [summaries](weighted_final_evaluation/summaries/) |

Each experiment summary links its canonical Python module, relocated tests and
protocol. The original 13 modules and compatibility shims remain unchanged.
[Data-partition evidence](data_partition/summaries/README.md) is also available.

## Upstream availability

| Stage / canonical Python | Tests | Availability in this checkout |
|---|---|---|
| [data_partition](../../src/trustpcb/rq2/data_partition.py) | [test](../../tests/rq2/test_rq2_data_partition.py) | Local frozen manifests and verification report in data/splits/rq2_train_development_split; dataset template in configs/datasets/dspcbsd_rq2_train_dev.yaml. |
| [detector_training](../../src/trustpcb/rq2/detector_training.py) | [test](../../tests/rq2/test_rq2_detector_training.py) | Run directory/checkpoint absent locally; locate completed DICC artifacts and verify their recorded identities. |
| [confidence_distribution](../../src/trustpcb/rq2/confidence_distribution.py) | [test](../../tests/rq2/test_rq2_confidence_distribution.py) | Raw development predictions, inventory and confidence histogram absent locally. |
| [calibration_analysis](../../src/trustpcb/rq2/calibration_analysis.py) | [test](../../tests/rq2/test_rq2_calibration_analysis.py) | Development correctness/calibration tables and raw reliability/histogram figures absent locally. |
| [calibrator_comparison](../../src/trustpcb/rq2/calibrator_comparison.py) | [test](../../tests/rq2/test_rq2_calibrator_comparison.py) | Development comparison tables and reliability_comparison.png absent locally. |
| [final_calibrator](../../src/trustpcb/rq2/final_calibrator.py) | [test](../../tests/rq2/test_rq2_final_calibrator.py) | Development-fitted Beta mapping and its upstream fitting artifacts absent locally; downstream summaries are not substitutes. |
| [transformation_stability](../../src/trustpcb/rq2/transformation_stability.py) | [test](../../tests/rq2/test_rq2_transformation_stability.py) | Development stability and transformed-prediction artifacts absent locally. |

Raw images/labels and model checkpoints are external. Missing paths are documented
dependencies, not proof that artifacts were never generated or that their DICC
locations are verified. Retrieve and verify existing evidence rather than rerunning.
Detailed missing final-evaluation files are listed in [manifest.json](manifest.json)
and [the artifact inventory](../../docs/artifact_inventory.md).

## Evidence and presentation

- Original evidence stays in `runs/rq2/` and `data/splits/`; no writer destinations,
  schemas, memberships, identities or frozen weights change.
- CSV/JSON files here are **reporting copies**, preserving original bytes and metadata.
- All figures here are **newly rendered visualizations** from saved numerical evidence;
  no original RQ2 figure files were available locally. These are not recreated ROC/PR
  curves: aggregate AUROC/AUPRC cannot supply those missing curves.
- Development selection, original final evaluation and weighted final evaluation
  remain distinct. Final-test images were held out from RQ2 training/development,
  but previously used for RQ1 validation. Saved intervals are displayed, not recalculated.
- The candidate grid is linked at its original path instead of duplicated. No large
  prediction tables, dataset images or checkpoints are copied.
- This manifest verifies local reporting provenance, not the complete scientific
  dependency chain. Historical provenance is never rewritten to conceal local gaps.

## Reproduction and verification

New presentation code: [reporting.py](../../src/trustpcb/rq2/reporting.py).
No scientific workflow is imported or executed. Matplotlib is loaded only to render.
With `PYTHONPATH=src`, use `python -B -m trustpcb.rq2.reporting --verify-only` to
verify all recorded sources, outputs and referenced originals. Use
`python -B -m trustpcb.rq2.reporting` to build/rebuild this small reporting area.
Existing edited/unlisted reports, changed source hashes or newly available upstream
artifacts block rebuilding for inventory review. Hashes describe exact local bytes,
including line endings; they do not replace historical LF-canonical scientific hashes.

The manifest records source paths/SHA-256, output SHA-256, recipes, generator identity
and rendering versions. Its base commit is the pre-batch reference, not the original
execution commit of each experiment. `manifest.json` is the inventory itself.

[Reporting protocol and test commands](../../docs/rq2/reporting.md) ·
[RQ1 reporting](../rq1/README.md)
