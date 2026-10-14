# Development sensitivity · IoU ≥ 0.75

Secondary correctness-definition sensitivity analysis on development evidence.

- [Tables](../tables/): unchanged reporting copies of saved CSVs.
- [Figures](../figures/): newly rendered visualizations, not original DICC figures.
- JSON files beside this summary are unchanged copies of saved summaries/provenance.
- [Original evidence](../../../../runs/rq2/risk_sensitivity_iou75/) remains authoritative.
- [Scientific implementation](../../../../src/trustpcb/rq2/risk_sensitivity.py)
- [Tests](../../../../tests/rq2/test_rq2_risk_sensitivity.py)
- [Protocol](../../../../docs/rq2/rq2_risk_sensitivity.md)

Secondary analysis; it does not replace the primary IoU ≥ 0.50 results. bootstrap_metrics.csv is absent locally.

Observed AUROC/AUPRC and interval endpoints come directly from saved files. Missing
values are not zeros. Original `trustpcb_risk` denotes the original equal-weight
composite; it must not be confused with `weighted_trustpcb_risk`. Paired differences
retain the saved method-A-minus-method-B direction. No significance claims are added.
Copied historical provenance describes the original execution, not this reporting run.
See the [reporting manifest](../../manifest.json) for paths, hashes and derivations.
