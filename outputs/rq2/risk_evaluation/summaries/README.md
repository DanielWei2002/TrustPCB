# Development risk evaluation · IoU ≥ 0.50

Development risk ablations and incorrect-prediction ranking.

- [Tables](../tables/): unchanged reporting copies of saved CSVs.
- [Figures](../figures/): newly rendered visualizations, not original DICC figures.
- JSON files beside this summary are unchanged copies of saved summaries/provenance.
- [Original evidence](../../../../runs/rq2/risk_evaluation/) remains authoritative.
- [Scientific implementation](../../../../src/trustpcb/rq2/risk_evaluation.py)
- [Tests](../../../../tests/rq2/test_rq2_risk_evaluation.py)
- [Protocol](../../../../docs/rq2/rq2_risk_evaluation.md)

Development diagnostics, not independent test evidence. bootstrap_metrics.csv is absent locally; saved interval summaries remain available.

Observed AUROC/AUPRC and interval endpoints come directly from saved files. Missing
values are not zeros. Original `trustpcb_risk` denotes the original equal-weight
composite; it must not be confused with `weighted_trustpcb_risk`. Paired differences
retain the saved method-A-minus-method-B direction. No significance claims are added.
Copied historical provenance describes the original execution, not this reporting run.
See the [reporting manifest](../../manifest.json) for paths, hashes and derivations.
