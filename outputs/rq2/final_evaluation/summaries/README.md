# Original final evaluation

Frozen Beta calibration and original equal-weight/ablation ranking on final-test evidence.

- [Tables](../tables/): unchanged reporting copies of saved CSVs.
- [Figures](../figures/): newly rendered visualizations, not original DICC figures.
- JSON files beside this summary are unchanged copies of saved summaries/provenance.
- [Original evidence](../../../../runs/rq2/final_evaluation/) remains authoritative.
- [Scientific implementation](../../../../src/trustpcb/rq2/final_evaluation.py)
- [Tests](../../../../tests/rq2/test_rq2_final_evaluation.py)
- [Protocol](../../../../docs/rq2/rq2_final_evaluation.md)

Held out from RQ2 training/development, but previously used for RQ1 validation. Detailed prediction, stability, transform-match and inference-audit files, and both bootstrap_metrics.csv files, are absent locally. Original provenance is copied unchanged despite those missing dependencies.

Observed AUROC/AUPRC and interval endpoints come directly from saved files. Missing
values are not zeros. Original `trustpcb_risk` denotes the original equal-weight
composite; it must not be confused with `weighted_trustpcb_risk`. Paired differences
retain the saved method-A-minus-method-B direction. No significance claims are added.
Copied historical provenance describes the original execution, not this reporting run.
See the [reporting manifest](../../manifest.json) for paths, hashes and derivations.
