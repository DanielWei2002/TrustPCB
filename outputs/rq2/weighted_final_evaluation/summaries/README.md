# Frozen weighted final evaluation

Previously selected weights applied to final-test signals; saved primary/secondary ranking and paired differences.

- [Tables](../tables/): unchanged reporting copies of saved CSVs.
- [Figures](../figures/): newly rendered visualizations, not original DICC figures.
- JSON files beside this summary are unchanged copies of saved summaries/provenance.
- [Original evidence](../../../../runs/rq2/weighted_final_evaluation/) remains authoritative.
- [Scientific implementation](../../../../src/trustpcb/rq2/weighted_final_evaluation.py)
- [Tests](../../../../tests/rq2/test_rq2_weighted_final_evaluation.py)
- [Protocol](../../../../docs/rq2/rq2_weighted_final_evaluation.md)

Held out from RQ2 training/development, previously used for RQ1 validation. No test-driven weight revision. Summaries exist locally; the complete upstream prediction/calibration chain does not.

Observed AUROC/AUPRC and interval endpoints come directly from saved files. Missing
values are not zeros. Original `trustpcb_risk` denotes the original equal-weight
composite; it must not be confused with `weighted_trustpcb_risk`. Paired differences
retain the saved method-A-minus-method-B direction. No significance claims are added.
Copied historical provenance describes the original execution, not this reporting run.
See the [reporting manifest](../../manifest.json) for paths, hashes and derivations.
