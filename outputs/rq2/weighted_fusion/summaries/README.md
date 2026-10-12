# Weighted fusion · development selection

Frozen development-only selection of confidence/class/localisation fusion weights.

- [Tables](../tables/): unchanged reporting copies of saved CSVs.
- [Figures](../figures/): newly rendered visualizations, not original DICC figures.
- JSON files beside this summary are unchanged copies of saved summaries/provenance.
- [Original evidence](../../../../runs/rq2/weighted_fusion/) remains authoritative.
- [Scientific implementation](../../../../src/trustpcb/rq2/weighted_fusion.py)
- [Tests](../../../../tests/rq2/test_rq2_weighted_fusion.py)
- [Protocol](../../../../docs/rq2/rq2_weighted_fusion.md)

Post-selection development metrics are descriptive, not independent generalisation evidence. The 5,151-row candidate grid remains at its original path, linked and hashed rather than duplicated. No weights are selected again.

[Original candidate grid](../../../../runs/rq2/weighted_fusion/candidate_weights.csv) is referenced in the manifest.

Observed AUROC/AUPRC and interval endpoints come directly from saved files. Missing
values are not zeros. Original `trustpcb_risk` denotes the original equal-weight
composite; it must not be confused with `weighted_trustpcb_risk`. Paired differences
retain the saved method-A-minus-method-B direction. No significance claims are added.
Copied historical provenance describes the original execution, not this reporting run.
See the [reporting manifest](../../manifest.json) for paths, hashes and derivations.
