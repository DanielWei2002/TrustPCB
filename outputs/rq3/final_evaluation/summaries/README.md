# Frozen-threshold final evaluation

Scientific experiment identity: `rq3_final_evaluation`.

Fixed development thresholds applied to 2,052 test crops. This population was held out from RQ2 training/development but previously used for RQ1 validation. Plots retain achieved test workloads, not nominal budget targets; no test quantiles or threshold adjustment is performed.

- [Tables](../tables/): small, byte-identical saved CSV copies.
- [Figures](../figures/): newly rendered visualizations from saved evidence.
- JSON files beside this guide retain historical execution metadata, not reporting-run metadata.
- [Original evidence](../../../../runs/rq3/final_evaluation/)
- [Canonical scientific module](../../../../src/trustpcb/rq3/final_evaluation.py)
- [Tests](../../../../tests/rq3/test_rq3_final_evaluation.py)
- [Scientific protocol](../../../../docs/rq3/rq3_final_evaluation.md)
- [Reporting guide and gaps](../../README.md) · [manifest](../../manifest.json)

Intervals are saved intervals, not newly estimated uncertainty. Prediction-level
AURC and crop-level referral are different outcomes; an AURC advantage is not a
claim of improved crop referral. Paired difference signs and achieved workloads
are preserved without claims of universal improvement, factory readiness or real-time operation.

Full [risk–coverage boundaries](../../../../runs/rq3/final_evaluation/prediction_risk_coverage.csv) and [crop risks](../../../../runs/rq3/final_evaluation/crop_risks.csv) remain at their original paths, hashed in the manifest instead of duplicated.
