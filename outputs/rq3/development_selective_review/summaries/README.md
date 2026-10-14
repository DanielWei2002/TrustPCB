# Original development selective review

Scientific experiment identity: `rq3_development_selective_review`.

Original 821-crop development protocol evidence; not independent test generalisation evidence. The original CSV schema remains intact. Supplementary automatic-error metrics and paired differences live in reporting_extension, not in this parent.

- [Tables](../tables/): small, byte-identical saved CSV copies.
- [Figures](../figures/): newly rendered visualizations from saved evidence.
- JSON files beside this guide retain historical execution metadata, not reporting-run metadata.
- [Original evidence](../../../../runs/rq3/development_selective_review/)
- [Canonical scientific module](../../../../src/trustpcb/rq3/selective_review.py)
- [Tests](../../../../tests/rq3/test_rq3_selective_review.py)
- [Scientific protocol](../../../../docs/rq3/rq3_selective_review.md)
- [Reporting guide and gaps](../../README.md) · [manifest](../../manifest.json)

Intervals are saved intervals, not newly estimated uncertainty. Prediction-level
AURC and crop-level referral are different outcomes; an AURC advantage is not a
claim of improved crop referral. Paired difference signs and achieved workloads
are preserved without claims of universal improvement, factory readiness or real-time operation.

Full [risk–coverage boundaries](../../../../runs/rq3/development_selective_review/prediction_risk_coverage.csv) and [crop risks](../../../../runs/rq3/development_selective_review/crop_risks.csv) remain at their original paths, hashed in the manifest instead of duplicated.

The historical [reporting extension](../reporting_extension/summaries/README.md) remains separate.
