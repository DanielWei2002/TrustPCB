# Matched repeated training

Four CSV tables and the pretrained-model identity are copied without changes from
[results/rq1](../../../../results/rq1/). The new figure uses saved aggregate means
and sample SD directly. It does not show confidence intervals or significance.

Ten original portable run records remain in [run_records](../../../../results/rq1/run_records/).
The canonical implementation remains [workflow.py](../../../../src/trustpcb/rq1/workflow.py).
The producer of paired_differences.csv and paired_summary.csv remains unresolved;
copying them does not establish their computational provenance. No paired statistics
were recalculated. Raw runs/rq1 artifacts, checkpoints and their original training
figures are external/absent locally. The new aggregate figure is not an original
DICC training plot. Preserve string seed label 02092026 when loading CSVs.
