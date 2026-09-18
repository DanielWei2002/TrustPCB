# RQ1 Results

Portable records for the five matched-seed RQ1 experiments comparing the supplied and similarity-aware DsPCBSD+ splits.

`per_seed_metrics.csv` contains the selected best-epoch metrics for all ten runs.

`aggregate_metrics.csv` contains the mean and sample standard deviation across the five seeds for each split.

`run_records/` contains portable copies of each run's `args.yaml`, complete 100-epoch `results.csv`, completion record, and provenance record.

`pretrained_model.json` records the pretrained YOLOv8n model identity used by the experiment.

Raw checkpoints, generated training images, plots, logs, and other large Ultralytics artifacts remain under `runs/rq1/` and are intentionally excluded from Git.

Machine-specific absolute project paths in this curated result package are converted to repository-relative paths. Metric values and scientific experiment records are otherwise unchanged.
