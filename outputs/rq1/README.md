# RQ1 research reporting

RQ1 investigated detector performance under supplied versus similarity-aware splits,
first through dataset auditing/preliminary baselines, then five matched training seeds.

| Experiment | Tables | Figures | Summary |
|---|---|---|---|
| Dataset audit and split preparation | [tables](dataset_audit/tables/) | [figures](dataset_audit/figures/) | [summary](dataset_audit/summaries/README.md) |
| Preliminary baseline v1 | [tables](preliminary_baseline/tables/) | [figures](preliminary_baseline/figures/) | [summary](preliminary_baseline/summaries/README.md) |
| Matched training | [tables](matched_training/tables/) | [figures](matched_training/figures/) | [summary](matched_training/summaries/README.md) |

These are reporting group names, not replacements for frozen experiment identifiers.
Original evidence remains in outputs/tables, runs/preliminary_baseline/v1 and
results/rq1. No existing evidence is moved or rewritten. Raw data, checkpoints and
raw matched-training plots are external; the paired-table producer is unresolved.

Code: [canonical workflow](../../src/trustpcb/rq1/workflow.py),
[report builder](../../src/trustpcb/rq1/reporting.py).
Historical helper extractions: [dataset audit](../../src/trustpcb/rq1/dataset_audit.py)
and [similarity split](../../src/trustpcb/rq1/similarity_split.py).
They are reconstructed code, not the originally executed programs.
See the [cell/source map and limitations](../../docs/rq1/notebook_extraction.md).
Tests: [tests/rq1](../../tests/rq1/).

[manifest.json](manifest.json) records each artifact's source paths, byte SHA-256,
output SHA-256 and recipe. It distinguishes byte copies, exported notebook outputs,
newly rendered figures and authored summaries. Its source commit identifies the
pre-batch evidence snapshot, not an execution commit for the new reporting code.
The generator hash records the new code; manifest.json itself is the inventory.
CSV/JSON copies preserve bytes, schemas and historical metadata, including paths.
Figure rendering does not recompute metrics, intervals, splits or scientific results.

From the repository root with PYTHONPATH=src and existing NumPy/Matplotlib/PyYAML:
`python -B -m trustpcb.rq1.reporting --verify-only` checks recorded bytes.
`python -B -m trustpcb.rq1.reporting` builds/rebuilds these lightweight reports.
Existing edited/unlisted files or changed sources block rebuilding. No raw data or
ML environment is needed. Do not use the research extraction/training commands to
refresh this reporting area.

DsPCBSD+ derived visual evidence retains the dataset attribution in the
[repository README](../../README.md) (Lv et al., 2024; CC BY 4.0).
