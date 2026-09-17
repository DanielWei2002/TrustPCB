# Final RQ1: five matched seeds, two split conditions

RQ1 asks how detector performance differs between the supplied DsPCBSD+ split
and the similarity-aware split. This workflow creates ten **new** controlled
runs. Historical preliminary runs are never targeted or reused.

## Frozen execution order

| Order | Seed label | Integer value | Split |
| --- | --- | --- | --- |
| 1 | `24209199` | 24209199 | supplied |
| 2 | `24209199` | 24209199 | similarity_aware |
| 3 | `20260902` | 20260902 | supplied |
| 4 | `20260902` | 20260902 | similarity_aware |
| 5 | `02092026` | 2092026 | supplied |
| 6 | `02092026` | 2092026 | similarity_aware |
| 7 | `12345678` | 12345678 | supplied |
| 8 | `12345678` | 12345678 | similarity_aware |
| 9 | `20202027` | 20202027 | supplied |
| 10 | `20202027` | 20202027 | similarity_aware |

Each output directory is `runs/rq1/<split>/seed_<label>/`. Labels are strings:
`seed_02092026` stays exactly that, while Python/Ultralytics receives 2092026.

## Configuration and architecture

`src/trustpcb/rq1.py` loads
`configs/experiments/yolov8n_preliminary_baseline_v1.yaml`. A guard rejects any
drift from the approved configuration rather than silently changing a setting.
Only the seed, dataset, and output arguments vary. `exist_ok=False` prevents
overwriting. Model, image size, epochs, batch, deterministic setting, device,
workers, patience, pretrained, AMP, plots, save, and validation stay unchanged.

`trustpcb.dataset_config` loads `configs/local/paths.yaml` and generates the
runtime YAMLs in `configs/local/datasets/`. No split construction or image
processing occurs in this step. Set absolute project/dataset roots for DICC;
run commands from that project checkout with its existing YOLO environment.

Planning, inspection, and extraction do not import Ultralytics or PyTorch.
Training uses one blocking subprocess per run, with `PYTHONHASHSEED` set to the
integer seed **before interpreter startup**. No concurrent jobs are submitted.
Use the same installed training environment for all ten runs; package versions
and the effective Ultralytics `args.yaml` are retained for audit.

Before training or reuse, Git must report clean, committed execution inputs:
`src/trustpcb/`, the v1 baseline YAML, `configs/datasets/`, and `data/splits/`.
Tracked changes (including staged changes) and untracked files in those paths
stop execution. The provenance `git_dirty` flag covers these inputs only;
`runs/rq1/`, `results/rq1/`, `configs/local/` and runner locks do not make it dirty.
The Git commit hash is still recorded. Commit the reviewed implementation
explicitly before DICC training; the runner does not commit anything itself.

For older Git compatibility, cleanliness uses scoped `git diff --name-only -z`
and `git diff --cached --name-only -z` for unstaged/staged tracked changes, and
`git ls-files --others --exclude-standard -z` for untracked files. Git runs with
the checkout as its working directory, without `git -C` or scoped `git status`.
Any Git command failure stops the check rather than treating the inputs as clean.

Keep the baseline setting `model: yolov8n.pt`. Exactly one approved local
candidate must exist: `<project_root>/yolov8n.pt` or
`<project_root>/notebooks/yolov8n.pt`. The existing DICC file in `notebooks/`
is recognized without moving it. No recursive search or download is performed.
If neither file exists, or both exist (even with identical contents), stop
rather than choosing silently. Resolution is independent of the process cwd.
The runner requires this local file, streams its bytes through SHA-256, and
records its resolved path, hash, and byte size in each run's runtime provenance.
`runs/rq1/pretrained_model.json` freezes that identity across separate commands,
including `--remaining`. Missing/changed weights stop training or reuse before
a worker starts. Each worker uses the verified absolute path; the runner checks
the file again before and after every run and before skipping a completed run.
It never downloads weights. Do not remove the binding to switch pretrained
weights; missing bindings alongside existing RQ1 artifacts require inspection.
The `.pt` file remains excluded by the existing Git ignore rule.

## Lightweight commands

From the repository root (PowerShell: set `$env:PYTHONPATH = 'src'` first):

```bash
PYTHONPATH=src python -B -m trustpcb.rq1 plan
PYTHONPATH=src python -B -m unittest discover -s tests -v
```

`plan` emits JSON provenance to stdout without creating runs or needing a
dataset root. Redirect it to a chosen file to retain a planning snapshot.
`--seed <label>`, `--seed <label> --split <split>`, and `--remaining` also work
with `plan`. No extra seeds can be supplied; selection retains the frozen order.

## DICC-only training commands — do not execute locally

Only the first matched pair:

```bash
PYTHONPATH=src python -B -m trustpcb.rq1 train --seed 24209199 --dicc
```

Then all four remaining pairs, sequentially in the frozen order:

```bash
PYTHONPATH=src python -B -m trustpcb.rq1 train --remaining --dicc
```

One specified condition, or all ten sequentially:

```bash
PYTHONPATH=src python -B -m trustpcb.rq1 train --seed 02092026 --split supplied --dicc
PYTHONPATH=src python -B -m trustpcb.rq1 train --all --dicc
```

The CLI rejects training on Windows and requires `--dicc` elsewhere. The hidden
`_worker` command is an implementation detail, not a standalone entry point.
Do not run notebook training cells for this workflow.

## Provenance, completion, and restart behavior

Before each new run the runner exclusively creates the adjacent
`seed_<label>.provenance.json`. It records the experiment ID, exact label/value,
split, Git commit/dirty state, hashes of relevant source/config/split files,
baseline and submitted training settings, runtime manifest/YAML hashes, output
and checkpoint locations, timestamps, Python/package versions and environment.
It leaves YOLO's target directory absent so `exist_ok=False` remains effective.

Completion requires all of:

- The training call returned successfully to the runner.
- `args.yaml` agrees with the submitted settings and output location.
- `results.csv` has exactly contiguous epochs 1 through 100 and finite metrics.
- `weights/best.pt` exists and is nonempty.
- Matching successful sidecar and `rq1_complete.json` receipts exist.

Receipts record CSV/args hashes and output checkpoint size/mtime. Inspection never loads
or hashes the trained output checkpoint; this is completion evidence, not a verification of
checkpoint tensor contents. A crash after training but before receipts is
conservatively treated as ambiguous, even if CSV/checkpoint files exist.

On restart, a completed run is skipped only if its receipts/artifacts, current
plan, commit/source hashes and runtime data hashes still match. Any existing
incomplete, unrecognized, changed, or partial run stops execution. There is no
force-overwrite or automatic checkpoint-resume switch. Inspect such artifacts
on DICC and make an explicit preservation/recovery decision before retrying.

`runs/rq1/.runner.lock` prevents a second orchestrator from running concurrently.
An abrupt process/host termination can leave the lock; inspect the process and
artifacts before manually removing it. Ordinary exits release the lock.
Failures stop the sequence; later seeds are not launched.

## Result extraction and aggregation

After DICC training, read existing logs without inference or evaluation:

```bash
PYTHONPATH=src python -B -m trustpcb.rq1 extract
```

Outputs:

- `results/rq1/per_seed_metrics.csv`: ten rows in frozen run order; status and
  reason plus seed label/value, split, experiment ID, best epoch, precision,
  recall, mAP@0.50, mAP@0.50:0.95, run/checkpoint paths, execution Git commit and
  metric-selection definition. Missing/ambiguous runs have blank metric fields.
- `results/rq1/aggregate_metrics.csv`: one row per split/metric, completed and
  expected counts, status, mean, and **sample standard deviation (ddof=1)**.
  Mean/std stay blank until all five frozen seeds for that split are complete.

All four metrics come from the same CSV row: the epoch with highest logged
mAP@0.50:0.95, with the first epoch winning a logged tie. This retains notebook
03's preliminary CSV selection convention; it is not a new model evaluation.
`best_checkpoint` points to Ultralytics' `best.pt`. CSV rounding/tie handling
means the selected CSV epoch need not identify the exact checkpoint save epoch.
The CSV states this selection rule explicitly. Full raw logs remain available.

Reporting preserves each run's execution commit even when reporting code has
changed; scientific/configuration inputs must still agree. CSV has no string
column type: when importing with pandas use `dtype={"seed_label": str}`, and
import that column as text in spreadsheets to retain `02092026`.

Training output provenance contains machine-local absolute paths by design.
Checkpoints are already ignored by Git. No result metrics are supplied by this
implementation; all tests use temporary synthetic text artifacts.
