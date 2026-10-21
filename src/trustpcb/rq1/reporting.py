"""Build RQ1 reporting copies and presentation figures from saved evidence only.

This is new reporting code, not a historical experiment implementation. It never
calls workflow.extract_results, notebook cells, models, or statistical estimators.
Run with PYTHONPATH=src: python -B -m trustpcb.rq1.reporting [--verify-only].
Writes are confined to outputs/rq1; unrecognized or edited files block rebuilding.
"""

import argparse
from trustpcb.documentation_archive import verify_reporting_manifest, retired_summary
import base64
import csv
import hashlib
import io
import json
from pathlib import Path

from trustpcb.dataset_config import find_project_root
from trustpcb.rq1.archived_sources import read_source, notebook_record, read_notebook

SOURCE_COMMIT = "1937995fb78d979a66e237b7ab7facbe20ca0513"
GENERATOR = "src/trustpcb/rq1/reporting.py"


def source_record(root, path):
    if path.startswith('notebooks/') and path.endswith('.ipynb'):
        record = notebook_record(root, Path(path).name)
        read_source(root, record)
        return record
    return {'path': path, 'sha256': digest((Path(root) / path).read_bytes())}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(root):
    """Check all recorded output/source bytes and reject unlisted report files."""
    root = Path(root)
    manifest = json.loads((root / "outputs/rq1/manifest.json").read_text(encoding="utf-8"))
    return verify_reporting_manifest(root, "outputs/rq1", manifest)


def build(root):
    root = Path(root)
    output = root / "outputs/rq1"
    if output.exists():
        # A generator edit requires explicit review of the old manifest, not an
        # implicit overwrite of an existing presentation under a new identity.
        verify(root)
    records = []

    def emit(path, data, kind, sources, recipe):
        archived = retired_summary(root, "outputs/rq1", path)
        if archived is not None:
            records.append(archived)
            return
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({"path": path, "sha256": digest(data), "bytes": len(data),
                        "classification": kind, "recipe": recipe,
                        "sources": [source_record(root, p)
                                    for p in dict.fromkeys([*sources, GENERATOR])]})

    def copy(source, target):
        emit(target, (root / source).read_bytes(), "reporting_copy", [source],
             "Byte-for-byte copy; original remains authoritative.")

    def summary(experiment, text, sources):
        emit(f"{experiment}/summaries/README.md", text.encode("utf-8"),
             "authored_reporting_summary", sources, "Navigation and limitations; no new results.")

    def rows(source):
        with (root / source).open(encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))

    for source in sorted((root / "outputs/tables").glob("*.csv")):
        experiment = "preliminary_baseline" if source.name.startswith("preliminary_") else "dataset_audit"
        copy(source.relative_to(root).as_posix(), f"{experiment}/tables/{source.name}")
    for name in ("per_seed_metrics.csv", "aggregate_metrics.csv", "paired_differences.csv", "paired_summary.csv"):
        copy(f"results/rq1/{name}", f"matched_training/tables/{name}")
    copy("results/rq1/pretrained_model.json", "matched_training/summaries/pretrained_model.json")
    for split in ("smoke_test", "supplied", "similarity_aware"):
        base = f"runs/preliminary_baseline/v1/{split}"
        copy(f"{base}/results.csv", f"preliminary_baseline/tables/v1/{split}/results.csv")
        copy(f"{base}/results.png", f"preliminary_baseline/figures/v1/{split}/results.png")
        if split != "smoke_test":
            copy(f"{base}/confusion_matrix_normalized.png",
                 f"preliminary_baseline/figures/v1/{split}/confusion_matrix_normalized.png")

    source = "notebooks/01_dataset_structure_audit.ipynb"
    notebook = read_notebook(root, Path(source).name)
    # One representative existing panel from each display cell. No cell executes.
    for index in (11, 13):
        for number, item in enumerate(notebook["cells"][index].get("outputs", [])):
            png = item.get("data", {}).get("image/png")
            if png:
                if isinstance(png, list):
                    png = "".join(png)
                emit(f"dataset_audit/figures/notebook01_cell{index}_output{number}.png",
                     base64.b64decode(png), "exported_notebook_output", [source],
                     f"Decode existing image/png from zero-based cell {index}, output {number}; no rerender.")
                break

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    colors = ["#23658a", "#ce7435"]

    def figure(fig, target, source, recipe):
        data = io.BytesIO()
        fig.savefig(data, format="png", dpi=160, facecolor="white",
                    metadata={"Software": "TrustPCB RQ1 reporting"})
        plt.close(fig)
        emit(target, data.getvalue(), "newly_rendered_figure", [source], recipe)

    source = "outputs/tables/dataset_audit_summary.csv"
    saved = {r["metric"]: r["value"] for r in rows(source)}
    labels = ["Training images", "Validation images", "Total images"]
    values = [int(saved[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    bars = ax.barh(labels, values, color=[*colors, "#687883"])
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=5)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel("Images (saved audit counts)")
    ax.set_title("Historical dataset audit: supplied split")
    fig.supxlabel("Source: outputs/tables/dataset_audit_summary.csv · historical audit population", fontsize=8)
    figure(fig, "dataset_audit/figures/saved_image_counts.png", source,
           "Render the three saved image counts; no dataset recount or split reconstruction.")

    source = "outputs/tables/preliminary_baseline_comparison_v1.csv"
    saved = [r for r in rows(source) if r["Metric"] != "Best epoch"]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for offset, key, color in zip((-0.18, 0.18), ("Supplied", "Similarity-aware"), colors):
        bars = ax.bar([i + offset for i in range(len(saved))], [float(r[key]) for r in saved],
                      width=0.36, label=key, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xticks(range(len(saved)), [r["Metric"] for r in saved])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Saved metric value")
    ax.set_title("Preliminary baseline v1: saved best-epoch metrics")
    ax.legend(loc="lower right")
    fig.supxlabel("Single preliminary run per split · no uncertainty estimated", fontsize=8)
    figure(fig, "preliminary_baseline/figures/saved_metric_comparison.png", source,
           "Plot supplied/similarity-aware values already saved in comparison CSV; no best-epoch reselection.")

    source = "results/rq1/aggregate_metrics.csv"
    saved = rows(source)
    metrics = ["precision", "recall", "mAP@0.50", "mAP@0.50:0.95"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), layout="constrained")
    for ax, metric in zip(axes, metrics):
        selected = [next(r for r in saved if r["metric"] == metric and r["split"] == split)
                    for split in ("supplied", "similarity_aware")]
        bars = ax.bar([0, 1], [float(r["mean"]) for r in selected],
                      yerr=[float(r["std"]) for r in selected], capsize=5, color=colors)
        ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=13)
        ax.set_xticks([0, 1], ["Supplied", "Similarity\naware"])
        ax.set_ylim(0, 1)
        ax.set_title(metric)
    axes[0].set_ylabel("Saved mean metric")
    fig.suptitle("Matched training: five seeds per split")
    fig.supxlabel("Bars: saved means · error bars: saved sample SD (ddof=1), not confidence intervals", fontsize=9)
    figure(fig, "matched_training/figures/saved_mean_and_sd.png", source,
           "Render saved aggregate means and sample SD directly; no aggregation, fitting or interval calculation.")

    summary("dataset_audit", """# Dataset audit and similarity-aware split

Tables are byte-for-byte copies of six historical audit CSVs. Figures include two
existing notebook panels and one new rendering of saved image counts. Additional
panels remain embedded in archived notebook 01. Counts describe that historical audit;
do not substitute later split membership counts.

Original notebooks and all outputs are recoverable through the
[Git archive and Python conversion guide](../../../../docs/rq1/notebook_extraction.md).
Frozen split manifests remain in [data/splits](../../../../data/splits/).
Raw images/labels are external. Manual review decisions are evidence, not an
automated classifier. Extracted helper functions did not generate these records.
""", ["notebooks/01_dataset_structure_audit.ipynb", "notebooks/02_similarity_aware_split.ipynb",
       "outputs/tables/dataset_audit_summary.csv"])
    summary("preliminary_baseline", """# Preliminary baseline v1

Includes the saved comparison table and epoch logs for smoke_test, supplied and
similarity_aware; copied learning curves for all three; copied normalized confusion
matrices for the two full preliminary runs; and a newly rendered comparison figure.
The smoke run is preliminary execution evidence, not a matched-training replicate.

Original evidence: [runs/preliminary_baseline/v1](../../../../runs/preliminary_baseline/v1/).
Other precision/recall curves and prediction examples remain there. Execution code
is archived in Git; the reconstructed [preliminary workflow](../../../../src/trustpcb/rq1/preliminary_baseline.py)
is documented in the [conversion guide](../../../../docs/rq1/notebook_extraction.md).
Checkpoints are unavailable locally. No training or metric selection was rerun.
""", ["notebooks/03_yolo_baseline.ipynb", "outputs/tables/preliminary_baseline_comparison_v1.csv"])
    summary("matched_training", """# Matched repeated training

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
""", ["results/rq1/README.md", "results/rq1/paired_summary.csv", "results/rq1/paired_differences.csv"])
    emit("README.md", """# RQ1 research reporting

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
Complete reconstructed workflows: [audit](../../src/trustpcb/rq1/audit_workflow.py),
[split](../../src/trustpcb/rq1/split_workflow.py), and
[preliminary baseline](../../src/trustpcb/rq1/preliminary_baseline.py).
Shared historical kernels: [dataset audit](../../src/trustpcb/rq1/dataset_audit.py)
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
`python -B -m trustpcb.rq1.reporting --verify-only` checks recorded bytes,
including archived Git blobs and the recorded historical checkout transforms.
Full Git history is required; missing history is a verification failure.
`python -B -m trustpcb.rq1.reporting` builds/rebuilds these lightweight reports.
Existing edited/unlisted files or changed sources block rebuilding. No raw data or
ML environment is needed. Do not use the research extraction/training commands to
refresh this reporting area.

DsPCBSD+ derived visual evidence retains the dataset attribution in the
[repository README](../../README.md) (Lv et al., 2024; CC BY 4.0).
""".encode("utf-8"), "authored_reporting_summary", ["README.md", "docs/rq1/notebook_extraction.md"],
         "RQ1 navigation and availability guide; no scientific conclusions added.")
    manifest = {"schema_version": 2, "source_snapshot_commit": SOURCE_COMMIT,
                "hash_convention": "SHA-256 of exact local bytes; not a replacement for scientific LF-canonical hashes",
                "validation_sources": [source_record(root, p) for p in
                                       (GENERATOR, 'src/trustpcb/rq1/archived_sources.py',
                                        'docs/rq1/notebook_archive.json')],
                "artifacts": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = find_project_root()
    count = verify(root) if args.verify_only else build(root)
    print(f"Verified {count} RQ1 reporting artifacts.")


if __name__ == "__main__":
    main()
