"""RQ2 presentation copies and plots from saved evidence; no research execution.

Run from the checkout with PYTHONPATH=src:
    python -B -m trustpcb.rq2.reporting [--verify-only]
Matplotlib is imported only when rendering. No scientific workflow is imported.
All writes are confined to outputs/rq2. Existing reports must verify before rebuild.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import textwrap

BASE_COMMIT = "b703ba010d677d557a05ed8c4d8e4f2b4ffce9c9"
GENERATOR = "src/trustpcb/rq2/reporting.py"
GUIDE = "docs/rq2/reporting.md"
OUTPUT = "outputs/rq2"
EXPERIMENTS = {
    "risk_evaluation": {
        "module": "risk_evaluation", "title": "Development risk evaluation · IoU ≥ 0.50",
        "purpose": "Development risk ablations and incorrect-prediction ranking.",
        "caveat": "Development diagnostics, not independent test evidence. bootstrap_metrics.csv is absent locally; saved interval summaries remain available.",
        "csv": ["method_metrics.csv", "pairwise_auroc_differences.csv", "per_class_diagnostics.csv", "signal_correlations.csv", "signal_descriptives.csv"],
        "json": ["summary.json", "bootstrap_summary.json", "provenance.json"],
    },
    "risk_sensitivity_iou75": {
        "module": "risk_sensitivity", "title": "Development sensitivity · IoU ≥ 0.75",
        "purpose": "Secondary correctness-definition sensitivity analysis on development evidence.",
        "caveat": "Secondary analysis; it does not replace the primary IoU ≥ 0.50 results. bootstrap_metrics.csv is absent locally.",
        "csv": ["method_metrics.csv", "pairwise_auroc_differences.csv"],
        "json": ["summary.json", "bootstrap_summary.json", "provenance.json"],
    },
    "weighted_fusion": {
        "module": "weighted_fusion", "title": "Weighted fusion · development selection",
        "purpose": "Frozen development-only selection of confidence/class/localisation fusion weights.",
        "caveat": "Post-selection development metrics are descriptive, not independent generalisation evidence. The 5,151-row candidate grid remains at its original path, linked and hashed rather than duplicated. No weights are selected again.",
        "csv": ["development_metrics.csv", "sensitivity_iou75_metrics.csv", "fold_assignments.csv"],
        "json": ["summary.json", "selected_weights.json", "provenance.json"],
    },
    "final_evaluation": {
        "module": "final_evaluation", "title": "Original final evaluation",
        "purpose": "Frozen Beta calibration and original equal-weight/ablation ranking on final-test evidence.",
        "caveat": "Held out from RQ2 training/development, but previously used for RQ1 validation. Detailed prediction, stability, transform-match and inference-audit files, and both bootstrap_metrics.csv files, are absent locally. Original provenance is copied unchanged despite those missing dependencies.",
        "csv": ["calibration_bins.csv", "primary_iou50/method_metrics.csv", "primary_iou50/pairwise_auroc_differences.csv", "primary_iou50/signal_correlations.csv", "primary_iou50/signal_descriptives.csv", "sensitivity_iou75/method_metrics.csv", "sensitivity_iou75/pairwise_auroc_differences.csv"],
        "json": ["calibration_metrics.json", "transformation_manifest.json", "provenance.json", "primary_iou50/summary.json", "primary_iou50/bootstrap_summary.json", "sensitivity_iou75/summary.json", "sensitivity_iou75/bootstrap_summary.json"],
    },
    "weighted_final_evaluation": {
        "module": "weighted_final_evaluation", "title": "Frozen weighted final evaluation",
        "purpose": "Previously selected weights applied to final-test signals; saved primary/secondary ranking and paired differences.",
        "caveat": "Held out from RQ2 training/development, previously used for RQ1 validation. No test-driven weight revision. Summaries exist locally; the complete upstream prediction/calibration chain does not.",
        "csv": ["primary_method_metrics.csv", "primary_pairwise_differences.csv", "sensitivity_iou75_method_metrics.csv", "sensitivity_iou75_pairwise_differences.csv"],
        "json": ["summary.json", "provenance.json"],
    },
}
UPSTREAM = [
    ("data_partition", "Local frozen manifests and verification report in data/splits/rq2_train_development_split; dataset template in configs/datasets/dspcbsd_rq2_train_dev.yaml."),
    ("detector_training", "Run directory/checkpoint absent locally; locate completed DICC artifacts and verify their recorded identities."),
    ("confidence_distribution", "Raw development predictions, inventory and confidence histogram absent locally."),
    ("calibration_analysis", "Development correctness/calibration tables and raw reliability/histogram figures absent locally."),
    ("calibrator_comparison", "Development comparison tables and reliability_comparison.png absent locally."),
    ("final_calibrator", "Development-fitted Beta mapping and its upstream fitting artifacts absent locally; downstream summaries are not substitutes."),
    ("transformation_stability", "Development stability and transformed-prediction artifacts absent locally."),
]
MISSING = [f"runs/rq2/{stage}" for stage, _ in UPSTREAM if stage != "data_partition"] + [
    "runs/rq2/risk_evaluation/bootstrap_metrics.csv",
    "runs/rq2/risk_sensitivity_iou75/bootstrap_metrics.csv",
] + [f"runs/rq2/final_evaluation/{name}" for name in (
    "inference_audit.json", "original_predictions.csv", "labelled_predictions.csv",
    "prediction_stability_scores.csv", "prediction_transform_matches.csv", "transformed_predictions.csv",
    "primary_iou50/bootstrap_metrics.csv", "sensitivity_iou75/bootstrap_metrics.csv")]
REFERENCES = ["runs/rq2/weighted_fusion/candidate_weights.csv",
              "data/splits/rq2_train_development_split/detector_train_images.txt",
              "data/splits/rq2_train_development_split/development_calibration_images.txt",
              "configs/datasets/dspcbsd_rq2_train_dev.yaml"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inside(root, relative):
    """Confine manifest paths to their declared root, including on Windows."""
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe relative path: {relative}")
    result = Path(root).joinpath(*part.parts).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path escapes root: {relative}")
    return result


def source_record(root, path):
    data = inside(root, path).read_bytes()
    return {"path": path, "sha256": digest(data), "bytes": len(data)}


def verify(root):
    """Verify exact local bytes; this does not verify the scientific input chain."""
    root = Path(root)
    output = inside(root, OUTPUT)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected = {"manifest.json"}
    for record in manifest["artifacts"]:
        if record["path"] in expected:
            raise ValueError("Duplicate reporting path")
        expected.add(record["path"])
        if digest(inside(output, record["path"]).read_bytes()) != record["sha256"]:
            raise ValueError(f"Reporting artifact changed: {record['path']}")
        for source in record["sources"]:
            if digest(inside(root, source["path"]).read_bytes()) != source["sha256"]:
                raise ValueError(f"Reporting source changed: {source['path']}")
    for reference in manifest["references"]:
        if digest(inside(root, reference["path"]).read_bytes()) != reference["sha256"]:
            raise ValueError(f"Referenced evidence changed: {reference['path']}")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError(f"Unlisted/missing reporting files: {actual ^ expected}")
    return len(manifest["artifacts"])


def number(value):
    """Missing/undefined evidence stays missing, never a fabricated zero."""
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def interval(row, lower, upper):
    lo, hi = number(row.get(lower)), number(row.get(upper))
    if (lo is None) != (hi is None) or (lo is not None and lo > hi):
        raise ValueError(f"Incomplete or reversed saved interval: {lower}/{upper}")
    return lo, hi


def read_rows(root, source):
    with inside(root, source).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def label(method):
    # Display formatting only; original method identifiers remain in copied CSVs.
    return textwrap.fill(method.replace("_", " "), 29)


def plot_metrics(plt, rows, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.5, len(rows) * 0.62 + 1.6)), layout="constrained")
    has_intervals = False
    for ax, metric, color in zip(axes, ("auroc", "auprc"), ("#23658a", "#b26330")):
        for i, row in enumerate(rows):
            value = number(row[metric])
            lo, hi = interval(row, f"{metric}_lower_95", f"{metric}_upper_95")
            if value is not None:
                ax.plot(value, i, "o", color=color, markersize=6)
                if lo is not None:
                    ax.hlines(i, lo, hi, color=color, linewidth=2)
                    has_intervals = True
            ax.text(1.025, i, "undefined" if value is None else f"{value:.4f}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        ax.set_yticks(range(len(rows)), [label(r["method"]) for r in rows], fontsize=9)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel(metric.upper())
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(title, fontsize=15)
    suffix = "Lines: saved 95% bootstrap intervals" if has_intervals else "No intervals supplied in this table"
    fig.supxlabel(f"Incorrect predictions are positive · dots: saved observed metrics · {suffix}", fontsize=9)
    return fig


def plot_differences(plt, rows, title):
    metrics = list(dict.fromkeys(r.get("metric", "auroc") for r in rows))
    fig, axes = plt.subplots(1, len(metrics), squeeze=False,
                             figsize=(7 * len(metrics), max(4.5, len(rows) / len(metrics) * 0.9 + 1.7)),
                             layout="constrained")
    for ax, metric in zip(axes[0], metrics):
        selected = [r for r in rows if r.get("metric", "auroc") == metric]
        for i, row in enumerate(selected):
            value = number(row.get("observed_difference", row.get("observed_delta_auroc")))
            lo, hi = interval(row, "lower_95", "upper_95")
            if value is not None:
                ax.plot(value, i, "o", color="#23658a")
                if lo is not None:
                    ax.hlines(i, lo, hi, color="#23658a", linewidth=2)
            ax.text(1.025, i, "undefined" if value is None else f"{value:+.4f}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        ax.set_yticks(range(len(selected)),
                      [label(r["method_A"]) + "\nminus " + label(r["method_B"]) for r in selected], fontsize=9)
        ax.set_ylim(len(selected) - 0.5, -0.5)
        ax.axvline(0, color="#687883", linestyle="--", linewidth=1)
        ax.set_xlabel(f"Saved difference in {metric.upper()} (method A − method B)")
        ax.grid(axis="x", alpha=0.2)
        ax.margins(x=0.15)
    fig.suptitle(title, fontsize=15)
    fig.supxlabel("Dots: saved observed differences · lines: saved paired 95% bootstrap intervals · no new bootstrap", fontsize=9)
    return fig


def build(root):
    root = Path(root).resolve()
    output = inside(root, OUTPUT)
    if output.exists():
        verify(root)  # Fail before any write if an existing artifact/source changed.
    # Check the complete fixed input list before making output directories.
    copies = []
    for experiment, spec in EXPERIMENTS.items():
        for extension, folder in (("csv", "tables"), ("json", "summaries")):
            for name in spec[extension]:
                copies.append((f"runs/rq2/{experiment}/{name}", f"{experiment}/{folder}/{name}"))
    copies.append(("data/splits/rq2_train_development_split/verification_report.json",
                   "data_partition/summaries/verification_report.json"))
    for path in [s for s, _ in copies] + REFERENCES + [GENERATOR, GUIDE]:
        source_record(root, path)
    for path in MISSING:
        if inside(root, path).exists():
            raise ValueError(f"Availability changed; review reporting inventory before rebuilding: {path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white"})
    records = []

    def emit(path, data, kind, sources, recipe):
        target = inside(output, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({"path": path, "sha256": digest(data), "bytes": len(data),
                        "classification": kind, "recipe": recipe,
                        "sources": [source_record(root, s) for s in dict.fromkeys([*sources, GENERATOR])]})

    def save_figure(fig, path, sources, recipe):
        data = io.BytesIO()
        fig.savefig(data, format="png", dpi=160, facecolor="white", metadata={"Software": "TrustPCB RQ2 reporting"})
        plt.close(fig)
        emit(path, data.getvalue(), "newly_rendered_visualization", sources, recipe)

    for source, target in copies:
        emit(target, inside(root, source).read_bytes(), "reporting_copy", [source],
             "Byte-for-byte copy; canonical original evidence remains authoritative and unchanged.")

    # Every plot reads already-saved metrics. No ranking, calibration, resampling or selection is called.
    for experiment, spec in EXPERIMENTS.items():
        metric_files = [name for name in spec["csv"] if name.endswith("metrics.csv")]
        difference_files = [name for name in spec["csv"] if "differences" in name]
        for name in metric_files + difference_files:
            source = f"runs/rq2/{experiment}/{name}"
            suffix = " · secondary IoU ≥ 0.75" if "sensitivity_iou75" in name else ""
            if experiment in ("final_evaluation", "weighted_final_evaluation") and not suffix:
                suffix = " · primary IoU ≥ 0.50"
            is_difference = name in difference_files
            fig = (plot_differences if is_difference else plot_metrics)(plt, read_rows(root, source), spec["title"] + suffix)
            filename = name.removesuffix(".csv").replace("/", "_") + ".png"
            save_figure(fig, f"{experiment}/figures/{filename}", [source],
                        "Render saved observed values and saved interval endpoints, preserving row order and method identities; undefined entries stay undefined.")

    source = "runs/rq2/weighted_fusion/selected_weights.json"
    saved = json.loads(inside(root, source).read_text(encoding="utf-8"))["integer_percentages"]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    values = [saved[key] for key in ("conf_percent", "class_percent", "localisation_percent")]
    bars = ax.barh(["Calibrated confidence", "Class consistency", "Localisation stability"], values,
                   color=["#23658a", "#b26330", "#52846b"])
    ax.bar_label(bars, labels=[f"{v}%" for v in values], padding=5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Frozen integer percentage")
    ax.set_title("Weighted fusion: saved development-selected weights")
    fig.supxlabel("No selection or optimisation rerun; values copied from selected_weights.json", fontsize=9)
    save_figure(fig, "weighted_fusion/figures/selected_weights.png", [source],
                "Display stored integer percentages directly, without renormalizing or reselecting weights.")

    source = "runs/rq2/final_evaluation/calibration_bins.csv"
    rows = read_rows(root, source)
    fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
    ax.plot([0, 1], [0, 1], "--", color="#687883", label="Ideal calibration")
    for method, color in (("Raw", "#b26330"), ("Beta", "#23658a")):
        selected = [r for r in rows if r["method"] == method and int(r["count"]) > 0]
        ax.plot([number(r["mean_confidence"]) for r in selected],
                [number(r["correctness_rate"]) for r in selected], "o-", color=color, label=method)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Saved bin mean confidence", ylabel="Saved bin correctness rate",
           title="Final-test reliability · IoU ≥ 0.50")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.supxlabel("Saved nonempty bins only · no prediction rebinning or calibrator fitting", fontsize=9)
    save_figure(fig, "final_evaluation/figures/reliability_saved_bins.png", [source],
                "Plot stored nonempty-bin means and correctness rates. Connecting lines guide the eye, not a fitted curve.")

    source = "runs/rq2/final_evaluation/calibration_metrics.json"
    saved = json.loads(inside(root, source).read_text(encoding="utf-8"))["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), layout="constrained")
    for ax, metric in zip(axes, ("brier", "nll", "ece")):
        bars = ax.bar(["Raw", "Beta"], [saved[m][metric] for m in ("Raw", "Beta")], color=["#b26330", "#23658a"])
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.margins(y=0.2)
        ax.set_title(metric.upper())
    fig.suptitle("Final-test calibration: saved metrics · IoU ≥ 0.50")
    fig.supxlabel("Stored values only; no calibration fitting or metric calculation", fontsize=9)
    save_figure(fig, "final_evaluation/figures/calibration_saved_metrics.png", [source],
                "Render saved Brier, NLL and ECE values from calibration_metrics.json; no recomputation.")

    for experiment, spec in EXPERIMENTS.items():
        extra = ""
        if experiment == "weighted_fusion":
            extra = "\n[Original candidate grid](../../../../runs/rq2/weighted_fusion/candidate_weights.csv) is referenced in the manifest.\n"
        text = f"""# {spec['title']}

{spec['purpose']}

- [Tables](../tables/): unchanged reporting copies of saved CSVs.
- [Figures](../figures/): newly rendered visualizations, not original DICC figures.
- JSON files beside this summary are unchanged copies of saved summaries/provenance.
- [Original evidence](../../../../runs/rq2/{experiment}/) remains authoritative.
- [Scientific implementation](../../../../src/trustpcb/rq2/{spec['module']}.py)
- [Tests](../../../../tests/rq2/test_rq2_{spec['module']}.py)
- [Protocol](../../../../docs/rq2/rq2_{spec['module']}.md)

{spec['caveat']}
{extra}
Observed AUROC/AUPRC and interval endpoints come directly from saved files. Missing
values are not zeros. Original `trustpcb_risk` denotes the original equal-weight
composite; it must not be confused with `weighted_trustpcb_risk`. Paired differences
retain the saved method-A-minus-method-B direction. No significance claims are added.
Copied historical provenance describes the original execution, not this reporting run.
See the [reporting manifest](../../manifest.json) for paths, hashes and derivations.
"""
        emit(f"{experiment}/summaries/README.md", text.encode("utf-8"), "authored_reporting_summary",
             [GUIDE, f"runs/rq2/{experiment}/provenance.json"], "Navigation, scope and known local gaps; no new scientific conclusions.")

    text = """# RQ2 data partition

`verification_report.json` is a byte-identical reporting copy of the frozen
[original verification report](../../../../data/splits/rq2_train_development_split/verification_report.json).
Membership files remain in that original directory; no split was reconstructed.
[Implementation](../../../../src/trustpcb/rq2/data_partition.py),
[tests](../../../../tests/rq2/test_rq2_data_partition.py), and
[protocol](../../../../docs/rq2/rq2_data_partition.md) retain their scientific roles.
There are no local partition figure files to copy; no artificial figure/table
folders are added. Original hashes and historical protocol labels are unchanged.
"""
    emit("data_partition/summaries/README.md", text.encode("utf-8"), "authored_reporting_summary",
         [GUIDE, "data/splits/rq2_train_development_split/verification_report.json"], "Local partition evidence navigation.")

    rows = "\n".join(f"| [{name}]({name}/summaries/README.md) | {s['purpose']} | [tables]({name}/tables/) · [figures]({name}/figures/) · [summaries]({name}/summaries/) |"
                     for name, s in EXPERIMENTS.items())
    upstream = "\n".join(f"| [{stage}](../../src/trustpcb/rq2/{stage}.py) | [test](../../tests/rq2/test_rq2_{stage}.py) | {availability} |"
                         for stage, availability in UPSTREAM)
    text = f"""# RQ2 research reporting

RQ2 investigates confidence calibration, transformation consistency/localisation
stability and frozen prediction-risk fusion for PCB defect detection.

| Experiment | Purpose | Reporting materials |
|---|---|---|
{rows}

Each experiment summary links its canonical Python module, relocated tests and
protocol. The original 13 modules and compatibility shims remain unchanged.
[Data-partition evidence](data_partition/summaries/README.md) is also available.

## Upstream availability

| Stage / canonical Python | Tests | Availability in this checkout |
|---|---|---|
{upstream}

Raw images/labels and model checkpoints are external. Missing paths are documented
dependencies, not proof that artifacts were never generated or that their DICC
locations are verified. Retrieve and verify existing evidence rather than rerunning.
Detailed missing final-evaluation files are listed in [manifest.json](manifest.json)
and [the artifact inventory](../../docs/artifact_inventory.md).

## Evidence and presentation

- Original evidence stays in `runs/rq2/` and `data/splits/`; no writer destinations,
  schemas, memberships, identities or frozen weights change.
- CSV/JSON files here are **reporting copies**, preserving original bytes and metadata.
- All figures here are **newly rendered visualizations** from saved numerical evidence;
  no original RQ2 figure files were available locally. These are not recreated ROC/PR
  curves: aggregate AUROC/AUPRC cannot supply those missing curves.
- Development selection, original final evaluation and weighted final evaluation
  remain distinct. Final-test images were held out from RQ2 training/development,
  but previously used for RQ1 validation. Saved intervals are displayed, not recalculated.
- The candidate grid is linked at its original path instead of duplicated. No large
  prediction tables, dataset images or checkpoints are copied.
- This manifest verifies local reporting provenance, not the complete scientific
  dependency chain. Historical provenance is never rewritten to conceal local gaps.

## Reproduction and verification

New presentation code: [reporting.py](../../src/trustpcb/rq2/reporting.py).
No scientific workflow is imported or executed. Matplotlib is loaded only to render.
With `PYTHONPATH=src`, use `python -B -m trustpcb.rq2.reporting --verify-only` to
verify all recorded sources, outputs and referenced originals. Use
`python -B -m trustpcb.rq2.reporting` to build/rebuild this small reporting area.
Existing edited/unlisted reports, changed source hashes or newly available upstream
artifacts block rebuilding for inventory review. Hashes describe exact local bytes,
including line endings; they do not replace historical LF-canonical scientific hashes.

The manifest records source paths/SHA-256, output SHA-256, recipes, generator identity
and rendering versions. Its base commit is the pre-batch reference, not the original
execution commit of each experiment. `manifest.json` is the inventory itself.

[Reporting protocol and test commands](../../docs/rq2/reporting.md) ·
[RQ1 reporting](../rq1/README.md)
"""
    emit("README.md", text.encode("utf-8"), "authored_reporting_summary", [GUIDE, "docs/artifact_inventory.md"],
         "RQ2 navigation and availability guide; no new results.")
    manifest = {"schema_version": 1, "base_commit": BASE_COMMIT,
                "hash_convention": "SHA-256 of exact local file bytes, including line endings",
                "rendering": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "backend": "Agg", "dpi": 160},
                "artifacts": records, "references": [source_record(root, p) for p in REFERENCES],
                "unavailable_local": [{"path": p, "status": "absent locally; documented dependency, external location unverified"} for p in MISSING]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    count = verify(root) if args.verify_only else build(root)
    print(f"Verified {count} RQ2 reporting artifacts.")


if __name__ == "__main__":
    main()
