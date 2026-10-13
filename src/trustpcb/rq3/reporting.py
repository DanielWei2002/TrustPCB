"""Present saved RQ3 evidence without running scientific workflows.

PYTHONPATH=src python -B -m trustpcb.rq3.reporting [--verify-only]
Only outputs/rq3 is written. Matplotlib is lazy. Reused RQ2 reporting helpers
perform file hashing, CSV reading and missing-value handling, not experiments.
"""

import argparse
import io
import json
from pathlib import Path
import platform

from trustpcb.rq2.reporting import digest, inside, interval, number, read_rows, source_record

BASE_COMMIT = "f92b69baef91a1ed54cf893b1e659cf65368cffd"
GENERATOR = "src/trustpcb/rq3/reporting.py"
HELPERS = "src/trustpcb/rq2/reporting.py"
GUIDE = "docs/rq3/reporting.md"
OUTPUT = "outputs/rq3"
DEVELOPMENT = "development_selective_review"
EXTENSION = DEVELOPMENT + "/reporting_extension"
SPECS = {
    DEVELOPMENT: {
        "id": "rq3_development_selective_review", "module": "selective_review",
        "title": "Original development selective review",
        "csv": ["aurc.csv", "bootstrap_intervals.csv", "crop_referral_metrics.csv"],
        "json": ["development_thresholds.json", "summary.json", "bootstrap_summary.json", "provenance.json"],
        "txt": ["rq3_dicc_run_info.txt"],
        "scope": "Original 821-crop development protocol evidence; not independent test generalisation evidence. The original CSV schema remains intact. Supplementary automatic-error metrics and paired differences live in reporting_extension, not in this parent.",
    },
    EXTENSION: {
        "id": "rq3_development_reporting_extension", "module": "selective_review",
        "title": "Development reporting extension",
        "csv": ["crop_referral_metrics.csv", "bootstrap_intervals.csv", "paired_bootstrap_differences.csv"],
        "json": ["bootstrap_summary.json", "provenance.json"], "txt": [],
        "scope": "Historical supplementary development evidence, including remaining automatic error and paired differences. It supplements the parent; it does not replace original thresholds, outputs or schema. No bootstrap or extension command is rerun.",
    },
    "computational_efficiency": {
        "id": "rq3_computational_efficiency", "module": "computational_efficiency",
        "title": "DICC computational efficiency",
        "csv": ["repetition_timings.csv"], "json": ["summary.json", "provenance.json"], "txt": [],
        "scope": "Saved five-repetition DICC benchmark on 821 development crops, batch 1. Timings include the recorded pipeline work and exclude startup/model loading, provenance validation, hashing and output writes. Sample SD is not a confidence interval. These measurements are not evidence of factory-ready or real-time operation.",
    },
    "final_evaluation": {
        "id": "rq3_final_evaluation", "module": "final_evaluation",
        "title": "Frozen-threshold final evaluation",
        "csv": ["aurc.csv", "bootstrap_intervals.csv", "crop_referral_metrics.csv", "paired_bootstrap_differences.csv"],
        "json": ["summary.json", "bootstrap_summary.json", "provenance.json"], "txt": [],
        "scope": "Fixed development thresholds applied to 2,052 test crops. This population was held out from RQ2 training/development but previously used for RQ1 validation. Plots retain achieved test workloads, not nominal budget targets; no test quantiles or threshold adjustment is performed.",
    },
}
METHODS = {
    "raw_confidence_risk": ("Raw confidence", "#24658c", "o"),
    "calibrated_confidence_risk": ("Calibrated confidence", "#bd7436", "s"),
    "class_consistency_risk": ("Class consistency", "#777777", "^"),
    "localisation_stability_risk": ("Localisation stability", "#45856b", "D"),
    "weighted_trustpcb_risk": ("Weighted TrustPCB", "#805a9d", "x"),
}
REFERENCES = [f"runs/rq3/{experiment}/{name}" for experiment in (DEVELOPMENT, "final_evaluation")
              for name in ("prediction_risk_coverage.csv", "crop_risks.csv")]
UNAVAILABLE = ["runs/rq2/" + name for name in (
    "detector_training", "confidence_distribution", "calibration_analysis", "calibrator_comparison",
    "final_calibrator", "transformation_stability", "final_evaluation/original_predictions.csv",
    "final_evaluation/labelled_predictions.csv", "final_evaluation/prediction_stability_scores.csv",
    "final_evaluation/prediction_transform_matches.csv", "final_evaluation/transformed_predictions.csv")]


def verify(root):
    """Verify all local reporting bytes/references, not missing scientific inputs."""
    root = Path(root)
    output = inside(root, OUTPUT)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected = {"manifest.json"}
    for record in manifest["artifacts"]:
        if record["path"] in expected:
            raise ValueError("Duplicate artifact path")
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


def interval_index(rows):
    result = {}
    for row in rows:
        key = (row["method"], number(row["budget_percent"]), row["metric"])
        if key in result:
            raise ValueError(f"Duplicate interval identity: {key}")
        result[key] = interval(row, "lower_95", "upper_95")
    return result


def curve_points(rows, method):
    selected = [r for r in rows if r["method"] == method]
    # Missing risk at zero coverage remains undefined. No replacement with zero,
    # interpolation, re-ranking, tie splitting or AURC integration occurs here.
    return ([number(r["coverage"]) for r in selected],
            [float("nan") if number(r["empirical_risk"]) is None else number(r["empirical_risk"]) for r in selected])


def plot_curve(plt, rows, title):
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for method, (label, color, _) in METHODS.items():
        x, y = curve_points(rows, method)
        ax.step(x, y, where="pre", label=label, color=color,
                linestyle="--" if method == "calibrated_confidence_risk" else "-", linewidth=1.3)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Saved retained-prediction coverage",
           ylabel="Saved empirical error risk", title=title)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2)
    fig.supxlabel("All saved score boundaries · right-endpoint steps · undefined zero-coverage risk is not set to zero", fontsize=8)
    return fig


def plot_aurc(plt, rows, intervals, title):
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for i, row in enumerate(rows):
        method = row["method"]
        value = number(row["aurc"])
        lo, hi = intervals.get((method, None, "aurc"), (None, None))
        if value is not None:
            ax.plot(value, i, "o", color=METHODS[method][1])
            if lo is not None:
                ax.hlines(i, lo, hi, color=METHODS[method][1], linewidth=2)
        ax.text(1.015, i, "undefined" if value is None else f"{value:.4f}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [METHODS[r["method"]][0] for r in rows])
    ax.set(xlim=(0, 1), ylim=(len(rows) - 0.5, -0.5), xlabel="Saved AURC (lower is better)", title=title)
    fig.supxlabel("Dots: saved observed AURC · lines: saved 95% bootstrap intervals · no integration or resampling", fontsize=8)
    return fig


def plot_referral(plt, rows, intervals, title):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
    for ax, budget in zip(axes, (5, 10, 20)):
        for method, (label, color, marker) in METHODS.items():
            row = next(r for r in rows if r["method"] == method and number(r["budget_percent"]) == budget)
            # Use workload, never budget_percent/100 or development_achieved_workload.
            x, y = number(row["workload"]), number(row["error_capture"])
            if x is None or y is None:
                continue
            xlo, xhi = intervals.get((method, budget, "workload"), (None, None))
            ylo, yhi = intervals.get((method, budget, "error_capture"), (None, None))
            if xlo is not None:
                ax.hlines(y, xlo, xhi, color=color, alpha=0.7, linewidth=1)
            if ylo is not None:
                ax.vlines(x, ylo, yhi, color=color, alpha=0.7, linewidth=1)
            ax.plot(x, y, marker=marker, linestyle="none", color=color,
                    markerfacecolor="none", markersize=7, label=label)
        ax.set(title=f"Development nominal budget: {budget}%", xlabel="Achieved crop workload (fraction)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Captured incorrect predictions (fraction)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle(title + "\nSaved achieved operating points and marginal 95% intervals", fontsize=14)
    return fig


def plot_automatic(plt, rows, intervals, title):
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for offset, (method, (label, color, marker)) in zip((-0.24, -0.12, 0, 0.12, 0.24), METHODS.items()):
        selected = [r for r in rows if r["method"] == method]
        for row in selected:
            budget = number(row["budget_percent"])
            x = (5, 10, 20).index(budget) + offset
            value = number(row["remaining_automatic_error_rate"])
            lo, hi = intervals.get((method, budget, "remaining_automatic_error_rate"), (None, None))
            if value is not None:
                if lo is not None:
                    ax.vlines(x, lo, hi, color=color, linewidth=1.5)
                ax.plot(x, value, marker=marker, color=color, linestyle="none", markerfacecolor="none",
                        label=label if budget == 5 else None)
    ax.set_xticks([0, 1, 2], ["5%", "10%", "20%"])
    ax.set(xlabel="Development nominal budget (group labels, not achieved test workload)",
           ylabel="Saved remaining automatic error rate", ylim=(0, 1), title=title)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    fig.supxlabel("Points and 95% interval endpoints are saved values; horizontal offsets separate methods only", fontsize=8)
    return fig


def plot_paired(plt, rows, metric, title):
    selected = [r for r in rows if r["metric"] == metric]
    fig, ax = plt.subplots(figsize=(10, max(4, len(selected) * 0.7 + 1.4)), layout="constrained")
    labels = []
    for i, row in enumerate(selected):
        value = number(row["observed_difference"])
        lo, hi = interval(row, "lower_95", "upper_95")
        if value is not None:
            ax.plot(value, i, "o", color="#24658c")
            if lo is not None:
                ax.hlines(i, lo, hi, color="#24658c", linewidth=2)
        budget = row["budget_percent"]
        labels.append(f"{METHODS[row['method_A']][0]} minus\n{METHODS[row['method_B']][0]}" + (f" · {budget}% budget" if budget else ""))
        ax.text(1.02, i, "undefined" if value is None else f"{value:+.5f}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.set_yticks(range(len(selected)), labels, fontsize=9)
    ax.set_ylim(len(selected) - 0.5, -0.5)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Saved observed difference (method A − method B)")
    ax.set_title(title + "\n" + metric.replace("_", " "))
    ax.margins(x=0.15)
    direction = "Positive favours weighted for capture" if metric == "error_capture" else "Negative favours weighted for this metric"
    fig.supxlabel(direction + " · lines: saved paired 95% intervals · no new significance conclusions", fontsize=8)
    return fig


def plot_runtime(plt, summary, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for i, (pipeline, color) in enumerate((("baseline", "#24658c"), ("trustpcb", "#805a9d"))):
        saved = summary["pipelines"][pipeline]
        axes[0].bar(i, saved["mean_seconds"], yerr=saved["standard_deviation_seconds"], color=color, capsize=5)
        axes[0].text(i, saved["mean_seconds"] + 25, f"{saved['mean_seconds']:.3f} s", ha="center", fontsize=9)
        selected = [r for r in rows if r["pipeline"] == pipeline]
        axes[1].plot([number(r["repetition"]) for r in selected], [number(r["seconds"]) for r in selected],
                     "o-", color=color, label=pipeline)
    axes[0].set_xticks([0, 1], ["Baseline", "TrustPCB"])
    axes[0].set(ylabel="Seconds per 821-crop repetition", title="Saved mean ± sample SD")
    axes[0].margins(y=0.16)
    axes[1].set(yscale="log", xlabel="Repetition", ylabel="Saved seconds (log scale)", title="Five saved timings per pipeline")
    axes[1].set_xticks([1, 2, 3, 4, 5])
    axes[1].legend()
    fig.suptitle("DICC computational efficiency · batch 1", fontsize=15)
    fig.supxlabel("Recorded protocol excludes startup/model loading and output writes · no real-time or factory-readiness claim", fontsize=8)
    return fig


def build(root):
    root = Path(root).resolve()
    output = inside(root, OUTPUT)
    if output.exists():
        verify(root)
    copies = []
    for experiment, spec in SPECS.items():
        provenance = json.loads(inside(root, f"runs/rq3/{experiment}/provenance.json").read_text(encoding="utf-8"))
        if provenance["experiment"] != spec["id"]:
            raise ValueError(f"Unexpected scientific identity: {experiment}")
        for extension, folder in (("csv", "tables"), ("json", "summaries"), ("txt", "summaries")):
            copies.extend((experiment, f"runs/rq3/{experiment}/{name}", f"{experiment}/{folder}/{name}") for name in spec[extension])
    for path in [s for _, s, _ in copies] + REFERENCES + [GENERATOR, HELPERS, GUIDE]:
        source_record(root, path)
    for path in UNAVAILABLE:
        if inside(root, path).exists():
            raise ValueError(f"Availability changed; review inventory before rebuilding: {path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "path.simplify": False})
    records = []

    def emit(experiment, path, data, kind, sources, recipe):
        target = inside(output, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({"path": path, "sha256": digest(data), "bytes": len(data),
                        "experiment_identity": SPECS[experiment]["id"] if experiment else "rq3_reporting_index",
                        "availability": "local", "classification": kind, "recipe": recipe,
                        "derivation_scripts": [GENERATOR, HELPERS],
                        "sources": [source_record(root, s) for s in dict.fromkeys([*sources, GENERATOR, HELPERS])]})

    def figure(experiment, name, fig, sources, recipe):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160, facecolor="white", metadata={"Software": "TrustPCB RQ3 reporting"})
        plt.close(fig)
        emit(experiment, f"{experiment}/figures/{name}.png", buffer.getvalue(), "newly_rendered_visualization", sources, recipe)

    for experiment, source, target in copies:
        emit(experiment, target, inside(root, source).read_bytes(), "reporting_copy", [source],
             "Byte-identical copy; original evidence, metadata, identity and schema retained.")
    for experiment, spec in SPECS.items():
        base = f"runs/rq3/{experiment}"
        title = spec["title"]
        if experiment == "computational_efficiency":
            summary = json.loads(inside(root, base + "/summary.json").read_text(encoding="utf-8"))
            figure(experiment, "saved_runtime", plot_runtime(plt, summary, read_rows(root, base + "/repetition_timings.csv")),
                   [base + "/summary.json", base + "/repetition_timings.csv"],
                   "Render saved means/sample SD and saved repeat timings, without recomputing statistics or benchmarking.")
            continue
        interval_source = base + "/bootstrap_intervals.csv"
        intervals = interval_index(read_rows(root, interval_source))
        referral_source = base + "/crop_referral_metrics.csv"
        referral = read_rows(root, referral_source)
        figure(experiment, "achieved_referral", plot_referral(plt, referral, intervals, title), [referral_source, interval_source],
               "Display saved achieved workload versus saved error capture for each nominal development-budget group; marginal interval endpoints unchanged.")
        if experiment != EXTENSION:
            curve_source, aurc_source = base + "/prediction_risk_coverage.csv", base + "/aurc.csv"
            figure(experiment, "prediction_risk_coverage", plot_curve(plt, read_rows(root, curve_source), title), [curve_source],
                   "Plot all saved boundaries as right-endpoint steps (where=pre); zero-coverage risk remains undefined; no new curve or AURC calculation.")
            figure(experiment, "saved_aurc", plot_aurc(plt, read_rows(root, aurc_source), intervals, title), [aurc_source, interval_source],
                   "Render stored AURC and exact stored 95% interval endpoints; no integration or resampling.")
        if experiment in (EXTENSION, "final_evaluation"):
            figure(experiment, "remaining_automatic_error", plot_automatic(plt, referral, intervals, title), [referral_source, interval_source],
                   "Render saved remaining automatic-error rates and intervals by nominal development-budget category; offsets are display-only.")
            paired_source = base + "/paired_bootstrap_differences.csv"
            paired = read_rows(root, paired_source)
            for metric in ("aurc", "error_capture", "remaining_automatic_error_rate"):
                figure(experiment, "paired_" + metric, plot_paired(plt, paired, metric, title), [paired_source],
                       "Display saved observed method-A-minus-method-B differences and saved paired interval endpoints; no subtraction of CI bounds, bootstrap or hypothesis test.")

    for experiment, spec in SPECS.items():
        # Summary is one level deeper for the historical extension.
        back = "../" * (3 + len(experiment.split("/")))
        index = "../" * (1 + len(experiment.split("/")))
        text = f"""# {spec['title']}

Scientific experiment identity: `{spec['id']}`.

{spec['scope']}

- [Tables](../tables/): small, byte-identical saved CSV copies.
- [Figures](../figures/): newly rendered visualizations from saved evidence.
- JSON files beside this guide retain historical execution metadata, not reporting-run metadata.
- [Original evidence]({back}runs/rq3/{experiment}/)
- [Canonical scientific module]({back}src/trustpcb/rq3/{spec['module']}.py)
- [Tests]({back}tests/rq3/test_rq3_{spec['module']}.py)
- [Scientific protocol]({back}docs/rq3/rq3_{spec['module']}.md)
- [Reporting guide and gaps]({index}README.md) · [manifest]({index}manifest.json)

Intervals are saved intervals, not newly estimated uncertainty. Prediction-level
AURC and crop-level referral are different outcomes; an AURC advantage is not a
claim of improved crop referral. Paired difference signs and achieved workloads
are preserved without claims of universal improvement, factory readiness or real-time operation.
"""
        if experiment in (DEVELOPMENT, "final_evaluation"):
            text += f"\nFull [risk–coverage boundaries]({back}runs/rq3/{experiment}/prediction_risk_coverage.csv) and [crop risks]({back}runs/rq3/{experiment}/crop_risks.csv) remain at their original paths, hashed in the manifest instead of duplicated.\n"
        if experiment == DEVELOPMENT:
            text += "\nThe historical [reporting extension](../reporting_extension/summaries/README.md) remains separate.\n"
        emit(experiment, f"{experiment}/summaries/README.md", text.encode("utf-8"), "authored_reporting_summary",
             [GUIDE, f"runs/rq3/{experiment}/provenance.json"], "Scope, provenance and navigation; no new scientific conclusions.")

    rows = "\n".join(f"| [{s['title']}]({e}/summaries/README.md) | [tables]({e}/tables/) · [figures]({e}/figures/) · [summaries]({e}/summaries/) | `{s['id']}` |" for e, s in SPECS.items())
    text = f"""# RQ3 research reporting

RQ3 investigates selective crop review, prediction risk–coverage and the
computational cost of the frozen TrustPCB pipeline.

| Experiment | Reporting materials | Original scientific identity |
|---|---|---|
{rows}

Each experiment summary links the canonical Python workflow, relocated tests,
protocol and original evidence. The three scientific modules and compatibility
shims remain unchanged. Original development and supplementary reporting-extension
schemas stay separate; final evaluation retains the frozen development thresholds.

## How to read these artifacts

- **Original evidence:** `runs/rq3/`, unchanged. Large curve/crop-risk tables stay
  there and are referenced with hashes instead of duplicated.
- **Reporting copies:** small CSV/JSON files here are byte-identical; original
  provenance, thresholds, scientific identities and execution metadata remain intact.
- **Newly rendered visualizations:** figures are new presentations of saved data,
  not original DICC figures. No inference, calibration, ranking, metric calculation,
  threshold selection, bootstrap or benchmark was performed to create them.
- **Development versus test:** final-test crops were held out from RQ2 training
  and development, but previously used for RQ1 validation. Development plots are
  conditional development diagnostics, not independent generalisation evidence.
- Referral plots use **achieved workloads** on their horizontal axes. Nominal
  development budgets identify frozen operating points; they are not enforced
  test workloads. The automatic-error figure uses nominal budget categories only.
- Whole-score-group risk–coverage boundaries are rendered as right-endpoint steps.
  Undefined zero-coverage risk is not replaced with zero. Saved AURC is not recalculated.
- Paired differences retain weighted-method-minus-reference direction. Negative
  differences favour weighted for AURC/remaining automatic error; positive favour
  weighted for capture. Prediction-level results do not imply crop-level gains.
- Runtime figures show the saved DICC protocol, including its exclusions. They
  do not establish real-time performance or factory readiness. Saved sample SD
  across repetitions is distinguished from bootstrap confidence intervals.

## Missing/external inputs

The local RQ3 reporting inputs listed in the manifest are present. Full upstream
RQ2 detector/checkpoints, raw images/labels, development confidence/calibration/
stability artifacts and detailed final prediction/transform-match files remain
absent locally. Some downstream RQ2 summaries/weights exist; they do not replace
that missing chain. External/DICC locations must be confirmed and identities
verified when retrieving original artifacts. Absence locally does not mean they
were never generated. See [artifact inventory](../../docs/artifact_inventory.md)
and [RQ2 availability](../rq2/README.md). No missing evidence was regenerated.

## Reproduction and traceability

[reporting.py](../../src/trustpcb/rq3/reporting.py) is new presentation code. It
reuses only pure [RQ2 reporting helpers](../../src/trustpcb/rq2/reporting.py),
whose hash is recorded alongside its own. It does not import scientific workflows.
With `PYTHONPATH=src` and the existing Matplotlib environment:

```text
python -B -m trustpcb.rq3.reporting
python -B -m trustpcb.rq3.reporting --verify-only
```

The [manifest](manifest.json) records original source paths/SHA-256, reporting
paths/SHA-256, experiment identities, availability, derivation scripts/recipes and
rendering versions. Copies, visualizations and authored navigation are classified
separately. Its base commit is the pre-batch reference, not an experiment execution
commit. Hashes cover exact local bytes/line endings and do not replace historical
scientific LF-canonical hashes. The manifest is the inventory and does not hash itself.
Hash verification certifies these reporting inputs/outputs, not unavailable upstream
evidence. Existing edited/unlisted reports or changed inputs block rebuilding.

[Reporting protocol and CPU test commands](../../docs/rq3/reporting.md) ·
[RQ1 reporting](../rq1/README.md) · [RQ2 reporting](../rq2/README.md)
"""
    emit(None, "README.md", text.encode("utf-8"), "authored_reporting_summary", [GUIDE, "docs/artifact_inventory.md"], "RQ3 navigation and limits; no new results.")
    references = []
    for path in REFERENCES:
        experiment = path.split("/")[2]
        references.append({**source_record(root, path), "availability": "local_original_only",
                           "experiment_identity": SPECS[experiment]["id"], "reason": "Avoid redundant full curve/crop-risk copies"})
    manifest = {"schema_version": 1, "base_commit": BASE_COMMIT,
                "hash_convention": "SHA-256 of exact local bytes including line endings",
                "rendering": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "backend": "Agg", "dpi": 160},
                "artifacts": records, "references": references,
                "unavailable_local": [{"path": p, "availability": "absent locally; external location unverified"} for p in UNAVAILABLE]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    count = verify(root) if args.verify_only else build(root)
    print(f"Verified {count} RQ3 reporting artifacts.")


if __name__ == "__main__":
    main()
