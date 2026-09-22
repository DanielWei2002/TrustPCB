"""Secondary IoU>=0.75 sensitivity analysis; primary results remain unchanged."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform

import numpy as np

from trustpcb import rq1, rq2_risk_evaluation as primary
from trustpcb.dataset_config import find_project_root

OUTPUT = "runs/rq2/risk_sensitivity_iou75"
LABEL = "Secondary IoU>=0.75 sensitivity analysis"
TARGET = "incorrect_iou75 = 1 - frozen IoU75 correctness"
CORRECTNESS = "same-class one-to-one IoU >= 0.75; existing secondary labels only"
SEED, REPLICATES = 24209199, 10000
DEFINITIONS = {**primary.DEFINITIONS, "target": TARGET}


def sensitivity_rows(rows):
    """Copy rows and adapt the shared metric target; never overwrite IoU50 labels."""
    adapted = []
    for row in rows:
        fields = [name for name in ("correct_at_iou75", "correct_iou75") if name in row]
        if not fields:
            raise ValueError("Missing frozen IoU75 correctness field")
        try:
            labels = [float(row[name]) for name in fields]
        except (ValueError, TypeError) as exc:
            raise ValueError("Missing/invalid IoU75 correctness") from exc
        if any(value not in (0, 1) for value in labels) or len(set(labels)) != 1:
            raise ValueError("IoU75 labels must be binary and aliases must agree")
        incorrect = 1 - int(labels[0])
        adapted.append({**row, "incorrect_iou75": incorrect, "incorrect_prediction": incorrect})
    return adapted


def load_inputs(root):
    images, rows, hashes = primary.load_inputs(root)
    return images, sensitivity_rows(rows), hashes


def run(root):
    root = Path(root).resolve()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError("Existing complete/incomplete sensitivity output; archive explicitly before retry")
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution inputs before DICC sensitivity analysis")
    images, rows, hashes = load_inputs(root)
    provenance = {"experiment": "rq2_risk_sensitivity_iou75", "analysis_label": LABEL,
                  "status": "running", **git, "input_path": primary.INPUT, "input_sha256": hashes,
                  "development_images": len(images), "prediction_count": len(rows),
                  "correctness_definition": CORRECTNESS, "target_definition": TARGET,
                  "primary_correctness_iou": .50, "sensitivity_correctness_iou": .75,
                  "bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES,
                  "metric_definitions": DEFINITIONS, "risk_formulas": primary.FORMULAS,
                  "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
                  "python": platform.python_version(), "started_utc": datetime.now(timezone.utc).isoformat(),
                  "scope": "secondary robustness only; no replacement of primary results or model/weight/threshold selection"}
    output.mkdir(parents=True, exist_ok=False)
    target = output / "provenance.json"
    rq1._write_json(target, provenance, exclusive=True)
    try:
        scores = primary.risk_scores(rows)
        samples, bootstrap_info = primary.bootstrap(images, rows, scores, replicates=REPLICATES, seed=SEED)
        methods, differences = primary.evaluation_tables(rows, scores, samples)
        for name, records in (("method_metrics.csv", methods), ("pairwise_auroc_differences.csv", differences)):
            primary.final.comparison.write_csv(output / name, [{"analysis_label": LABEL, **r} for r in records])
        with (output / "bootstrap_metrics.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("analysis_label", "replicate", "method", "auroc", "auprc", "auroc_valid", "auprc_valid"))
            writer.writeheader()
            for i in range(len(samples)):
                for j, method in enumerate(primary.METHODS):
                    writer.writerow({"analysis_label": LABEL, "replicate": i, "method": method,
                        "auroc": primary.nullable(samples[i, j, 0]), "auprc": primary.nullable(samples[i, j, 1]),
                        "auroc_valid": bool(np.isfinite(samples[i, j, 0])), "auprc_valid": bool(np.isfinite(samples[i, j, 1]))})
        bootstrap_info.update(analysis_label=LABEL, target_definition=TARGET,
            validity_by_method=[{"method": r["method"], **{k: v for k, v in r.items() if k.endswith("replicates")}} for r in methods])
        rq1._write_json(output / "bootstrap_summary.json", bootstrap_info, exclusive=True)
        incorrect = sum(r["incorrect_iou75"] for r in rows)
        rq1._write_json(output / "summary.json", {"analysis_label": LABEL, "target_definition": TARGET,
            "development_images": len(images), "prediction_count": len(rows),
            "correct": len(rows) - incorrect, "incorrect": incorrect, "incorrect_prevalence": incorrect / len(rows),
            "auprc_reference_level": incorrect / len(rows), "metric_definitions": DEFINITIONS,
            "interpretation": "Secondary robustness analysis only. IoU>=0.50 remains primary; never choose whichever IoU result looks better.",
            "threshold_selected": False, "weights_tuned": False, "calibrator_refitted": False}, exclusive=True)
        if load_inputs(root)[2] != hashes or rq1.git_provenance(root) != git:
            raise RuntimeError("Inputs/source changed during sensitivity analysis")
        provenance.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                          output_sha256={p.name: rq1._sha(p) for p in output.iterdir() if p != target})
        rq1._write_json(target, provenance)
        return output
    except BaseException:
        provenance["status"] = "incomplete"
        rq1._write_json(target, provenance)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicc", action="store_true")
    args = parser.parse_args(argv)
    if os.name == "nt" or not args.dicc:
        raise RuntimeError("Real sensitivity analysis is DICC-only; --dicc required")
    print(run(find_project_root()))


if __name__ == "__main__":
    main()
