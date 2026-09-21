"""DICC-only final Beta fit on frozen development predictions; no model selection."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform

from trustpcb import rq1, rq2_calibrator_comparison as comparison
from trustpcb.dataset_config import find_project_root

OUTPUT = "runs/rq2/final_calibrator"
FORM = "sigmoid(a * log(p) - b * log(1 - p) + c)"
TARGET = "same-class one-to-one IoU >= 0.50"
DIAGNOSTIC = "in-sample fit diagnostics; not independent evidence of generalisation"


def load_inputs(root, *, approve_beta_selection=False):
    """Validate the completed comparison without rerunning any comparison logic."""
    if not approve_beta_selection:
        raise ValueError("Reviewed Beta selection required: --approve-beta-selection")
    root = Path(root)
    images, rows, _, hashes = comparison.load_inputs(root)
    folder = root / comparison.OUTPUT
    provenance_path = folder / "provenance.json"
    provenance = rq1._read_json(provenance_path)
    expected = {"status": "complete", "final_fit_performed": False,
                "seed": comparison.SEED, "folds": 5, "resamples": 10000,
                "inclusion": .01, "primary_iou": .50, "log_epsilon": 1e-15}
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("Expected completed frozen comparison with no final fit")
    if provenance.get("final_fit_performed") is not False:
        raise ValueError("Comparison final_fit_performed must be false")
    if provenance.get("input_sha256") != hashes:
        raise ValueError("Comparison input hashes differ from final-fit population")
    for name in ("selection_summary.json", "method_metrics.csv"):
        path = folder / name
        digest = rq1._sha(path)
        if digest != provenance.get("output_sha256", {}).get(name):
            raise ValueError(f"Comparison artifact hash mismatch: {name}")
        hashes[f"{comparison.OUTPUT}/{name}"] = digest
    selection = rq1._read_json(folder / "selection_summary.json")
    if (selection.get("selected_method") != "Beta"
            or selection.get("status") != "comparison_complete_pending_review"
            or selection.get("final_fit_performed") is not False):
        raise ValueError("Expected completed Beta selection without a prior final fit")
    hashes[f"{comparison.OUTPUT}/provenance.json"] = rq1._sha(provenance_path)
    analysis_folder = root / comparison.analysis.OUTPUT
    analysis_provenance = rq1._read_json(analysis_folder / "provenance.json")
    summary = analysis_folder / "summary.json"
    summary_hash = rq1._sha(summary)
    if summary_hash != analysis_provenance.get("output_sha256", {}).get(summary.name):
        raise ValueError("Calibration-analysis summary hash mismatch")
    hashes[f"{comparison.analysis.OUTPUT}/summary.json"] = summary_hash
    # Preserve original identifying columns/row order; IoU75 is copied opaquely,
    # never interpreted or supplied to the optimiser or diagnostic metrics.
    with (analysis_folder / "labelled_predictions.csv").open(encoding="utf-8", newline="") as file:
        originals = list(csv.DictReader(file))
    if len(originals) != len(rows):
        raise ValueError("Source row inventory changed during validation")
    for original, row in zip(originals, rows):
        if f"{original['image']}#{int(original['source_row'])}" != row["prediction_id"]:
            raise ValueError("Source prediction order changed during validation")
    if any(rq1._sha(root / name) != digest for name, digest in hashes.items()):
        raise ValueError("Inputs changed during validation")
    return images, rows, originals, hashes, selection, provenance


def run(root, *, approve_beta_selection=False):
    root = Path(root).resolve()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(f"Existing final calibrator directory requires explicit archival: {output}")
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution inputs before DICC final fitting")
    images, rows, originals, hashes, selection, source_provenance = load_inputs(
        root, approve_beta_selection=approve_beta_selection)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy")}
    timestamp = datetime.now(timezone.utc).isoformat()
    provenance = {"experiment": "rq2_final_calibrator", "status": "running", **git,
                  "input_sha256": hashes, "versions": versions, "python": platform.python_version(),
                  "started_utc": timestamp, "method": "Beta", "beta_selection_reviewed": True,
                  "review_confirmation": "explicit --approve-beta-selection",
                  "diagnostic_scope": DIAGNOSTIC}
    output.mkdir(parents=True, exist_ok=False)
    target = output / "provenance.json"
    rq1._write_json(target, provenance, exclusive=True)
    try:
        confidence = [row["raw_confidence"] for row in rows]
        labels = [row["correct_iou50"] for row in rows]
        model = comparison.fit_calibrator("Beta", confidence, labels)
        calibrated = comparison.apply_calibrator(model, confidence)
        raw_metrics = comparison.metric_summary(confidence, labels)
        fitted_metrics = comparison.metric_summary(calibrated, labels)
        a, b, c = model["parameters"]
        optimizer = {"implementation": "trustpcb.rq2_calibrator_comparison.fit_calibrator",
                     "algorithm": "L-BFGS-B", "objective": "unregularised mean binary NLL",
                     "initial_parameters": [1., 1., 0.], "bounds": [[0., None], [0., None], [None, None]],
                     "maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50,
                     "converged": True, "iterations": model["iterations"],
                     "message": model["optimizer_message"], "randomness": "none"}
        artifact = {"method": "Beta", "mathematical_form": FORM, "a": a, "b": b, "c": c,
                    "epsilon": comparison.EPSILON, "monotonicity_constraints": "a >= 0; b >= 0; c unrestricted",
                    "inclusion_threshold": .01, "correctness_target": TARGET,
                    "development_images": len(images), "fitted_predictions": len(rows),
                    "source_prediction_sha256": hashes[f"{comparison.analysis.OUTPUT}/labelled_predictions.csv"],
                    "source_comparison": {"path": comparison.OUTPUT, "selection": selection,
                                          "git_commit": source_provenance.get("git_commit"),
                                          "provenance_sha256": hashes[f"{comparison.OUTPUT}/provenance.json"]},
                    "input_sha256": hashes, "git_commit": git["git_commit"], "versions": versions,
                    "optimizer": optimizer, "timestamp_utc": timestamp, "beta_selection_reviewed": True}
        rq1._write_json(output / "final_calibrator.json", artifact, exclusive=True)
        exported = [{**original, "prediction_id": row["prediction_id"],
                     "raw_confidence": row["raw_confidence"], "calibrated_confidence": float(value),
                     "correct_iou50": row["correct_iou50"]}
                    for original, row, value in zip(originals, rows, calibrated)]
        comparison.write_csv(output / "calibrated_development_predictions.csv", exported)
        rq1._write_json(output / "fit_summary.json", {
            "diagnostic_scope": DIAGNOSTIC, "method_selection_evidence": comparison.OUTPUT,
            "a": a, "b": b, "c": c, "optimizer": optimizer,
            "fitted_predictions": len(rows), "development_images": len(images),
            "raw": raw_metrics, "fitted_beta": fitted_metrics,
            "ece_definition": "10 equal-width bins [0,.1), ... [.9,1]; prediction-weighted absolute gap",
        }, exclusive=True)
        if (load_inputs(root, approve_beta_selection=True)[3] != hashes
                or rq1.git_provenance(root) != git):
            raise RuntimeError("Inputs/source changed during final fitting")
        provenance.update(status="complete", final_fit_performed=True,
                          output_sha256={p.name: rq1._sha(p) for p in output.iterdir() if p != target},
                          completed_utc=datetime.now(timezone.utc).isoformat())
        rq1._write_json(target, provenance)
        return output
    except BaseException:
        provenance["status"] = "incomplete"
        rq1._write_json(target, provenance)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicc", action="store_true")
    parser.add_argument("--approve-beta-selection", action="store_true",
                        help="Confirm review and acceptance of the completed Beta selection")
    args = parser.parse_args(argv)
    if os.name == "nt" or not args.dicc:
        raise RuntimeError("Real final fitting is DICC-only; --dicc required")
    print(run(find_project_root(), approve_beta_selection=args.approve_beta_selection))


if __name__ == "__main__":
    main()
