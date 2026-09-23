"""Proposal-specified weighted fusion: development-only selection, DICC execution."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path

import numpy as np

from trustpcb import rq1, rq2_risk_evaluation as risk
from trustpcb import rq2_calibrator_comparison as comparison, rq2_risk_sensitivity as sensitivity
from trustpcb.dataset_config import find_project_root

OUTPUT = "runs/rq2/weighted_fusion"
SEED, CANDIDATE_COUNT = 24209199, 5151
TOLERANCE = 1e-12
PURPOSE = "Proposal-specified weighted fusion; selection on frozen RQ2 development signals only"
WEIGHT_KEYS = ("conf_percent", "class_percent", "localisation_percent")
HIERARCHY = "maximum mean fold AUPRC; AUROC within absolute 1e-12 AUPRC tie set; exact minimum equal-weight distance; descending conf, localisation, class"


def validate_weights(weights):
    if len(weights) != 3 or any(type(w) is not int or w < 0 for w in weights) or sum(weights) != 100:
        raise ValueError("Weights must be nonnegative integer percentages summing exactly to 100")
    return weights


def grid():
    candidates = [(a, b, 100 - a - b) for a in range(101) for b in range(101 - a)]
    if len(candidates) != CANDIDATE_COUNT or len(set(candidates)) != CANDIDATE_COUNT:
        raise ValueError("Expected exactly 5151 unique simplex candidates")
    for weights in candidates:
        validate_weights(weights)
    return candidates


def component_matrix(rows):
    values = np.array([[r[key] for key in risk.SIGNALS] for r in rows], dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid frozen reliability signals")
    return 1 - values


def weighted_risk(components, weights):
    validate_weights(weights)
    # Fixed term order, integer enumeration; floating arithmetic only for scores.
    return (components[:, 0] * weights[0] + components[:, 1] * weights[1] + components[:, 2] * weights[2]) / 100


def load_inputs(root):
    images, rows, hashes = risk.load_inputs(root)
    linked_images, _, edges, link_hashes = comparison.load_inputs(root)
    if linked_images != images or any(hashes.get(k) != v for k, v in link_hashes.items()):
        raise ValueError("Similarity inventory differs from frozen development artifacts")
    return images, rows, edges, hashes


def evaluate_candidates(images, rows, folds, candidates):
    """Pure IoU50 selection inputs. No IoU75 field is read by this function."""
    if [f["image"] for f in folds] != images or len(set(images)) != len(images):
        raise ValueError("Fold manifest must cover the image inventory exactly in order")
    lookup = {f["image"]: f["fold_id"] for f in folds}
    if set(lookup.values()) != set(range(5)) or any(r["image"] not in lookup for r in rows):
        raise ValueError("Expected five image folds and development-only predictions")
    fold_ids = np.array([lookup[r["image"]] for r in rows])
    labels = np.array([1 - r["correct_iou50"] for r in rows])
    masks = [fold_ids == fold for fold in range(5)]
    if any(set(labels[mask]) != {0, 1} for mask in masks):
        raise ValueError("Each frozen fold requires both IoU50 target classes; no fold dropping or reassignment")
    components = component_matrix(rows)
    records = []
    for weights in candidates:
        scores = weighted_risk(components, weights)
        metrics = [risk.metrics(scores[mask], labels[mask]) for mask in masks]
        record = dict(zip(WEIGHT_KEYS, weights))
        record.update({f"fold_{i}_{metric}": m[metric] for i, m in enumerate(metrics) for metric in ("auprc", "auroc")})
        record.update(mean_fold_auprc=math.fsum(m["auprc"] for m in metrics) / 5,
                      mean_fold_auroc=math.fsum(m["auroc"] for m in metrics) / 5,
                      equal_weight_distance=math.sqrt(sum((3*w - 100)**2 for w in weights)) / 300)
        records.append(record)
    return records


def select(records):
    if not records:
        raise ValueError("No candidate evaluations")
    for row in records:
        validate_weights(tuple(row[k] for k in WEIGHT_KEYS))
        if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in ("mean_fold_auprc", "mean_fold_auroc")):
            raise ValueError("Undefined/invalid candidate selection metric")
    best_pr = max(r["mean_fold_auprc"] for r in records)
    tied = [r for r in records if best_pr - r["mean_fold_auprc"] <= TOLERANCE]
    best_roc = max(r["mean_fold_auroc"] for r in tied)
    tied = [r for r in tied if best_roc - r["mean_fold_auroc"] <= TOLERANCE]
    # Squared distance numerator is integer-exact, avoiding false geometric ties.
    distance = lambda r: sum((3*r[k] - 100)**2 for k in WEIGHT_KEYS)
    nearest = min(map(distance, tied))
    tied = [r for r in tied if distance(r) == nearest]
    winner = max(tied, key=lambda r: (r["conf_percent"], r["localisation_percent"], r["class_percent"]))
    weights = tuple(winner[k] for k in WEIGHT_KEYS)
    return {"integer_percentages": dict(zip(WEIGHT_KEYS, weights)),
            "weights": dict(zip(("w_conf", "w_class", "w_localisation"), [w / 100 for w in weights])),
            "mean_fold_auprc": winner["mean_fold_auprc"], "mean_fold_auroc": winner["mean_fold_auroc"],
            "equal_weight_distance": math.sqrt(nearest) / 300, "tie_absolute_tolerance": TOLERANCE,
            "tie_relative_tolerance": 0, "distance_tie_tolerance": 0, "selection_hierarchy": HIERARCHY,
            "selection_target": "incorrect_iou50 = 1 - correct_iou50", "iou75_used_for_selection": False}


def report_metrics(rows, selected):
    weights = tuple(selected["integer_percentages"][k] for k in WEIGHT_KEYS)
    components = component_matrix(rows)
    unchanged = risk.risk_scores(rows)
    scores = {"calibrated_confidence_risk": unchanged["calibrated_confidence_risk"],
              "equal_weight_trustpcb_risk": unchanged["trustpcb_risk"],
              "selected_weighted_trustpcb_risk": weighted_risk(components, weights)}
    primary_y = [1 - r["correct_iou50"] for r in rows]
    # Only reached after selection. No secondary label enters candidate evaluation.
    secondary_y = [r["incorrect_iou75"] for r in sensitivity.sensitivity_rows(rows)]
    reports = []
    for y, label in ((primary_y, "Development IoU>=0.50 reporting after weight selection"),
                     (secondary_y, sensitivity.LABEL)):
        reports.append([{"analysis_label": label, "method": method, **risk.metrics(values, y),
                         "prediction_count": len(rows), "incorrect": sum(y), "incorrect_prevalence": sum(y) / len(y)}
                        for method, values in scores.items()])
    return reports


def run(root):
    root = Path(root).resolve()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError("Existing complete/partial weighted fusion; archive explicitly before retry")
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution inputs before DICC weighted selection")
    images, rows, edges, hashes = load_inputs(root)
    candidates = grid()
    folds = comparison.assign_folds(images, edges, seed=SEED, n_folds=5)
    provenance = {"experiment": "rq2_weighted_fusion", "purpose": PURPOSE, "status": "running", **git,
        "input_sha256": hashes, "development_images": len(images), "prediction_count": len(rows),
        "correctness_definition": risk.final.TARGET, "fold_count": 5, "seed": SEED,
        "fold_design": "image-level, connected similarity groups indivisible; image-count allocation independent of labels",
        "grid_resolution": .01, "expected_candidate_count": CANDIDATE_COUNT, "actual_candidate_count": len(candidates),
        "selection_metric": "mean fold AUPRC (non-interpolated average precision)", "secondary_metric": "mean fold AUROC",
        "selection_hierarchy": HIERARCHY, "tie_absolute_tolerance": TOLERANCE,
        "risk_formula": "w_conf*(1-calibrated_confidence)+w_class*(1-class_consistency)+w_localisation*(1-localisation_stability)",
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
        "started_utc": datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True, exist_ok=False)
    target = output / "provenance.json"
    rq1._write_json(target, provenance, exclusive=True)
    try:
        comparison.write_csv(output / "fold_assignments.csv", folds)
        # Pass a primary-only projection to make secondary-target leakage impossible.
        selection_rows = [{key: r[key] for key in ("image", "correct_iou50", *risk.SIGNALS)} for r in rows]
        records = evaluate_candidates(images, selection_rows, folds, candidates)
        if len(records) != CANDIDATE_COUNT or len({tuple(r[k] for k in WEIGHT_KEYS) for r in records}) != CANDIDATE_COUNT:
            raise ValueError("Expected all 5151 unique candidate evaluations")
        comparison.write_csv(output / "candidate_weights.csv", records)
        selected = select(records)
        rq1._write_json(output / "selected_weights.json", selected, exclusive=True)
        primary, secondary = report_metrics(rows, selected)
        comparison.write_csv(output / "development_metrics.csv", primary)
        comparison.write_csv(output / "sensitivity_iou75_metrics.csv", secondary)
        rq1._write_json(output / "summary.json", {"purpose": PURPOSE, "selected_weights": selected,
            "development_images": len(images), "prediction_count": len(rows), "candidate_count": len(records),
            "fold_image_counts": {str(i): sum(f["fold_id"] == i for f in folds) for i in range(5)},
            "scope": "development selection and descriptive reporting, not independent final-test evidence",
            "equal_weight_results_replaced": False, "threshold_selected": False}, exclusive=True)
        if load_inputs(root)[3] != hashes or rq1.git_provenance(root) != git:
            raise RuntimeError("Development inputs/source changed during weight selection")
        provenance.update(status="complete", selected_weights=selected,
            completed_utc=datetime.now(timezone.utc).isoformat(),
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
        raise RuntimeError("Real development weight selection is DICC-only; --dicc required")
    print(run(find_project_root()))


if __name__ == "__main__":
    main()
