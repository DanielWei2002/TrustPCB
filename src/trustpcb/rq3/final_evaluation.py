"""Final RQ3 text-artifact evaluation with unchanged development thresholds."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import platform

import numpy as np

from trustpcb import rq1
from trustpcb.rq3 import selective_review as review
from trustpcb.rq2 import weighted_final_evaluation as frozen
from trustpcb.dataset_config import find_project_root

OUTPUT = "runs/rq3/final_evaluation"
DEVELOPMENT = "runs/rq3/development_selective_review"
THRESHOLDS = f"{DEVELOPMENT}/development_thresholds.json"
ROLE = "held out from RQ2 training and development, previously used for RQ1 validation."
IMAGE_COUNT, PREDICTION_COUNT = 2052, 17840
SEED, REPLICATES = 24209199, 10000
FLAGS = {"thresholds_rederived": False, "fitting_performed": False,
         "calibration_performed": False, "weight_selection_performed": False,
         "inference_performed": False, "transformation_processing_performed": False,
         "computational_benchmark_performed": False}


def load_thresholds(root):
    """Read/validate original development evidence, never derive a threshold."""
    root = Path(root)
    source_path = root/DEVELOPMENT/"provenance.json"
    source = rq1._read_json(source_path)
    if (source.get("status") != "complete" or source.get("development_images") != 821
            or source.get("prediction_count") != 7131 or source.get("weights") != [.89, .01, .10]
            or source.get("budgets_percent") != list(review.BUDGETS)):
        raise ValueError("Expected completed frozen RQ3 development provenance")
    path = root/THRESHOLDS
    digest = rq1._sha(path)
    if source.get("output_sha256", {}).get(path.name) != digest:
        raise ValueError("Frozen development threshold hash mismatch")
    saved = rq1._read_json(path)
    if (saved.get("comparison") != ">" or saved.get("test_retuning_permitted") is not False
            or saved.get("derived_from") != "development scores only"):
        raise ValueError("Expected frozen strict-greater development referral protocol")
    thresholds = saved.get("thresholds", {})
    if set(thresholds) != set(review.METHODS):
        raise ValueError("Frozen threshold methods differ")
    for method in review.METHODS:
        entries = thresholds[method]
        if not isinstance(entries, list) or [t.get("budget_percent") for t in entries] != list(review.BUDGETS):
            raise ValueError("Frozen threshold budgets differ")
        for t in entries:
            value = t.get("threshold")
            if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("Invalid frozen threshold value")
            nominal = 821*t["budget_percent"]//100
            achieved = t.get("achieved_crops")
            if (t.get("nominal_crops") != nominal or type(achieved) is not int or not 0 <= achieved <= nominal
                    or t.get("achieved_workload") != achieved/821):
                raise ValueError("Invalid recorded development workload")
    return thresholds, {THRESHOLDS: digest, f"{DEVELOPMENT}/provenance.json": rq1._sha(source_path)}, source


def validate_population(images, rows):
    # Existing validator also checks frozen fields, identities and confidence inclusion.
    return frozen.validate_population(images, rows)


def load_inputs(root):
    thresholds, threshold_hashes, source = load_thresholds(root)
    images, rows, hashes, identity = frozen.load_inputs(root)  # text artifacts only
    rows = validate_population(images, rows)
    if source.get("selected_weight_identity") != identity:
        raise ValueError("Development thresholds use a different frozen weight identity")
    return images, rows, {**hashes, **threshold_hashes}, identity, thresholds


def observed_tables(images, rows, scores, thresholds):
    """Apply fixed scalar values; never call review.tables()/threshold_for()."""
    _, labels, counts, errors, crops, plans = review.prepare(images, rows, scores)
    curves, areas, inventory, referrals = [], [], [], []
    for method in review.METHODS:
        records, area = review.curve(plans[method], labels, np.ones(len(rows)))
        curves.extend({"method": method, **r} for r in records)
        areas.append({"method": method, "aurc": area})
        inventory.extend({"method": method, "image": image, "prediction_count": int(counts[i]),
            "incorrect_predictions": int(errors[i]), "crop_risk": float(crops[method][i]) if counts[i] else None,
            "referral_eligible": bool(counts[i])} for i, image in enumerate(images))
        for t in thresholds[method]:
            referrals.append({"method": method, "budget_percent": t["budget_percent"],
                "threshold": t["threshold"], "development_nominal_crops": t["nominal_crops"],
                "development_achieved_crops": t["achieved_crops"],
                "development_achieved_workload": t["achieved_workload"],
                **review.referral_metrics(crops[method], counts, errors, t["threshold"], np.ones(len(images)))})
    return curves, areas, inventory, referrals


def run(root):
    root = Path(root).resolve()
    output = root/OUTPUT
    if output.exists():
        raise FileExistsError("Existing partial/completed RQ3 final evaluation; explicit archival required")
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution inputs before DICC final evaluation")
    images, rows, hashes, identity, thresholds = load_inputs(root)
    metadata = {"experiment": "rq3_final_evaluation", "status": "running", **git, **FLAGS,
        "test_role": ROLE, "test_crops": len(images), "prediction_count": len(rows),
        "input_sha256": hashes, "development_threshold_artifact": THRESHOLDS,
        "development_threshold_sha256": hashes[THRESHOLDS], "applied_thresholds": thresholds,
        "frozen_weight_identity": identity, "weights": [.89, .01, .10],
        "correctness_definition": "same-class one-to-one IoU >= 0.50",
        "bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES,
        "definitions": {**review.DEFINITIONS, "scope": "final evaluation of frozen development thresholds on 2052 test crops",
                        "threshold": "load exact saved development values; never derive from test data"},
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
        "python": platform.python_version(), "started_utc": datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True, exist_ok=False)
    target = output/"provenance.json"
    rq1._write_json(target, metadata, exclusive=True)
    try:
        scores = review.scores_for(rows)  # reconstruct 89/1/10; ignore legacy trustpcb_risk
        curves, areas, inventory, referrals = observed_tables(images, rows, scores, thresholds)
        intervals, bootstrap_info = review.bootstrap(images, rows, scores, thresholds, replicates=REPLICATES, seed=SEED)
        bootstrap_info["sampling_unit"] = "whole test image/crop"
        writer = review.risk.final.comparison.write_csv
        for name, records in (("prediction_risk_coverage.csv", curves), ("aurc.csv", areas),
                ("crop_risks.csv", inventory), ("crop_referral_metrics.csv", referrals),
                ("bootstrap_intervals.csv", intervals),
                ("paired_bootstrap_differences.csv", bootstrap_info["paired_differences"])):
            writer(output/name, records)
        rq1._write_json(output/"bootstrap_summary.json", bootstrap_info, exclusive=True)
        rq1._write_json(output/"summary.json", {"test_role": ROLE, "test_crops": len(images),
            "prediction_count": len(rows), "empty_crops": len(images)-len({r["image"] for r in rows}),
            "definitions": metadata["definitions"], **FLAGS}, exclusive=True)
        if load_inputs(root)[2:] != (hashes, identity, thresholds) or rq1.git_provenance(root) != git:
            raise RuntimeError("Frozen inputs/source changed during final evaluation")
        metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
            output_sha256={p.name: rq1._sha(p) for p in output.iterdir() if p != target})
        rq1._write_json(target, metadata)
        return output
    except BaseException:
        metadata["status"] = "incomplete"
        rq1._write_json(target, metadata)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicc", action="store_true")
    args = parser.parse_args(argv)
    if os.name == "nt" or not args.dicc:
        raise RuntimeError("Real RQ3 final evaluation is DICC-only; --dicc required")
    print(run(find_project_root()))


if __name__ == "__main__":
    main()
