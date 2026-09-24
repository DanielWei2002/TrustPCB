"""Apply development-frozen weights to existing final-test text artifacts only."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from trustpcb import rq1, rq2_risk_evaluation as risk, rq2_weighted_fusion as fusion
from trustpcb import rq2_final_evaluation as final, rq2_risk_sensitivity as sensitivity
from trustpcb.dataset_config import find_project_root
from trustpcb.rq2_artifact_paths import resolve_input

OUTPUT = "runs/rq2/weighted_final_evaluation"
SELECTED = "runs/rq2/weighted_fusion/selected_weights.json"
EVIDENCE_COMMIT = "74510277056c72114dc421bd900ddb5dfbc91933"
WEIGHTS = (89, 1, 10)
PREDICTION_COUNT = 17840
SEED, REPLICATES = 24209199, 10000
METHODS = ("calibrated_confidence_risk", "equal_weight_trustpcb_risk", "weighted_trustpcb_risk")
FORMULA = "0.89*(1-calibrated_confidence)+0.01*(1-class_consistency)+0.10*(1-localisation_stability)"
NO_ACTIONS = {"weight_selection_occurred": False, "calibration_fitting_occurred": False,
              "inference_occurred": False, "transformation_processing_occurred": False,
              "referral_threshold_selected": False}


def validate_weights(record):
    if (record.get("weights") != dict(zip(("w_conf", "w_class", "w_localisation"), (.89, .01, .10)))
            or record.get("integer_percentages") != dict(zip(fusion.WEIGHT_KEYS, WEIGHTS))
            or record.get("iou75_used_for_selection") is not False):
        raise ValueError("Selected weights must be the frozen 0.89 / 0.01 / 0.10 with IoU75 excluded")


def selected_identity(root):
    path = resolve_input(root, SELECTED)
    working = path.read_bytes()
    try:
        committed = subprocess.run(["git", "show", f"{EVIDENCE_COMMIT}:{SELECTED}"],
                                   cwd=root, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        raise ValueError("Frozen weight evidence commit/file unavailable; obtain the evidence history before execution") from exc
    if working.replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
        raise ValueError("selected_weights.json differs from the frozen evidence commit")
    validate_weights(json.loads(working))
    return {"path": SELECTED, "sha256": hashlib.sha256(working).hexdigest(),
            "evidence_commit": EVIDENCE_COMMIT, "committed_blob_sha256": hashlib.sha256(committed).hexdigest()}


def validate_population(images, rows):
    if len(images) != 2052 or len(set(images)) != 2052 or len(rows) != PREDICTION_COUNT:
        raise ValueError("Expected exactly 2052 final-test images and 17840 predictions")
    allowed, seen, parsed = set(images), set(), []
    for row in rows:
        identity = row.get("prediction_id")
        if not identity or identity in seen or row.get("image") not in allowed:
            raise ValueError("Duplicate identity or non-final-test prediction")
        seen.add(identity)
        try:
            values = {key: float(row[key]) for key in ("raw_confidence", *risk.SIGNALS)}
            primary = float(row["correct_iou50"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Missing/invalid frozen score or primary label") from exc
        if (any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values())
                or values["raw_confidence"] < .01 or primary not in (0, 1)):
            raise ValueError("Invalid frozen signal/inclusion/correctness")
        # Validate aliases but never use the secondary target to choose anything.
        secondary = sensitivity.sensitivity_rows([row])[0]["incorrect_iou75"]
        parsed.append({**row, **values, "correct_iou50": int(primary),
                       "incorrect_prediction": 1 - int(primary), "incorrect_iou75": secondary})
    return parsed


def load_inputs(root):
    root = Path(root)
    identity = selected_identity(root)
    images = final.frozen_test_images(root)  # manifests only; no model/image/label files
    folder = resolve_input(root, final.OUTPUT)
    provenance = rq1._read_json(folder / "provenance.json")
    if (provenance.get("status") != "complete" or provenance.get("test_role") != final.ROLE
            or provenance.get("test_images") != 2052 or provenance.get("test_manifest") != final.MANIFEST
            or provenance.get("test_manifest_sha256_lf") != final.MANIFEST_SHA
            or provenance.get("reference_threshold") != .01
            or provenance.get("checkpoint_identity", {}).get("sha256") != final.raw.CHECKPOINT_SHA
            or tuple(provenance.get("beta_parameters", ())) != final.BETA):
        raise ValueError("Final-test provenance is incomplete or differs from frozen evaluation")
    hashes = {SELECTED: identity["sha256"], f"{final.OUTPUT}/provenance.json": rq1._sha(folder / "provenance.json")}
    for name in ("prediction_stability_scores.csv", "labelled_predictions.csv", "primary_iou50/summary.json"):
        path = folder / name
        digest = rq1._sha(path)
        if provenance.get("output_sha256", {}).get(name) != digest:
            raise ValueError(f"Frozen final-test artifact hash mismatch: {name}")
        hashes[f"{final.OUTPUT}/{name}"] = digest
    summary = rq1._read_json(folder / "primary_iou50/summary.json")
    if summary.get("test_images") != 2052 or summary.get("prediction_count") != PREDICTION_COUNT or summary.get("test_role") != final.ROLE:
        raise ValueError("Final-test summary population/role differs")
    rows = risk.read_csv(folder / "prediction_stability_scores.csv")
    labelled = risk.read_csv(folder / "labelled_predictions.csv")
    if len(rows) != len(labelled) or any(any(row.get(k) != v for k, v in label.items()) for row, label in zip(rows, labelled)):
        raise ValueError("Final reference fields/order differ from frozen labelled predictions")
    for relative in (final.MANIFEST, final.PARTITION_REPORT, *(s[0] for s in final.raw.rq2_detector_training.MANIFESTS.values())):
        hashes[relative] = rq1._sha(resolve_input(root, relative))
    return images, validate_population(images, rows), hashes, identity


def scores_for(rows):
    components = fusion.component_matrix(rows)
    equal = risk.risk_scores(rows)["trustpcb_risk"]
    return dict(zip(METHODS, (components[:, 0], equal, fusion.weighted_risk(components, WEIGHTS))))


def evaluate(images, rows, scores, *, secondary=False):
    target_rows = [{**row, "incorrect_prediction": row["incorrect_iou75"] if secondary else 1-row["correct_iou50"]} for row in rows]
    label = sensitivity.LABEL if secondary else "Primary IoU>=0.50 frozen weighted final-test evaluation"
    samples, info = risk.bootstrap(images, target_rows, scores, methods=METHODS, replicates=REPLICATES, seed=SEED)
    info.update(sampling_unit="final-test image", analysis_label=label)
    y = [r["incorrect_prediction"] for r in target_rows]
    observed = {method: risk.metrics(scores[method], y) for method in METHODS}
    metrics = []
    for j, method in enumerate(METHODS):
        record = {"analysis_label": label, "method": method, **observed[method]}
        for k, metric in enumerate(("auroc", "auprc")):
            record.update({f"{metric}_{key}": value for key, value in risk.interval(samples[:, j, k]).items()})
        metrics.append(record)
    pairs = []
    for reference in METHODS[:2]:
        for k, metric in enumerate(("auroc", "auprc")):
            a, b = observed[METHODS[2]][metric], observed[reference][metric]
            # Subtract within the existing paired draws; never resample for AUPRC.
            pairs.append({"analysis_label": label, "method_A": METHODS[2], "method_B": reference,
                "metric": metric, "observed_difference": a-b if a is not None and b is not None else None,
                **risk.interval(samples[:, 2, k] - samples[:, METHODS.index(reference), k])})
    info["validity_by_method"] = [{k: v for k, v in r.items() if k == "method" or k.endswith("replicates")} for r in metrics]
    info.update(correct=len(y)-sum(y), incorrect=sum(y), incorrect_prevalence=sum(y)/len(y))
    return metrics, pairs, info


def run(root):
    root = Path(root).resolve()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError("Existing completed/partial weighted final evaluation; explicit archival required")
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution inputs before DICC analysis")
    images, rows, hashes, identity = load_inputs(root)
    provenance = {"experiment": "rq2_weighted_final_evaluation", "status": "running", **git, **NO_ACTIONS,
        "weights": dict(zip(("w_conf", "w_class", "w_localisation"), (.89, .01, .10))),
        "selected_weight_identity": identity, "input_sha256": hashes, "test_images": len(images),
        "prediction_count": len(rows), "test_role": final.ROLE, "risk_formula": FORMULA,
        "primary_correctness": "same-class one-to-one IoU >= 0.50", "sensitivity_correctness": "same-class one-to-one IoU >= 0.75",
        "bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES, "metric_definitions": risk.DEFINITIONS,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
        "started_utc": datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True, exist_ok=False)
    target = output / "provenance.json"
    rq1._write_json(target, provenance, exclusive=True)
    try:
        scores = scores_for(rows)
        boot = {}
        for secondary, prefix in ((False, "primary"), (True, "sensitivity_iou75")):
            methods, pairs, boot[prefix] = evaluate(images, rows, scores, secondary=secondary)
            risk.final.comparison.write_csv(output / f"{prefix}_method_metrics.csv", methods)
            risk.final.comparison.write_csv(output / f"{prefix}_pairwise_differences.csv", pairs)
        if boot["primary"]["draw_indices_sha256"] != boot["sensitivity_iou75"]["draw_indices_sha256"]:
            raise RuntimeError("Primary/secondary bootstrap draws differ")
        rq1._write_json(output / "summary.json", {"test_role": final.ROLE, "test_images": len(images), "prediction_count": len(rows),
            "weights": provenance["weights"], "bootstrap": boot, **NO_ACTIONS,
            "interpretation": "Frozen development weights applied once; no test-driven revision or RQ3 selection."}, exclusive=True)
        after = load_inputs(root)
        if after[2:] != (hashes, identity) or rq1.git_provenance(root) != git:
            raise RuntimeError("Frozen inputs/source changed during evaluation")
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
        raise RuntimeError("Real final-test analysis is DICC-only; --dicc required")
    print(run(find_project_root()))


if __name__ == "__main__":
    main()
