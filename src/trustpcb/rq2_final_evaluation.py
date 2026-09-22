"""Frozen RQ2 final-test orchestration. DICC only; no fitting/selection operation."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform

import numpy as np

from trustpcb import rq1, rq2_confidence_distribution as raw
from trustpcb import rq2_calibration_analysis as matching, rq2_calibrator_comparison as calibration
from trustpcb import rq2_final_calibrator as final, rq2_transformation_stability as stability
from trustpcb import rq2_risk_evaluation as risk, rq2_risk_sensitivity as sensitivity
from trustpcb.dataset_config import find_project_root, load_paths
from trustpcb.rq2_artifact_paths import resolve_input

OUTPUT = "runs/rq2/final_evaluation"
ROLE = "held out from RQ2 training and development, previously used for RQ1 validation"
MANIFEST = "configs/datasets/similarity_aware_val_images.txt"
MANIFEST_SHA = "a2f246bf5be60cad5fac7f985ed91888bddb43601b73331f39d02e180075bc8e"
PARTITION_REPORT = "data/splits/rq2_train_development_split/verification_report.json"
REPORT_SHA = "a29cd91b4c7edf7d80c0b8c0b81b50082710835bb2a96a2f29e15c5afca5db9e"
IMAGE_COUNT = 2052
BETA = (0.9671379615387139, 2.2912177337533874, -0.6283462684200526)
ORIGINAL_SETTINGS = {**raw.SETTINGS, "conf": .01}
SEED, REPLICATES = 24209199, 10000


def validate_protocol():
    specs = stability.SPECS
    expected = [("brightness", .9), ("brightness", 1.1), ("contrast", .9), ("contrast", 1.1),
                ("blur", .6), ("rotation", -2), ("rotation", 2),
                ("translation", "x", -.02), ("translation", "x", .02),
                ("translation", "y", -.02), ("translation", "y", .02)]
    actual = []
    for spec in specs:
        family = spec["family"]
        if family == "translation":
            actual.append((family, spec["axis"], spec["fraction"]))
        else:
            key = {"brightness": "factor", "contrast": "factor", "blur": "sigma", "rotation": "degrees"}.get(family)
            actual.append((family, spec.get(key)))
    if actual != expected or len({s["id"] for s in specs}) != 11:
        raise ValueError("Frozen eleven transformation definitions differ")
    for settings, floor in ((ORIGINAL_SETTINGS, .01), (raw.SETTINGS, .001)):
        required = {"conf": floor, "iou": .7, "imgsz": 640, "max_det": 300, "device": 0,
                    "batch": 1, "rect": True, "augment": False, "agnostic_nms": False, "classes": None}
        if any(settings.get(k) != v for k, v in required.items()):
            raise ValueError("Frozen inference settings differ")


def lf_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def frozen_test_images(root):
    root = Path(root)
    report_path = resolve_input(root, PARTITION_REPORT)
    if lf_hash(report_path) != REPORT_SHA:
        raise ValueError("Frozen partition provenance changed")
    report = rq1._read_json(report_path)
    path = resolve_input(root, MANIFEST)
    digest = lf_hash(path)
    if digest != MANIFEST_SHA or digest != report["input_sha256"].get(MANIFEST):
        raise ValueError("Final-test manifest hash differs from frozen split provenance")
    images = path.read_text(encoding="utf-8").splitlines()
    if len(images) != IMAGE_COUNT or len(set(images)) != IMAGE_COUNT:
        raise ValueError("Expected exactly 2052 unique final-test images")
    for image in images:
        p = PurePosixPath(image)
        if (len(p.parts) != 3 or p.parts[0] != "images" or p.parts[1] not in ("train", "val")
                or ".." in p.parts or "\\" in image or ":" in image or p.as_posix() != image):
            raise ValueError("Invalid portable final-test path")
    # Read train/dev membership text only. No training or development images accessed.
    for relative, count, expected in raw.rq2_detector_training.MANIFESTS.values():
        other = resolve_input(root, relative)
        lines = other.read_text(encoding="utf-8").splitlines()
        if lf_hash(other) != expected or len(lines) != count or len(set(lines)) != count:
            raise ValueError("Frozen train/development manifest changed")
        if set(images) & set(lines):
            raise ValueError("Final test overlaps RQ2 train/development membership")
    return images


def frozen_inputs(root):
    root = Path(root)
    validate_protocol()
    images = frozen_test_images(root)
    identity = raw.checkpoint_identity(root)
    folder = resolve_input(root, final.OUTPUT)
    provenance = rq1._read_json(folder / "provenance.json")
    artifact = rq1._read_json(folder / "final_calibrator.json")
    if (provenance.get("status") != "complete" or provenance.get("final_fit_performed") is not True
            or provenance.get("beta_selection_reviewed") is not True
            or artifact.get("method") != "Beta" or artifact.get("epsilon") != 1e-15
            or artifact.get("mathematical_form") != final.FORM
            or tuple(artifact.get(k) for k in ("a", "b", "c")) != BETA
            or artifact.get("inclusion_threshold") != .01 or artifact.get("correctness_target") != final.TARGET
            or artifact.get("development_images") != 821 or artifact.get("fitted_predictions") != 7131
            or artifact.get("input_sha256") != provenance.get("input_sha256")):
        raise ValueError("Final Beta calibrator differs from frozen development decision")
    hashes = {}
    # Validate recorded development text dependencies read-only; never rerun them.
    for relative, expected in provenance["input_sha256"].items():
        p = PurePosixPath(relative)
        if p.is_absolute() or ".." in p.parts or "\\" in relative or ":" in relative:
            raise ValueError("Nonportable/escaped calibrator provenance input")
        digest = rq1._sha(resolve_input(root, relative))
        if digest != expected:
            raise ValueError(f"Frozen calibrator dependency changed: {relative}")
        hashes[relative] = digest
    for name in ("final_calibrator.json", "fit_summary.json", "calibrated_development_predictions.csv"):
        digest = rq1._sha(folder / name)
        if provenance.get("output_sha256", {}).get(name) != digest:
            raise ValueError("Frozen calibrator output hash mismatch")
        hashes[f"{final.OUTPUT}/{name}"] = digest
    for relative in (f"{final.OUTPUT}/provenance.json", MANIFEST, PARTITION_REPORT,
                     *(spec[0] for spec in raw.rq2_detector_training.MANIFESTS.values())):
        hashes[relative] = rq1._sha(resolve_input(root, relative))
    if (len(stability.SPECS) != 11 or raw.SETTINGS["conf"] != .001 or raw.SETTINGS["max_det"] != 300
            or raw.SETTINGS["imgsz"] != 640 or ORIGINAL_SETTINGS["conf"] != .01
            or matching.INCLUSION != .01 or calibration.EPSILON != 1e-15):
        raise ValueError("Frozen extraction/transformation constants differ")
    return images, hashes, identity


def write_csv(path, records, fields=None):
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _original(root, dataset, images, metadata):
    if os.name == "nt":
        raise RuntimeError("Real final-test inference is DICC-only")
    yield from raw._predict(root, dataset, images, metadata, settings=ORIGINAL_SETTINGS, record_image_hashes=True)


def calibrate_and_label(images, predictions, truth):
    labelled = matching.label_predictions(images, predictions, truth)
    if not labelled:
        raise ValueError("No retained final-test predictions; ranking/calibration undefined")
    probabilities = calibration.apply_calibrator({"method": "Beta", "parameters": BETA}, [r["confidence"] for r in labelled])
    for row, value in zip(labelled, probabilities):
        row.update(prediction_id=f"{row['image']}#{row['source_row']}", raw_confidence=row["confidence"],
                   calibrated_confidence=float(value), correct_iou50=row["correct_at_iou50"], correct_iou75=row["correct_at_iou75"],
                   checkpoint_sha256=raw.CHECKPOINT_SHA, checkpoint_epoch=raw.EPOCH)
    return labelled


def transformed_signals(root, dataset, images, labelled, metadata, predictor, output, evidence):
    by_image = {image: [] for image in images}
    audits = {r["prediction_id"]: [] for r in labelled}
    for row in labelled:
        by_image[row["image"]].append({"prediction_id": row["prediction_id"], "class_id": row["class_id"],
                                      "box": [row[k] for k in matching.BOX_KEYS]})
    views, count = [], 0
    with (output / "transformed_predictions.csv").open("x", newline="", encoding="utf-8") as pf, (output / "prediction_transform_matches.csv").open("x", newline="", encoding="utf-8") as mf:
        pw = csv.DictWriter(pf, fieldnames=("image", "transform_id", "detection_id", "class_id", "confidence", "box", "inverse_mapped_polygon", "max_det_saturation"))
        mw = csv.DictWriter(mf, fieldnames=("image", "prediction_id", "transform_id", "family", "evaluable", "status", "detection_id", "mapped_iou", "class_consistency_value", "localisation_stability_value", "max_det_saturation"))
        pw.writeheader()
        mw.writeheader()
        for image, width, height, transform_id, detections, info in predictor(root, dataset, images, metadata["transformed_inference"]):
            if count >= len(images) * 11:
                raise RuntimeError("Extra transformed test result")
            spec = stability.SPECS[count % 11]
            if image != images[count // 11] or transform_id != spec["id"]:
                raise RuntimeError("Foreign/duplicate/misordered transformed test result")
            if (width, height) != (evidence[image]["width"], evidence[image]["height"]):
                raise ValueError("Transformed dimensions differ from original image")
            original_hash = metadata["original_inference"]["source_image_sha256"][image]
            if info.get("source_image_sha256") != original_hash:
                raise ValueError("Original/transformed source image bytes differ")
            if len(detections) > 300:
                raise ValueError("Transformed detection count exceeds max_det")
            for d in detections:
                typed = dict(zip(raw.FIELDS, (image, d["class_id"], raw.rq2_detector_training.NAMES.get(d["class_id"]), d["confidence"], *d["box"])))
                raw.validate_row(typed, {image})
                if d["box"][2] > width or d["box"][3] > height or d["box"][2] <= d["box"][0] or d["box"][3] <= d["box"][1]:
                    raise ValueError("Invalid transformed detection box")
            mapped, matched = stability.match_transform(by_image[image], detections, spec, width, height)
            saturated = len(detections) == 300
            for d in mapped:
                pw.writerow({"image": image, "transform_id": transform_id, "detection_id": d["detection_id"],
                             "class_id": d["class_id"], "confidence": d["confidence"], "box": json.dumps(d["box"]),
                             "inverse_mapped_polygon": json.dumps(d["polygon"]), "max_det_saturation": saturated})
            for row in matched:
                audits[row["prediction_id"]].append(row)
                mw.writerow({"image": image, **row, "max_det_saturation": saturated})
            views.append({"image": image, "transform_id": transform_id, "prediction_count": len(detections),
                          "max_det_saturation": saturated, **info})
            count += 1
    if count != len(images) * 11:
        raise RuntimeError("Incomplete transformed final-test coverage")
    return [{**row, **stability.aggregate(audits[row["prediction_id"]])} for row in labelled], views


def ranking_outputs(output, images, records, scores, *, secondary=False):
    folder = output / ("sensitivity_iou75" if secondary else "primary_iou50")
    folder.mkdir()
    label = sensitivity.LABEL if secondary else "Primary IoU>=0.50 final-test evaluation"
    rows = sensitivity.sensitivity_rows(records) if secondary else [{**r, "incorrect_prediction": 1 - r["correct_iou50"]} for r in records]
    samples, boot = risk.bootstrap(images, rows, scores, replicates=REPLICATES, seed=SEED)
    methods, differences = risk.evaluation_tables(rows, scores, samples)
    for name, data in (("method_metrics.csv", methods), ("pairwise_auroc_differences.csv", differences)):
        write_csv(folder / name, [{"analysis_label": label, **r} for r in data])
    with (folder / "bootstrap_metrics.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("analysis_label", "replicate", "method", "auroc", "auprc", "auroc_valid", "auprc_valid"))
        writer.writeheader()
        for i in range(len(samples)):
            for j, method in enumerate(risk.METHODS):
                writer.writerow({"analysis_label": label, "replicate": i, "method": method,
                    "auroc": risk.nullable(samples[i, j, 0]), "auprc": risk.nullable(samples[i, j, 1]),
                    "auroc_valid": bool(np.isfinite(samples[i, j, 0])), "auprc_valid": bool(np.isfinite(samples[i, j, 1]))})
    boot.update(analysis_label=label, validity_by_method=[{"method": r["method"], **{k: v for k, v in r.items() if k.endswith("replicates")}} for r in methods])
    rq1._write_json(folder / "bootstrap_summary.json", boot, exclusive=True)
    incorrect = sum(r["incorrect_prediction"] for r in rows)
    rq1._write_json(folder / "summary.json", {"analysis_label": label, "test_role": ROLE, "test_images": len(images),
        "prediction_count": len(rows), "correct": len(rows) - incorrect, "incorrect": incorrect,
        "incorrect_prevalence": incorrect / len(rows), "auprc_reference_level": incorrect / len(rows),
        "primary_composite": "trustpcb_risk", "threshold_selected": False, "weights_tuned": False,
        "interpretation": "Secondary cannot replace primary; no test-driven revision of development decisions."}, exclusive=True)
    if not secondary:
        desc, corr, _ = risk.diagnostics(rows, scores)
        write_csv(folder / "signal_descriptives.csv", desc)
        write_csv(folder / "signal_correlations.csv", corr)
    return boot["draw_indices_sha256"]


def run(root, dataset, original_predictor=_original, transformed_predictor=stability._predict,
        truth_reader=matching.read_ground_truth):
    root, dataset = Path(root).resolve(), Path(dataset).resolve()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError("Existing complete/partial final evaluation; no overwrite or automatic restart")
    images, hashes, identity = frozen_inputs(root)
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution inputs before final-test execution")
    if not dataset.is_dir():
        raise FileNotFoundError("Configured dataset_root missing")
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "Pillow", "ultralytics", "torch")}
    if versions["ultralytics"] != raw.VERSION:
        raise RuntimeError(f"Expected ultralytics=={raw.VERSION}")
    metadata = {"experiment": "rq2_final_evaluation", "status": "running", "test_role": ROLE, **git,
        "test_manifest": MANIFEST, "test_manifest_sha256_lf": MANIFEST_SHA, "test_images": len(images),
        "input_sha256": hashes, "checkpoint_identity": identity,
        "final_calibrator_sha256": hashes[f"{final.OUTPUT}/final_calibrator.json"], "beta_parameters": BETA,
        "beta_epsilon": 1e-15, "reference_threshold": .01, "transformed_threshold": .001,
        "original_prediction_settings": ORIGINAL_SETTINGS, "transformed_prediction_settings": raw.SETTINGS,
        "transformations": stability.SPECS, "transformation_rules": stability.RULES,
        "risk_formulas": risk.FORMULAS, "metric_definitions": risk.DEFINITIONS,
        "bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES,
        "versions": versions, "python": platform.python_version(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "original_inference": {"population_role": "final-test"}, "transformed_inference": {"population_role": "final-test"}}
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance.json"
    rq1._write_json(provenance, metadata, exclusive=True)
    try:
        truth, evidence = truth_reader(dataset, images)
        predictions, seen = [], []
        for image, rows in original_predictor(root, dataset, images, metadata["original_inference"]):
            if len(seen) >= len(images) or image != images[len(seen)]:
                raise RuntimeError("Foreign/duplicate/misordered original test result")
            if len(rows) > 300:
                raise ValueError("Original detection count exceeds max_det")
            for row in rows:
                raw.validate_row(row, {image})
                if row["confidence"] < .01:
                    raise ValueError("Original prediction below frozen reference floor")
                if (row["x2"] > evidence[image]["width"] or row["y2"] > evidence[image]["height"]
                        or row["x2"] <= row["x1"] or row["y2"] <= row["y1"]):
                    raise ValueError("Original box outside image bounds")
            evidence[image]["original_prediction_count"] = len(rows)
            evidence[image]["original_max_det_saturation"] = len(rows) == 300
            predictions.extend(rows)
            seen.append(image)
        if seen != images:
            raise RuntimeError("Incomplete original final-test coverage")
        if set(metadata["original_inference"].get("source_image_sha256", {})) != set(images):
            raise RuntimeError("Original image hash inventory incomplete")
        labelled = calibrate_and_label(images, predictions, truth)
        write_csv(output / "original_predictions.csv", [{k: r[k] for k in (*raw.FIELDS, "prediction_id", "raw_confidence", "calibrated_confidence", "checkpoint_sha256", "checkpoint_epoch")} for r in labelled])
        write_csv(output / "labelled_predictions.csv", labelled)
        y = [r["correct_iou50"] for r in labelled]
        metrics = {name: calibration.metric_summary([r[key] for r in labelled], y)
                   for name, key in (("Raw", "raw_confidence"), ("Beta", "calibrated_confidence"))}
        rq1._write_json(output / "calibration_metrics.json", {"scope": "Independent final-test evaluation of the development-fitted Beta mapping", "test_role": ROLE,
            "correctness_iou": .50, "epsilon": 1e-15, "metrics": metrics}, exclusive=True)
        write_csv(output / "calibration_bins.csv", [{"method": m, **b} for m in metrics for b in metrics[m]["reliability_bins"]])
        rq1._write_json(output / "transformation_manifest.json", {"specifications": stability.SPECS, "rules": stability.RULES}, exclusive=True)
        records, views = transformed_signals(root, dataset, images, labelled, metadata, transformed_predictor, output, evidence)
        for row in records:
            if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in risk.SIGNALS):
                raise ValueError("Invalid final-test reliability signal")
        scores = risk.risk_scores(records)
        write_csv(output / "prediction_stability_scores.csv", [{**r, **{m: float(scores[m][i]) for m in risk.METHODS}} for i, r in enumerate(records)])
        primary_draws = ranking_outputs(output, images, records, scores)
        secondary_draws = ranking_outputs(output, images, records, scores, secondary=True)
        if primary_draws != secondary_draws:
            raise RuntimeError("Primary/secondary bootstrap draw mismatch")
        rq1._write_json(output / "inference_audit.json", {"images": evidence, "transformed_views": views,
            "expected_transformed_passes": len(images) * 11, "max_det_saturated_views": sum(v["max_det_saturation"] for v in views)}, exclusive=True)
        after = frozen_inputs(root)
        if after[1:] != (hashes, identity) or rq1.git_provenance(root) != git:
            raise RuntimeError("Frozen inputs/source changed during final evaluation")
        # Recheck only test image/label files already used; never enumerate the dataset.
        for image, info in evidence.items():
            if (rq1._sha(dataset / image) != metadata["original_inference"]["source_image_sha256"][image]
                    or rq1._sha(dataset / info["label"]) != info["label_sha256"]):
                raise RuntimeError("Test image/annotation changed during evaluation")
        metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
            output_sha256={p.relative_to(output).as_posix(): rq1._sha(p) for p in output.rglob("*") if p.is_file() and p != provenance})
        rq1._write_json(provenance, metadata)
        return output
    except BaseException:
        metadata["status"] = "incomplete"
        rq1._write_json(provenance, metadata)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicc", action="store_true")
    args = parser.parse_args(argv)
    if os.name == "nt" or not args.dicc:
        raise RuntimeError("Real final-test evaluation is DICC-only; --dicc required")
    root = find_project_root()
    paths = load_paths(root)
    if paths.project_root != root:
        raise ValueError("project_root must match checkout")
    print(run(root, paths.dataset_root))


if __name__ == "__main__":
    main()
