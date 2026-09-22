"""Development-only transformation stability. Real inference is DICC-only."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.optimize import linear_sum_assignment

from trustpcb import rq1, rq2_final_calibrator as final
from trustpcb import rq2_confidence_distribution as raw
from trustpcb.dataset_config import find_project_root, load_paths
from trustpcb.rq2_artifact_paths import resolve_input

OUTPUT = "runs/rq2/transformation_stability"
FAMILIES = ("brightness", "contrast", "blur", "rotation", "translation")
SPECS = tuple(
    [{"id": f"{family}_{factor:.2f}", "family": family, "factor": factor}
     for family in FAMILIES[:2] for factor in (.90, 1.10)]
    + [{"id": "blur_0.6", "family": "blur", "sigma": .6}]
    + [{"id": f"rotation_{degrees:+d}", "family": "rotation", "degrees": degrees} for degrees in (-2, 2)]
    + [{"id": f"translation_{axis}_{fraction:+.2f}", "family": "translation", "axis": axis, "fraction": fraction}
       for axis in ("x", "y") for fraction in (-.02, .02)])
RULES = {
    "affine_border": "constant per-original-image channel-wise median; grid-constant bilinear interpolation",
    "coordinates": "xy edge coordinates; pixel centres (column+0.5,row+0.5); image bounds [0,width] x [0,height]",
    "rotation": "positive counterclockwise visually, centre (width/2,height/2); unchanged canvas",
    "contrast": "single scalar mean over all original pixels and channels",
    "blur": "scipy gaussian_filter sigma=(0.6,0.6,0), reflect boundary, truncate=4.0; no channel mixing",
    "quantisation": "float64 operations, clip [0,255], round-to-nearest-even, uint8",
    "matching": "class-agnostic IoU>=0.50; maximise cardinality then total polygon IoU",
    "ties": "source reference order; transformed detections sorted by xyxy then detector index; deterministic SciPy assignment; no class/confidence tie weights",
    "aggregation": "mean within evaluable family, then equal mean across evaluable families",
}


def rectangle(box):
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)


def map_points(points, matrix):
    points = np.asarray(points, dtype=float)
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def forward_matrix(spec, width, height):
    matrix = np.eye(3)
    if spec["family"] == "rotation":
        angle = math.radians(spec["degrees"])
        matrix[:2, :2] = [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]]
        centre = np.array([width / 2, height / 2])
        matrix[:2, 2] = centre - matrix[:2, :2] @ centre
    elif spec["family"] == "translation":
        axis = 0 if spec["axis"] == "x" else 1
        matrix[axis, 2] = spec["fraction"] * (width if axis == 0 else height)
    return matrix


def transform(image, spec):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected uint8 three-channel image")
    h, w = image.shape[:2]
    values = image.astype(np.float64)
    matrix = forward_matrix(spec, w, h)
    median = np.median(values, axis=(0, 1))
    family = spec["family"]
    if family == "brightness":
        values = values * spec["factor"]
    elif family == "contrast":
        mean = values.mean()
        values = mean + spec["factor"] * (values - mean)
    elif family == "blur":
        values = gaussian_filter(values, sigma=(spec["sigma"], spec["sigma"], 0), mode="reflect", truncate=4.)
    elif family in ("rotation", "translation"):
        inverse = np.linalg.inv(matrix)
        # Edge-coordinate xy -> pixel-centre coordinates -> scipy row/column indices.
        swap = np.array([[0., 1.], [1., 0.]])
        linear = swap @ inverse[:2, :2] @ swap
        offset = swap @ (inverse[:2, :2] @ np.array([.5, .5]) + inverse[:2, 2] - .5)
        values = np.stack([affine_transform(values[:, :, ch], linear, offset=offset,
                           output_shape=(h, w), order=1, mode="grid-constant", cval=median[ch],
                           prefilter=False) for ch in range(3)], axis=2)
    else:
        raise ValueError("Unknown transformation family")
    return np.rint(np.clip(values, 0, 255)).astype(np.uint8), matrix, median.tolist()


def evaluable(box, matrix, width, height, geometric):
    if not geometric:
        return True
    corners = map_points(rectangle(box), matrix)
    return bool(np.all(corners >= 0) and np.all(corners <= [width, height]))


def signed_area(polygon):
    p = np.asarray(polygon, dtype=float)
    if len(p) < 3:
        return 0.
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)) / 2)


def polygon_area(polygon):
    return abs(signed_area(polygon))


def polygon_intersection(subject, clip):
    """Sutherland-Hodgman convex clipping; accepts either winding order."""
    output = [np.asarray(p, dtype=float) for p in subject]
    clip = np.asarray(clip, dtype=float)
    if signed_area(clip) < 0:
        clip = clip[::-1]
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if not output:
            break
        incoming, output = output, []
        previous = incoming[-1]
        previous_distance = cross(b - a, previous - a)
        for current in incoming:
            distance = cross(b - a, current - a)
            if (distance >= 0) != (previous_distance >= 0):
                output.append(previous + (current - previous) * (previous_distance / (previous_distance - distance)))
            if distance >= 0:
                output.append(current)
            previous, previous_distance = current, distance
    return np.asarray(output).reshape(-1, 2)


def polygon_iou(first, second):
    area1, area2 = polygon_area(first), polygon_area(second)
    if area1 <= 0 or area2 <= 0:
        return 0.
    intersection = min(area1, area2, polygon_area(polygon_intersection(first, second)))
    return float(np.clip(intersection / (area1 + area2 - intersection), 0, 1))


def assign(iou):
    """Dummies permit unmatched rows; invalid real edges cannot be chosen."""
    iou = np.asarray(iou, dtype=float)
    if iou.ndim != 2 or not np.isfinite(iou).all() or np.any((iou < 0) | (iou > 1)):
        raise ValueError("Invalid IoU matrix")
    n, m = iou.shape
    if not n or not m:
        return {}
    # One extra match dominates every possible total-IoU difference (<= min(n,m)).
    reward = np.zeros((n, m + n))
    reward[:, :m] = np.where(iou >= .50, min(n, m) + 1 + iou, -1.)
    row, col = linear_sum_assignment(-reward)
    return {int(i): int(j) for i, j in zip(row, col) if j < m}


def match_transform(references, detections, spec, width, height):
    matrix = forward_matrix(spec, width, height)
    inverse = np.linalg.inv(matrix)
    geometric = spec["family"] in ("rotation", "translation")
    flags = [evaluable(r["box"], matrix, width, height, geometric) for r in references]
    detections = sorted(enumerate(detections), key=lambda pair: (*pair[1]["box"], pair[0]))
    mapped = []
    for index, (_, detection) in enumerate(detections):
        mapped.append({**detection, "detection_id": index, "polygon": map_points(rectangle(detection["box"]), inverse).tolist()})
    valid = [i for i, flag in enumerate(flags) if flag]
    ious = np.array([[polygon_iou(rectangle(references[i]["box"]), d["polygon"]) for d in mapped] for i in valid]).reshape(len(valid), len(mapped))
    assignment = assign(ious)
    selected = {valid[i]: (j, float(ious[i, j])) for i, j in assignment.items()}
    audits = []
    for i, reference in enumerate(references):
        match = selected.get(i)
        audits.append({"prediction_id": reference["prediction_id"], "transform_id": spec["id"],
                       "family": spec["family"], "evaluable": flags[i],
                       "status": "not_evaluable" if not flags[i] else "matched" if match else "unmatched",
                       "detection_id": match[0] if match else None,
                       "mapped_iou": match[1] if match else None,
                       "class_consistency_value": int(mapped[match[0]]["class_id"] == reference["class_id"]) if match else 0 if flags[i] else None,
                       "localisation_stability_value": match[1] if match else 0. if flags[i] else None})
    return mapped, audits


def aggregate(audits):
    if len(audits) != 11 or {r["transform_id"] for r in audits} != {s["id"] for s in SPECS}:
        raise ValueError("Expected exactly eleven unique transformation results")
    scores = {}
    for signal in ("class_consistency", "localisation_stability"):
        family_scores = []
        for family in FAMILIES:
            values = [r[f"{signal}_value"] for r in audits if r["family"] == family and r["evaluable"]]
            value = math.fsum(values) / len(values) if values else None
            scores[f"{signal}_{family}"] = value
            if value is not None:
                family_scores.append(value)
        scores[signal] = math.fsum(family_scores) / len(family_scores) if family_scores else None
    scores.update(n_evaluable_transforms=sum(r["evaluable"] for r in audits),
                  n_matched_transforms=sum(r["status"] == "matched" for r in audits),
                  n_evaluable_families=sum(scores[f"class_consistency_{f}"] is not None for f in FAMILIES))
    return scores


def load_inputs(root):
    root = Path(root)
    # Validation only: no fitter, selection procedure or inference is invoked.
    images, source, originals, hashes, _, _ = final.load_inputs(root, approve_beta_selection=True)
    folder = root / final.OUTPUT
    provenance = rq1._read_json(folder / "provenance.json")
    if (provenance.get("status") != "complete" or provenance.get("final_fit_performed") is not True
            or provenance.get("input_sha256") != hashes or provenance.get("beta_selection_reviewed") is not True):
        raise ValueError("Final calibrator provenance incomplete or does not bind this population")
    for name in ("final_calibrator.json", "calibrated_development_predictions.csv", "fit_summary.json"):
        digest = rq1._sha(folder / name)
        if provenance.get("output_sha256", {}).get(name) != digest:
            raise ValueError(f"Frozen final calibrator artifact hash mismatch: {name}")
        hashes[f"{final.OUTPUT}/{name}"] = digest
    artifact = rq1._read_json(folder / "final_calibrator.json")
    if (artifact.get("method") != "Beta" or artifact.get("epsilon") != 1e-15
            or artifact.get("mathematical_form") != final.FORM or artifact.get("inclusion_threshold") != .01
            or artifact.get("correctness_target") != final.TARGET
            or artifact.get("development_images") != 821 or artifact.get("fitted_predictions") != len(source)
            or artifact.get("input_sha256") != provenance["input_sha256"]):
        raise ValueError("Frozen Beta specification differs")
    parameters = [float(artifact[k]) for k in ("a", "b", "c")]
    if not all(math.isfinite(v) for v in parameters) or min(parameters[:2]) < 0:
        raise ValueError("Invalid monotonic Beta parameters")
    with (folder / "calibrated_development_predictions.csv").open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))
    if len(records) != len(source):
        raise ValueError("Reference population count differs")
    expected = final.comparison.apply_calibrator({"method": "Beta", "parameters": parameters}, [r["raw_confidence"] for r in source])
    references = []
    for record, original, row, calibrated in zip(records, originals, source, expected):
        if any(record.get(key) != value for key, value in original.items()):
            raise ValueError("Reference identity/order/source fields differ")
        if (record["prediction_id"] != row["prediction_id"] or float(record["raw_confidence"]) != row["raw_confidence"]
                or int(record["correct_iou50"]) != row["correct_iou50"]
                or not math.isclose(float(record["calibrated_confidence"]), float(calibrated), rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Frozen calibrated reference values differ")
        typed = {"image": record["image"], "class_id": int(record["class_id"]), "class_name": record["class_name"],
                 "confidence": row["raw_confidence"], **{k: float(record[k]) for k in ("x1", "y1", "x2", "y2")}}
        raw.validate_row(typed, set(images))
        box = [typed[k] for k in ("x1", "y1", "x2", "y2")]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Degenerate reference box")
        references.append({"record": record, "image": row["image"], "prediction_id": row["prediction_id"],
                           "class_id": typed["class_id"], "box": box})
    hashes[f"{final.OUTPUT}/provenance.json"] = rq1._sha(folder / "provenance.json")
    identity = raw.checkpoint_identity(root)
    return images, references, hashes, identity


def _predict(root, dataset, images, metadata):
    if os.name == "nt":
        raise RuntimeError("Real image inference is DICC-only")
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    model = YOLO(str(resolve_input(root, raw.CHECKPOINT)))
    if model.names != raw.rq2_detector_training.NAMES:
        raise ValueError("Frozen detector class mapping differs")
    for image in images:
        source = (dataset / image).resolve()
        if not source.is_relative_to(dataset) or not source.is_file():
            raise ValueError("Missing/escaped development image")
        image_hash = rq1._sha(source)
        with Image.open(source) as opened:
            pixels = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))[:, :, ::-1].copy()
        height, width = pixels.shape[:2]
        for spec in SPECS:
            view, matrix, median = transform(pixels, spec)
            results = model.predict(source=view, **raw.SETTINGS)
            if len(results) != 1 or tuple(results[0].orig_shape) != (height, width):
                raise RuntimeError("Unexpected transformed result dimensions/coverage")
            effective = vars(model.predictor.args).copy()
            for key, value in raw.SETTINGS.items():
                if not raw.rq2_detector_training._argument_matches(key, effective.get(key), value):
                    raise RuntimeError(f"Unexpected inference argument: {key}")
            # Source arrays are large; record their type, not pixels, in full effective args.
            effective["source"] = f"in-memory uint8 BGR transformed {metadata.get('population_role', 'development')} image"
            effective = json.loads(json.dumps(effective, default=str))
            if "effective_prediction_settings" in metadata and metadata["effective_prediction_settings"] != effective:
                raise RuntimeError("Effective inference settings changed between views")
            metadata["effective_prediction_settings"] = effective
            metadata["resolved_device"] = str(model.predictor.device)
            detections = []
            boxes = results[0].boxes
            if boxes is not None:
                for box, confidence, cls in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
                    detections.append({"box": box, "class_id": int(cls), "confidence": confidence})
            yield image, width, height, spec["id"], detections, {"forward_affine": matrix.tolist(), "border_median_bgr": median, "source_image_sha256": image_hash}
        if rq1._sha(source) != image_hash:
            raise RuntimeError("Development image changed during inference")


def run(root, dataset, predictor):
    root, dataset = Path(root).resolve(), Path(dataset).resolve()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError("Existing complete/incomplete transformation output; archive explicitly before retry")
    images, references, hashes, identity = load_inputs(root)
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution inputs before DICC inference")
    if not dataset.is_dir():
        raise FileNotFoundError("dataset_root missing")
    if len(SPECS) != 11:
        raise ValueError("Expected eleven frozen transforms")
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "Pillow", "ultralytics", "torch")}
    if versions["ultralytics"] != raw.VERSION:
        raise RuntimeError(f"Expected ultralytics=={raw.VERSION}")
    metadata = {"experiment": "rq2_transformation_stability", "status": "running", **git,
                "input_sha256": hashes, "checkpoint_identity": identity, "versions": versions,
                "prediction_settings": raw.SETTINGS, "rules": RULES,
                "started_utc": datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance.json"
    rq1._write_json(provenance, metadata, exclusive=True)
    try:
        rq1._write_json(output / "transformation_manifest.json", {"specifications": SPECS, "rules": RULES}, exclusive=True)
        by_image = {image: [] for image in images}
        audits = {r["prediction_id"]: [] for r in references}
        for reference in references:
            by_image[reference["image"]].append(reference)
        views, expected_index = [], 0
        with (output / "transformed_predictions.csv").open("x", newline="", encoding="utf-8") as pf, (output / "prediction_transform_matches.csv").open("x", newline="", encoding="utf-8") as mf:
            predictions_writer = csv.DictWriter(pf, fieldnames=("image", "transform_id", "detection_id", "class_id", "confidence", "x1", "y1", "x2", "y2", "inverse_mapped_polygon", "max_det_saturation"))
            matches_writer = csv.DictWriter(mf, fieldnames=("image", "prediction_id", "transform_id", "family", "evaluable", "status", "detection_id", "mapped_iou", "class_consistency_value", "localisation_stability_value", "max_det_saturation"))
            predictions_writer.writeheader()
            matches_writer.writeheader()
            for image, width, height, transform_id, detections, view_info in predictor(root, dataset, images, metadata):
                if expected_index >= len(images) * 11:
                    raise RuntimeError("Extra transformed result")
                spec = SPECS[expected_index % 11]
                if image != images[expected_index // 11] or transform_id != spec["id"]:
                    raise RuntimeError("Foreign/duplicate/misordered transformed result")
                if width <= 0 or height <= 0 or len(detections) > 300:
                    raise ValueError("Invalid dimensions/detection count")
                for ref in by_image[image]:
                    if ref["box"][2] > width or ref["box"][3] > height:
                        raise ValueError("Reference box outside source dimensions")
                for detection in detections:
                    box = detection["box"]
                    typed = dict(zip(raw.FIELDS, (image, detection["class_id"], raw.rq2_detector_training.NAMES.get(detection["class_id"]), detection["confidence"], *box)))
                    raw.validate_row(typed, {image})
                    if box[2] <= box[0] or box[3] <= box[1] or box[2] > width or box[3] > height:
                        raise ValueError("Invalid transformed box")
                saturated = len(detections) == 300
                mapped, matched = match_transform(by_image[image], detections, spec, width, height)
                for detection in mapped:
                    predictions_writer.writerow({"image": image, "transform_id": transform_id,
                        "detection_id": detection["detection_id"], "class_id": detection["class_id"], "confidence": detection["confidence"],
                        **dict(zip(("x1", "y1", "x2", "y2"), detection["box"])),
                        "inverse_mapped_polygon": json.dumps(detection["polygon"]), "max_det_saturation": saturated})
                for audit in matched:
                    audits[audit["prediction_id"]].append(audit)
                    matches_writer.writerow({"image": image, **audit, "max_det_saturation": saturated})
                views.append({"image": image, "transform_id": transform_id, "width": width, "height": height,
                              "prediction_count": len(detections), "max_det_saturation": saturated, **view_info})
                expected_index += 1
        if expected_index != len(images) * 11:
            raise RuntimeError("Incomplete transformed development coverage")
        scores = [{**ref["record"], **aggregate(audits[ref["prediction_id"]])} for ref in references]
        final.comparison.write_csv(output / "prediction_stability_scores.csv", scores)
        rq1._write_json(output / "summary.json", {"development_images": len(images), "reference_predictions": len(references),
            "transformed_views": len(views), "max_det_saturated_views": sum(v["max_det_saturation"] for v in views),
            "views": views, "matched_transforms": sum(s["n_matched_transforms"] for s in scores),
            "not_evaluable_transforms": len(references) * 11 - sum(s["n_evaluable_transforms"] for s in scores)}, exclusive=True)
        after = load_inputs(root)
        if after[2] != hashes or after[3] != identity or rq1.git_provenance(root) != git:
            raise RuntimeError("Frozen inputs/source changed during inference")
        metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        output_sha256={p.name: rq1._sha(p) for p in output.iterdir() if p != provenance})
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
        raise RuntimeError("Real transformation inference is DICC-only; --dicc required")
    root = find_project_root()
    paths = load_paths(root)
    if paths.project_root != root:
        raise ValueError("project_root must match this checkout")
    print(run(root, paths.dataset_root, _predict))


if __name__ == "__main__":
    main()
