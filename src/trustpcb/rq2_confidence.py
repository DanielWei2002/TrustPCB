"""Stage 3B1: development-only raw confidence extraction; real inference is DICC-only."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path, PurePosixPath
import statistics

from trustpcb import rq1, rq2_train
from trustpcb.dataset_config import find_project_root, load_paths

DEVELOPMENT, IMAGE_COUNT, MANIFEST_SHA = rq2_train.MANIFESTS["val"]
CHECKPOINT = "runs/rq2/stage2/seed_24209199/weights/selected.pt"
CHECKPOINT_SHA = "793992d25ef671c45c820dc1b5a61bca5837af0d3726cd3830b2f2c656a6a85f"
EPOCH = 72
VERSION = "8.4.117"
OUTPUT = "runs/rq2/stage3b1/raw_confidence"
SETTINGS = {"conf": 0.001, "iou": 0.7, "max_det": 300, "imgsz": 640,
            "device": 0, "batch": 1, "rect": True, "augment": False,
            "agnostic_nms": False, "classes": None, "quantize": None,
            "save": False, "save_txt": False, "save_crop": False,
            "show": False, "verbose": False}
EDGES = (0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
THRESHOLDS = EDGES[:5]
FIELDS = ("image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2")


def development_images(root):
    """Read only the pinned development list, never a train/test list or labels."""
    raw = (Path(root) / DEVELOPMENT).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError("Frozen development manifest changed; no alternative partition is permitted")
    images = raw.decode("utf-8").splitlines()
    if len(images) != IMAGE_COUNT or len(set(images)) != IMAGE_COUNT:
        raise ValueError("Invalid development coverage/duplicates")
    for image in images:
        path = PurePosixPath(image)
        if (len(path.parts) != 3 or path.parts[0] != "images"
                or path.parts[1] not in ("train", "val") or ".." in path.parts
                or "\\" in image or ":" in image or path.as_posix() != image):
            raise ValueError("Invalid portable development image path")
    return images


def checkpoint_identity(root):
    identity = rq2_train.file_identity(Path(root) / CHECKPOINT)
    if identity["sha256"] != CHECKPOINT_SHA:
        raise ValueError("Checkpoint differs from the frozen epoch-72 selected.pt")
    return {"relative_path": CHECKPOINT, "epoch": EPOCH,
            "sha256": identity["sha256"], "size_bytes": identity["size_bytes"]}


def plan(root):
    development_images(root)
    return {"stage": "rq2_stage3b1", "development_manifest": DEVELOPMENT,
            "manifest_sha256_lf": MANIFEST_SHA, "development_images": IMAGE_COUNT,
            "checkpoint": CHECKPOINT, "checkpoint_epoch": EPOCH,
            "checkpoint_sha256": CHECKPOINT_SHA, "prediction_settings": SETTINGS,
            "ultralytics_version": VERSION, "output": OUTPUT,
            "purpose": "descriptive post-NMS raw confidence; no threshold recommendation",
            "final_calibration_threshold": None, **rq1.git_provenance(root)}


def validate_row(row, allowed):
    if row["image"] not in allowed:
        raise ValueError("Prediction outside the frozen development set")
    class_id = row["class_id"]
    if type(class_id) is not int or class_id not in rq2_train.NAMES:
        raise ValueError("Invalid predicted class ID")
    if row["class_name"] != rq2_train.NAMES[class_id]:
        raise ValueError("Predicted class mapping differs")
    c = row["confidence"]
    coords = [row[k] for k in ("x1", "y1", "x2", "y2")]
    if not all(math.isfinite(v) for v in [c, *coords]) or not EDGES[0] <= c <= 1:
        raise ValueError("Nonfinite/out-of-range prediction")
    if coords[0] < 0 or coords[1] < 0 or coords[2] < coords[0] or coords[3] < coords[1]:
        raise ValueError("Invalid original-image xyxy bounding box")


def summarize(images, rows):
    """Pure deterministic descriptive statistics; includes zero-prediction images."""
    counts = dict.fromkeys(images, 0)
    if not counts or len(counts) != len(images):
        raise ValueError("Expected nonempty unique image identifiers")
    classes = dict.fromkeys(rq2_train.NAMES, 0)
    confidences = []
    bins = [0] * (len(EDGES) - 1)
    for row in rows:
        validate_row(row, counts)
        counts[row["image"]] += 1
        classes[row["class_id"]] += 1
        c = row["confidence"]
        confidences.append(c)
        for i, (low, high) in enumerate(zip(EDGES, EDGES[1:])):
            if low <= c and (c < high or (i == len(bins) - 1 and c <= high)):
                bins[i] += 1
                break
    def stats(values):
        return {"min": min(values), "median": statistics.median(values),
                "mean": math.fsum(values) / len(values), "max": max(values)} if values else {
                    k: None for k in ("min", "median", "mean", "max")}
    total = len(confidences)
    percent = lambda n: 100 * n / total if total else 0.0
    return {
        "image_count": len(images), "total_predictions": total,
        "images_with_predictions": sum(n > 0 for n in counts.values()),
        "predictions_per_image": stats(list(counts.values())), "confidence": stats(confidences),
        "per_class": [{"class_id": k, "class_name": rq2_train.NAMES[k], "count": v}
                      for k, v in classes.items()],
        "confidence_intervals": [{"lower_inclusive": low, "upper": high,
                                  "upper_inclusive": i == len(bins) - 1,
                                  "count": bins[i], "percentage": percent(bins[i])}
                                 for i, (low, high) in enumerate(zip(EDGES, EDGES[1:]))],
        "cumulative_retained": [{"threshold_inclusive": t,
                                 "count": sum(c >= t for c in confidences),
                                 "percentage": percent(sum(c >= t for c in confidences))}
                                for t in THRESHOLDS],
        "images_at_max_det": sum(n == SETTINGS["max_det"] for n in counts.values()),
        "final_calibration_threshold": None,
    }


def histogram(rows, target):
    # Plotting is deferred to DICC (or deliberately tiny synthetic tests).
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist([r["confidence"] for r in rows], bins=[i / 100 for i in range(101)])
    ax.set(xlim=(0, 1), xlabel="Raw post-NMS YOLO confidence", ylabel="Prediction count",
           title="RQ2 development confidence (exploratory floor 0.001)")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _predict(root, dataset, images, metadata):
    """DICC-only adapter; receives exactly the validated development image IDs."""
    from ultralytics import YOLO
    model = YOLO(str(Path(root) / CHECKPOINT))
    if model.names != rq2_train.NAMES:
        raise ValueError("Frozen detector class mapping differs")
    for image in images:
        source = (dataset / image).resolve()
        if not source.is_relative_to(dataset) or not source.is_file():
            raise ValueError(f"Missing/escaped development image: {image}")
        results = model.predict(source=str(source), **SETTINGS)
        if len(results) != 1 or Path(results[0].path).resolve() != source:
            raise RuntimeError("Unexpected prediction source/coverage")
        effective = {k: getattr(model.predictor.args, k) for k in SETTINGS}
        for key, expected in SETTINGS.items():
            if not rq2_train._argument_matches(key, effective[key], expected):
                raise RuntimeError(f"Unexpected prediction setting: {key}")
        metadata["effective_prediction_settings"] = effective
        metadata["resolved_device"] = str(model.predictor.device)
        rows = []
        boxes = results[0].boxes
        if boxes is not None:
            for coords, confidence, cls in zip(boxes.xyxy.cpu().tolist(),
                                               boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
                class_id = int(cls)
                rows.append(dict(zip(FIELDS, (image, class_id, rq2_train.NAMES[class_id],
                                              confidence, *coords))))
        yield image, rows


def extract(root, dataset, predictor, plotter=histogram):
    """Exclusive output reservation; injectable adapters for tiny CPU fixtures."""
    root, dataset = Path(root).resolve(), Path(dataset).resolve()
    info = plan(root)
    if info["git_dirty"]:
        raise RuntimeError("Commit/resolve execution/scientific inputs before DICC extraction")
    images = development_images(root)
    identity = checkpoint_identity(root)
    if not dataset.is_dir():
        raise FileNotFoundError("Configured dataset_root does not exist")
    versions = rq1._environment()
    if importlib.metadata.version("ultralytics") != VERSION:
        raise RuntimeError(f"Use the recorded detector environment: ultralytics=={VERSION}")
    # Fail before inference if the required output-plot dependency is missing.
    matplotlib_version = importlib.metadata.version("matplotlib")
    out = root / OUTPUT
    out.mkdir(parents=True, exist_ok=False)  # blocks both partial and completed output reuse
    metadata = {**info, "checkpoint_identity": identity, "environment": versions,
                "matplotlib_version": matplotlib_version,
                "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "coordinates": "xyxy pixels in original image", "confidence": "uncalibrated post-NMS score"}
    provenance = out / "provenance.json"
    rq1._write_json(provenance, metadata, exclusive=True)
    try:
        rows, seen, per_image = [], [], []
        with (out / "predictions.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for image, predictions in predictor(root, dataset, images, metadata):
                if len(seen) >= len(images) or image != images[len(seen)]:
                    raise RuntimeError("Unexpected/duplicate/misordered development result")
                for row in predictions:
                    validate_row(row, {image})
                    writer.writerow(row)
                rows.extend(predictions)
                seen.append(image)
                per_image.append({"image": image, "prediction_count": len(predictions)})
        if seen != images:
            raise RuntimeError("Incomplete development inference coverage")
        if checkpoint_identity(root) != identity or plan(root) != info:
            raise RuntimeError("Frozen inputs/Git state changed during extraction")
        with (out / "images.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("image", "prediction_count"))
            writer.writeheader()
            writer.writerows(per_image)
        rq1._write_json(out / "summary.json", summarize(images, rows), exclusive=True)
        plotter(rows, out / "confidence_histogram.png")
        metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        output_sha256={p.name: rq1._sha(p) for p in out.iterdir() if p != provenance})
        rq1._write_json(provenance, metadata)
        return out
    except BaseException:
        metadata["status"] = "incomplete"
        rq1._write_json(provenance, metadata)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "extract"))
    parser.add_argument("--dicc", action="store_true")
    args = parser.parse_args(argv)
    root = find_project_root()
    if args.command == "plan":
        print(rq1._json(plan(root)))
        return
    if os.name == "nt" or not args.dicc:
        raise RuntimeError("Real extraction is DICC-only; --dicc required; Windows is prohibited")
    paths = load_paths(root)
    if paths.project_root != root:
        raise ValueError("project_root must match this checkout")
    print(extract(root, paths.dataset_root, _predict))


if __name__ == "__main__":
    main()
