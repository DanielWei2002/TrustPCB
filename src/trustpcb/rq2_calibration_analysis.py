"""Calibration analysis: fixed-threshold development correctness and raw calibration, CPU only."""

import argparse
from bisect import bisect_right
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path, PurePosixPath

from trustpcb import rq1, rq2_confidence_distribution as raw, rq2_detector_training
from trustpcb.rq2_artifact_paths import resolve_input, canonical_identifier, reject_historical_output
from trustpcb.dataset_config import find_project_root, load_paths

INCLUSION = 0.01
OUTPUT = "runs/rq2/calibration_analysis"
# Compatibility for this uncommitted analysis belongs with its implementation,
# not with the separate naming-only refactor of previously committed workflows.
HISTORICAL_OUTPUT = "runs/rq2/stage3b2/raw_calibration"
BIN_EDGES = tuple(i / 10 for i in range(11))
NLL_EPSILON = 1e-15
BOX_KEYS = ("x1", "y1", "x2", "y2")
MATCH_FIELDS = ("matched_gt_id", "matched_iou", "best_unused_iou", "best_same_class_iou", "status")
FIELDS = (*raw.FIELDS, "source_row", "correct_at_iou50", "correct_at_iou75",
          *(f"{field}_{suffix}" for suffix in ("iou50", "iou75") for field in MATCH_FIELDS))


def iou(a, b):
    """Continuous xyxy pixel geometry (no integer +1 pixel convention)."""
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def label_predictions(images, predictions, ground_truth):
    """Stable confidence-descending greedy matching; threshold states are independent."""
    if len(set(images)) != len(images) or set(ground_truth) != set(images):
        raise ValueError("Ground-truth coverage must equal development membership")
    groups = {image: [] for image in images}
    for number, row in enumerate(predictions, 1):
        raw.validate_row(row, groups)  # reject foreign rows even below inclusion
        if row["confidence"] >= INCLUSION:
            groups[row["image"]].append({**row, "source_row": number})
    output = []
    for image in images:
        ordered = sorted(groups[image], key=lambda row: (-row["confidence"], row["source_row"]))
        truth = ground_truth[image]
        used = {"iou50": set(), "iou75": set()}
        for row in ordered:
            box = tuple(row[key] for key in BOX_KEYS)
            candidates = [(j, iou(box, gt["box"])) for j, gt in enumerate(truth)
                          if gt["class_id"] == row["class_id"]]
            for suffix, threshold in (("iou50", 0.50), ("iou75", 0.75)):
                available = [(j, overlap) for j, overlap in candidates if j not in used[suffix]]
                # max keeps the first GT label-file line on exact IoU ties.
                best = max(available, key=lambda pair: pair[1], default=(None, 0.))
                correct = best[0] is not None and best[1] >= threshold
                if correct:
                    used[suffix].add(best[0])
                duplicate = not correct and any(j in used[suffix] and overlap >= threshold
                                               for j, overlap in candidates)
                row[f"correct_at_{suffix}"] = int(correct)
                details = (truth[best[0]]["id"] if correct else "", best[1] if correct else "",
                           best[1], max((v for _, v in candidates), default=0.),
                           "matched" if correct else "duplicate" if duplicate else "unmatched")
                row.update({f"{key}_{suffix}": value for key, value in zip(MATCH_FIELDS, details)})
            output.append(row)
    return output


def metrics(rows, label):
    bins = [[] for _ in range(10)]
    for row in rows:
        c, y = row["confidence"], row[label]
        if not math.isfinite(c) or not INCLUSION <= c <= 1 or y not in (0, 1):
            raise ValueError("Invalid retained probability/correctness label")
        bins[min(9, bisect_right(BIN_EDGES, c) - 1)].append((c, y))
    n = len(rows)
    correct = sum(row[label] for row in rows)
    reliability = []
    for index, entries in enumerate(bins):
        mean = math.fsum(c for c, _ in entries) / len(entries) if entries else None
        rate = sum(y for _, y in entries) / len(entries) if entries else None
        reliability.append({"lower_inclusive": BIN_EDGES[index], "upper": BIN_EDGES[index + 1],
                            "upper_inclusive": index == 9, "count": len(entries),
                            "mean_confidence": mean, "correctness_rate": rate})
    def logloss(row):
        p = min(1 - NLL_EPSILON, max(NLL_EPSILON, row["confidence"]))
        return -math.log(p) if row[label] else -math.log1p(-p)
    return {"count": n, "correct": correct, "incorrect": n - correct,
            "correctness_rate": correct / n if n else None,
            "brier": math.fsum((row["confidence"] - row[label]) ** 2 for row in rows) / n if n else None,
            "negative_log_likelihood": math.fsum(logloss(row) for row in rows) / n if n else None,
            "ece": math.fsum(b["count"] * abs(b["mean_confidence"] - b["correctness_rate"])
                             for b in reliability if b["count"]) / n if n else None,
            "reliability_bins": reliability}


def summarize(rows):
    return {"inclusion_threshold": INCLUSION, "total_retained_predictions": len(rows),
            "primary_iou50": metrics(rows, "correct_at_iou50"),
            "sensitivity_iou75": metrics(rows, "correct_at_iou75"),
            "per_class": [{"class_id": cls, "class_name": name,
                           **{suffix: metrics([r for r in rows if r["class_id"] == cls], f"correct_at_{suffix}")
                              for suffix in ("iou50", "iou75")}}
                          for cls, name in rq2_detector_training.NAMES.items()],
            "ece_definition": "10 equal-width bins [0,0.1),...,[0.9,1]; prediction-weighted absolute confidence/accuracy gap",
            "nll_definition": "Bernoulli correctness NLL, natural log; probabilities clipped only for NLL",
            "nll_epsilon": NLL_EPSILON}


def read_predictions(root):
    """Verify Confidence distribution completion, pinned identities and input table hashes."""
    images = raw.development_images(root)
    directory = resolve_input(root, raw.OUTPUT)
    provenance = rq1._read_json(directory / "provenance.json")
    expected = {"status": "complete", "development_manifest": raw.DEVELOPMENT,
                "manifest_sha256_lf": raw.MANIFEST_SHA, "development_images": raw.IMAGE_COUNT,
                "checkpoint": raw.CHECKPOINT, "checkpoint_epoch": raw.EPOCH,
                "checkpoint_sha256": raw.CHECKPOINT_SHA}
    if any((canonical_identifier(provenance.get(k)) if k in ("development_manifest", "checkpoint")
            else provenance.get(k)) != v for k, v in expected.items()):
        raise ValueError("Confidence distribution provenance is incomplete or not the frozen development extraction")
    if provenance.get("checkpoint_identity", {}).get("sha256") != raw.CHECKPOINT_SHA:
        raise ValueError("Confidence distribution checkpoint identity mismatch")
    fingerprints = {"provenance.json": rq1._sha(directory / "provenance.json")}
    for name in ("predictions.csv", "images.csv"):
        fingerprints[name] = rq1._sha(directory / name)
        if provenance.get("output_sha256", {}).get(name) != fingerprints[name]:
            raise ValueError(f"Confidence distribution artifact changed: {name}")
    counts = dict.fromkeys(images, 0)
    rows = []
    with (directory / "predictions.csv").open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(raw.FIELDS):
            raise ValueError("Unexpected Confidence distribution prediction columns")
        for record in reader:
            row = {**record, "class_id": int(record["class_id"]),
                   **{key: float(record[key]) for key in ("confidence", *BOX_KEYS)}}
            raw.validate_row(row, counts)
            counts[row["image"]] += 1
            rows.append(row)
    with (directory / "images.csv").open(encoding="utf-8", newline="") as file:
        inventory = list(csv.DictReader(file))
    if ([r["image"] for r in inventory] != images
            or any(int(r["prediction_count"]) != counts[r["image"]] for r in inventory)):
        raise ValueError("Confidence distribution image inventory/coverage differs")
    return images, rows, fingerprints


def read_ground_truth(dataset, images):
    """DICC: open only development image headers and corresponding YOLO text labels."""
    from PIL import Image
    dataset = Path(dataset).resolve()
    truth, evidence = {}, {}
    for image in images:
        relative = PurePosixPath("labels", *PurePosixPath(image).parts[1:]).with_suffix(".txt")
        image_path, label_path = (dataset / image).resolve(), (dataset / relative).resolve()
        if not image_path.is_relative_to(dataset) or not label_path.is_relative_to(dataset):
            raise ValueError("Development path escapes dataset_root")
        with Image.open(image_path) as opened:
            if opened.getexif().get(274, 1) != 1:
                raise ValueError(f"EXIF-oriented image requires explicit coordinate review: {image}")
            width, height = opened.size
        raw_labels = label_path.read_bytes()  # missing labels fail; empty file is valid
        truth[image] = []
        for number, line in enumerate(raw_labels.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Expected YOLO detection label at {relative}:{number}")
            cls = int(fields[0])
            x, y, w, h = map(float, fields[1:])
            if (cls not in rq2_detector_training.NAMES or not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h))
                    or w <= 0 or h <= 0):
                raise ValueError(f"Invalid ground-truth label: {relative}:{number}")
            truth[image].append({"id": f"{relative.as_posix()}:{number}", "class_id": cls,
                                 "box": ((x - w / 2) * width, (y - h / 2) * height,
                                         (x + w / 2) * width, (y + h / 2) * height)})
        evidence[image] = {"label": relative.as_posix(), "label_sha256": hashlib.sha256(raw_labels).hexdigest(),
                           "width": width, "height": height, "objects": len(truth[image])}
    return truth, evidence


def plot_diagnostics(rows, report, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    bins = [b for b in report["primary_iou50"]["reliability_bins"] if b["count"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot([b["mean_confidence"] for b in bins], [b["correctness_rate"] for b in bins], "o-", label="Raw predictions")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean raw confidence", ylabel="Correctness rate (IoU ≥ 0.50)",
           title="Development reliability: 10 equal-width bins; confidence ≥ 0.01")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "raw_reliability_iou50.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([r["confidence"] for r in rows], bins=[i / 100 for i in range(101)])
    ax.set(xlim=(0, 1), xlabel="Raw confidence", ylabel="Retained prediction count",
           title="Development confidence ≥ 0.01")
    fig.tight_layout()
    fig.savefig(output / "retained_confidence_histogram.png", dpi=160)
    plt.close(fig)


def analyze(root, dataset, truth_reader=read_ground_truth, plotter=plot_diagnostics):
    root = Path(root).resolve()
    historical = root / HISTORICAL_OUTPUT
    if historical.exists() or historical.with_name(historical.name + ".provenance.json").exists():
        raise FileExistsError("Historical calibration output exists; do not rerun or overwrite it")
    reject_historical_output(root, OUTPUT)
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution/scientific inputs before DICC analysis")
    images, predictions, hashes = read_predictions(root)
    versions = {name: importlib.metadata.version(name) for name in ("Pillow", "matplotlib")}
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"experiment": "rq2_calibration_analysis", "status": "running", **git, "inclusion_threshold": INCLUSION,
                  "primary_iou": 0.50, "sensitivity_iou": 0.75, "checkpoint_epoch": raw.EPOCH,
                  "checkpoint_sha256": raw.CHECKPOINT_SHA, "manifest_sha256_lf": raw.MANIFEST_SHA,
                  "confidence_distribution_input_sha256": hashes, "versions": versions,
                  "started_utc": datetime.now(timezone.utc).isoformat()}
    rq1._write_json(output / "provenance.json", provenance, exclusive=True)
    try:
        truth, evidence = truth_reader(dataset, images)
        rows = label_predictions(images, predictions, truth)
        report = summarize(rows)
        report.update(development_images=len(images), extracted_predictions=len(predictions),
                      excluded_below_inclusion=len(predictions) - len(rows))
        with (output / "labelled_predictions.csv").open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        rq1._write_json(output / "summary.json", report, exclusive=True)
        rq1._write_json(output / "ground_truth_provenance.json", evidence, exclusive=True)
        plotter(rows, report, output)
        if read_predictions(root)[2] != hashes or rq1.git_provenance(root) != git:
            raise RuntimeError("Confidence distribution inputs or source state changed during analysis")
        provenance.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                          output_sha256={p.name: rq1._sha(p) for p in output.iterdir() if p.name != "provenance.json"})
        rq1._write_json(output / "provenance.json", provenance)
        return output
    except BaseException:
        provenance["status"] = "incomplete"
        rq1._write_json(output / "provenance.json", provenance)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("analyze",))
    parser.add_argument("--dicc", action="store_true")
    args = parser.parse_args(argv)
    if os.name == "nt" or not args.dicc:
        raise RuntimeError("Real development analysis is DICC-only; --dicc required")
    root = find_project_root()
    paths = load_paths(root)
    if paths.project_root != root:
        raise ValueError("project_root must match checkout")
    print(analyze(root, paths.dataset_root))


if __name__ == "__main__":
    main()
