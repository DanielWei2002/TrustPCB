"""RQ2 Stage 1 only: freeze a group-preserving train/development split."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import subprocess

import yaml

from trustpcb.dataset_config import find_project_root, load_paths

SEED = 24209199
TARGET_DEV = 821
SOURCE = "configs/datasets/similarity_aware_train_images.txt"
RESERVED = "configs/datasets/similarity_aware_val_images.txt"
PAIRS = "outputs/tables/cross_split_high_confidence_near_duplicates.csv"
RECORDS = "data/splits/similarity_aware_split_manifest.csv"
REJECTED_OUTPUT = "data/splits/rq2_stage1"
OUTPUT = "data/splits/rq2_stage1_v2"
TARGET_FRACTION = 0.10
MAX_SWAPS = 200


def sha(data):
    return hashlib.sha256(data).hexdigest()


def components(names, edges):
    """Canonical connected components, including singleton images."""
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for left, right in edges:
        if left not in parent or right not in parent:
            raise ValueError(f"Unknown confirmed similarity endpoint: {left}, {right}")
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)
    groups = {}
    for name in sorted(parent):
        groups.setdefault(find(name), []).append(name)
    return sorted(groups.values(), key=lambda group: tuple(group))


def _vector(group, counts, classes):
    return [len(group)] + [sum(counts[n][c] > 0 for n in group) for c in range(classes)] + [
        sum(counts[n][c] for n in group) for c in range(classes)]


def _balance_score(vector, total):
    """Mean squared percentage-point deviation from 10%, equal per feature."""
    errors = [(100 * vector[i] / total[i] - 100 * TARGET_FRACTION) ** 2
              for i in range(1, len(total)) if total[i] > 0]
    return sum(errors) / len(errors) if errors else 0.0


def _allocate(groups, vectors, total, target_dev, seed, max_swaps, diagnostics):
    # Components with identical feature vectors are interchangeable for scoring.
    # Bucketing avoids scanning thousands of equivalent singleton candidates.
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    buckets = {}
    for index in order:
        buckets.setdefault(tuple(vectors[index]), []).append(index)
    features, members = list(buckets), list(buckets.values())
    taken = [0] * len(features)
    current = [0] * len(total)

    # Global best feasible addition at every step, not a rare-class-first order.
    while current[0] < target_dev:
        best = None
        for i, vector in enumerate(features):
            if taken[i] == len(members[i]) or current[0] + vector[0] > target_dev:
                continue
            candidate = [a + b for a, b in zip(current, vector)]
            option = (_balance_score(candidate, total), i, candidate)
            if best is None or option[:2] < best[:2]:
                best = option
        if best is None:
            break
        _, index, current = best
        taken[index] += 1

    # If the target is not reachable by these additions, preserve atomic groups
    # and allow a closer overshoot. Do not sacrifice an exact size for balance.
    for i, vector in enumerate(features):
        if taken[i] < len(members[i]) and current[0] + vector[0] < total[0]:
            if abs(current[0] + vector[0] - target_dev) < abs(current[0] - target_dev):
                current = [a + b for a, b in zip(current, vector)]
                taken[i] += 1

    initial_error = _balance_score(current, total)
    history = [initial_error]
    swaps = 0
    converged = False
    # Steepest deterministic whole-component exchange; size distance cannot grow.
    # Each accepted exchange strictly improves balance (or improves size while
    # preserving balance). A bounded search reports explicitly if it hits its cap.
    for _ in range(max_swaps):
        current_error = _balance_score(current, total)
        current_distance = abs(current[0] - target_dev)
        best = None
        for outgoing, remove in enumerate(features):
            if not taken[outgoing]:
                continue
            for incoming, add in enumerate(features):
                if outgoing == incoming or taken[incoming] == len(members[incoming]):
                    continue
                count = current[0] - remove[0] + add[0]
                distance = abs(count - target_dev)
                if not 0 < count < total[0] or distance > current_distance:
                    continue
                candidate = [a - b + c for a, b, c in zip(current, remove, add)]
                error = _balance_score(candidate, total)
                if error > current_error + 1e-12:
                    continue
                if not (error < current_error - 1e-12 or distance < current_distance):
                    continue
                option = (distance, error, outgoing, incoming, candidate)
                if best is None or option[:4] < best[:4]:
                    best = option
        if best is None:
            converged = True
            break
        _, error, outgoing, incoming, current = best
        taken[outgoing] -= 1
        taken[incoming] += 1
        swaps += 1
        history.append(error)
    diagnostics.update(initial_mean_squared_deviation_pp2=initial_error,
                       final_mean_squared_deviation_pp2=_balance_score(current, total),
                       accepted_swaps=swaps, max_swaps=max_swaps,
                       local_optimum_reached=converged,
                       improvement_limit_reached=not converged,
                       balance_error_history_pp2=history,
                       distinct_component_feature_vectors=len(features))
    return {index for i, group_ids in enumerate(members) for index in group_ids[:taken[i]]}


def partition(names, edges, counts, target_dev=TARGET_DEV, seed=SEED, diagnostics=None, max_swaps=MAX_SWAPS):
    """Deterministic group-aware multilabel allocation using counts only."""
    if len(names) != len(set(names)) or set(counts) != set(names):
        raise ValueError("Source/count identities must be unique and match exactly")
    if not 0 < target_dev < len(names):
        raise ValueError("Development target must leave both partitions nonempty")
    classes = len(next(iter(counts.values())))
    if not classes or any(len(v) != classes or any(type(x) is not int or x < 0 for x in v)
                          for v in counts.values()):
        raise ValueError("Annotation counts must be nonnegative integer vectors")
    groups = components(names, edges)
    vectors = [_vector(g, counts, classes) for g in groups]
    total = _vector(names, counts, classes)
    if max_swaps < 0:
        raise ValueError("max_swaps must be nonnegative")
    selected = _allocate(groups, vectors, total, target_dev, seed, max_swaps,
                         diagnostics if diagnostics is not None else {})
    dev = {name for i in selected for name in groups[i]}
    train = [name for name in names if name not in dev]
    development = [name for name in names if name in dev]
    validate(names, train, development, groups)
    return train, development, groups


def class_balance(source, dev, counts, names):
    """Comparable class-balance report, independent of any model performance."""
    total, selected = _vector(source, counts, len(names)), _vector(dev, counts, len(names))
    rows, deviations = [], []
    for c in range(len(names)):
        row = {"class_id": c, "name": names[c]}
        for feature, index in (("image", 1 + c), ("annotation", 1 + len(names) + c)):
            percentage = 100 * selected[index] / total[index] if total[index] else None
            deviation = abs(percentage - 100 * TARGET_FRACTION) if percentage is not None else None
            row.update({f"source_{feature}_count": total[index], f"dev_{feature}_count": selected[index],
                        f"dev_{feature}_percentage_of_source": percentage,
                        f"absolute_{feature}_deviation_pp": deviation})
            if deviation is not None:
                deviations.append(deviation)
        rows.append(row)
    mse = sum(d * d for d in deviations) / len(deviations) if deviations else 0.0
    return {"target_percentage_of_source": 100 * TARGET_FRACTION,
            "dev_images": len(dev), "dev_image_percentage_of_source": 100 * len(dev) / len(source),
            "classes": rows, "summary": {
                "mean_absolute_deviation_pp": sum(deviations) / len(deviations) if deviations else 0.0,
                "root_mean_squared_deviation_pp": math.sqrt(mse),
                "mean_squared_deviation_pp2": mse, "max_absolute_deviation_pp": max(deviations, default=0.0),
                "active_class_features": len(deviations),
                "zero_source_count_policy": "percentage/deviation null; excluded from error"}}


def validate(source, train, dev, groups):
    source_set, train_set, dev_set = set(source), set(train), set(dev)
    if not train or not dev:
        raise ValueError("Cannot form two nonempty partitions without breaking groups")
    if len(source) != len(source_set) or len(train) != len(train_set) or len(dev) != len(dev_set):
        raise ValueError("Duplicate membership")
    if train_set & dev_set or train_set | dev_set != source_set:
        raise ValueError("Partitions must cover source exactly once without overlap")
    if any(set(g) & train_set and set(g) & dev_set for g in groups):
        raise ValueError("Confirmed similarity component crosses partitions")


def load_inputs(root):
    root = Path(root)
    inputs = {}

    def read(name):
        data = (root / name).read_bytes()
        inputs[name] = sha(data)
        return data.decode("utf-8-sig")

    source, reserved = read(SOURCE).splitlines(), read(RESERVED).splitlines()
    for paths, expected in ((source, 8207), (reserved, 2052)):
        if len(paths) != expected or len(set(paths)) != expected:
            raise ValueError(f"Expected {expected} unique manifest entries")
        for entry in paths:
            parts = PurePosixPath(entry).parts
            if len(parts) != 3 or parts[0] != "images" or parts[1] not in ("train", "val") or entry != "/".join(parts) or "\\" in entry or parts[2] in (".", ".."):
                raise ValueError(f"Invalid portable image path: {entry!r}")
    if set(source) & set(reserved):
        raise ValueError("Source and reserved test membership overlap")
    lookup = {PurePosixPath(path).name: path for path in source + reserved}
    if len(lookup) != 10259:
        raise ValueError("Image filenames are not unique across source and reserved membership")
    for split, paths in (("train", source), ("val", reserved)):
        names = read(f"data/splits/similarity_aware_{split}.txt").splitlines()
        if names != [PurePosixPath(p).name for p in paths]:
            raise ValueError("Scientific filename manifest order differs from portable manifest")
    rows = list(csv.DictReader(io.StringIO(read(RECORDS))))
    if len(rows) != 10259 or {r["image"] for r in rows} != set(lookup):
        raise ValueError("Scientific split-record identities differ")
    source_set = set(source)
    for row in rows:
        path = lookup[row["image"]]
        if row["similarity_aware_split"] != ("train" if path in source_set else "val") or row["supplied_split"] != PurePosixPath(path).parts[1]:
            raise ValueError("Scientific membership/storage record mismatch")
    edges = [(lookup[r["train_file"]], lookup[r["val_file"]])
             for r in csv.DictReader(io.StringIO(read(PAIRS)))]
    all_groups = components(source + reserved, edges)
    if any(set(g) & source_set and not set(g) <= source_set for g in all_groups):
        raise ValueError("Confirmed component crosses existing source/reserved-test boundary")
    source_edges = [(a, b) for a, b in edges if a in source_set]
    config = yaml.safe_load(read("configs/datasets/dspcbsd_similarity_aware.yaml"))
    names = config["names"]
    if set(names) != set(range(9)):
        raise ValueError("Expected the existing nine-class mapping")
    return source, source_edges, names, inputs


def count_source_labels(source, dataset_root, classes=9):
    """DICC generation: read labels referenced by source only, never test labels."""
    counts = {}
    digest = hashlib.sha256()
    for image in source:
        parts = PurePosixPath(image).parts
        relative = PurePosixPath("labels", *parts[1:]).with_suffix(".txt")
        data = (Path(dataset_root) / Path(*relative.parts)).read_bytes()
        digest.update(relative.as_posix().encode() + b"\0" + hashlib.sha256(data).digest())
        vector = [0] * classes
        for line in data.decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            fields = line.split()
            value = float(fields[0])
            if len(fields) != 5 or not value.is_integer() or not 0 <= value < classes:
                raise ValueError(f"Invalid YOLO annotation in {relative}")
            vector[int(value)] += 1
        counts[image] = vector
    return counts, digest.hexdigest()


def freeze(root, source, edges, names, counts, inputs, label_hash, commit, target_dev=TARGET_DEV):
    destination = Path(root) / OUTPUT
    if destination.exists():
        raise FileExistsError(f"Frozen revision already exists: {OUTPUT}; no overwrite allowed")
    diagnostics = {}
    train, dev, groups = partition(source, edges, counts, target_dev, diagnostics=diagnostics)
    payloads = {
        "detector_train_images.txt": ("\n".join(train) + "\n").encode(),
        "development_calibration_images.txt": ("\n".join(dev) + "\n").encode(),
    }
    statistics = {}
    for label, entries in (("source", source), ("detector_train", train), ("development_calibration", dev)):
        vector = _vector(entries, counts, len(names))
        statistics[label] = {"images": len(entries), "classes": [
            {"class_id": c, "name": names[c], "image_count": vector[1 + c],
             "annotation_count": vector[1 + len(names) + c]} for c in range(len(names))]}
    report = {
        "protocol": "rq2_stage1_v2", "algorithm": "global whole-component greedy plus deterministic size-preserving/improving component exchanges",
        "split_seed": SEED, "target_development_images": target_dev,
        "development_count_deviation": len(dev) - target_dev,
        "ordering": "source manifest subsequence, unchanged relative path spelling",
        "git_commit": commit, "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": inputs, "source_label_inventory_sha256": label_hash,
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "manifest_sha256": {name: sha(data) for name, data in payloads.items()},
        "counts": statistics,
        "class_balance": class_balance(source, dev, counts, names),
        "optimisation": diagnostics,
        "integrity": {"all_source_images_exactly_once": True, "no_duplicate_membership": True,
                      "no_partition_overlap": True, "no_confirmed_group_crosses_partitions": True,
                      "reserved_test_labels_or_images_read": False,
                      "source_connected_components": len(groups),
                      "source_nonsingleton_components": sum(len(g) > 1 for g in groups)},
        "scope": "Frozen Stage 1 membership only; no model training, tuning or evaluation",
    }
    rejected = Path(root) / REJECTED_OUTPUT
    if rejected.exists():
        old_train = (rejected / "detector_train_images.txt").read_bytes()
        old_dev = (rejected / "development_calibration_images.txt").read_bytes()
        old_train_names, old_dev_names = old_train.decode().splitlines(), old_dev.decode().splitlines()
        validate(source, old_train_names, old_dev_names, groups)
        previous_balance = class_balance(source, old_dev_names, counts, names)
        report["rejected_split_comparison"] = {
            "identity": REJECTED_OUTPUT, "status": "rejected; preserved unchanged",
            "basis": "both partitions scored against the same current source-label inventory",
            "manifest_sha256": {"detector_train_images.txt": sha(old_train),
                                "development_calibration_images.txt": sha(old_dev)},
            "class_balance": previous_balance,
            "mean_squared_error_reduction_pp2": previous_balance["summary"]["mean_squared_deviation_pp2"]
                - report["class_balance"]["summary"]["mean_squared_deviation_pp2"],
        }
    else:
        report["rejected_split_comparison"] = {"status": "unavailable; rejected manifests not present"}
    payloads["verification_report.json"] = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    # Complete validation precedes output creation. Never overwrite a frozen split.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    for name, data in payloads.items():
        (destination / name).write_bytes(data)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicc", action="store_true", help="Authorize source-label processing on DICC only")
    args = parser.parse_args(argv)
    if os.name == "nt" or not args.dicc:
        parser.error("Real split generation is DICC-only. Local verification must use synthetic fixtures.")
    root = find_project_root()
    paths = load_paths(root)
    if paths.project_root != root:
        raise ValueError("Configured project_root must identify this checkout")
    if (root / OUTPUT).exists():
        raise FileExistsError(f"Frozen output already exists: {OUTPUT}; inspect it rather than overwriting")
    source, edges, names, inputs = load_inputs(root)
    counts, label_hash = count_source_labels(source, paths.dataset_root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    report = freeze(root, source, edges, names, counts, inputs, label_hash, commit)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
