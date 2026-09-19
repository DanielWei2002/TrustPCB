"""RQ2 Stage 1 only: freeze a group-preserving train/development split."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
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
OUTPUT = "data/splits/rq2_stage1"


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


def partition(names, edges, counts, target_dev=TARGET_DEV, seed=SEED):
    """Bounded greedy class balancing; no image access and no group splitting.

    Seeded ties and canonical components make input-edge ordering irrelevant.
    Rare-class groups are considered first. Whole groups are added when they
    reduce normalized squared class/image-count deficits. A final size pass
    approaches the target only when whole-group additions improve the distance.
    """
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
    target = [v * target_dev / len(names) for v in total]
    active = [i for i, value in enumerate(target) if i and value > 0]

    def score(vector):
        class_error = sum(((vector[i] - target[i]) / max(target[i], 1)) ** 2 for i in active)
        return 4 * ((vector[0] - target_dev) / target_dev) ** 2 + class_error / max(len(active), 1)

    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    rank = {index: i for i, index in enumerate(order)}
    order.sort(key=lambda i: (min((total[1 + c] for c in range(classes)
                                  if vectors[i][1 + c]), default=float("inf")), rank[i]))
    selected = set()
    current = [0] * len(total)
    for index in order:
        candidate = [a + b for a, b in zip(current, vectors[index])]
        if candidate[0] <= target_dev and score(candidate) < score(current):
            selected.add(index)
            current = candidate
    while current[0] != target_dev:
        options = []
        for index in order:
            if index in selected:
                continue
            candidate = [a + b for a, b in zip(current, vectors[index])]
            distance = abs(candidate[0] - target_dev)
            if candidate[0] < len(names) and distance < abs(current[0] - target_dev):
                options.append((distance, score(candidate), rank[index], index, candidate))
        if not options:
            break
        _, _, _, index, current = min(options)
        selected.add(index)
    dev = {name for i in selected for name in groups[i]}
    train = [name for name in names if name not in dev]
    development = [name for name in names if name in dev]
    validate(names, train, development, groups)
    return train, development, groups


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
    train, dev, groups = partition(source, edges, counts, target_dev)
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
        "protocol": "rq2_stage1_v1", "algorithm": "seeded rare-class-first whole-group greedy; normalized image/annotation deficits; whole-group size correction",
        "split_seed": SEED, "target_development_images": target_dev,
        "development_count_deviation": len(dev) - target_dev,
        "ordering": "source manifest subsequence, unchanged relative path spelling",
        "git_commit": commit, "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": inputs, "source_label_inventory_sha256": label_hash,
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "manifest_sha256": {name: sha(data) for name, data in payloads.items()},
        "counts": statistics,
        "integrity": {"all_source_images_exactly_once": True, "no_duplicate_membership": True,
                      "no_partition_overlap": True, "no_confirmed_group_crosses_partitions": True,
                      "reserved_test_labels_or_images_read": False,
                      "source_connected_components": len(groups),
                      "source_nonsingleton_components": sum(len(g) > 1 for g in groups)},
        "scope": "Frozen Stage 1 membership only; no model training, tuning or evaluation",
    }
    payloads["verification_report.json"] = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    # Complete validation precedes output creation. Never overwrite a frozen split.
    destination = Path(root) / OUTPUT
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
