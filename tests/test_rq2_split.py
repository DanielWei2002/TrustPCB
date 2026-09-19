"""Synthetic Stage 1 logic tests only; no real dataset labels or images."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustpcb import rq2_split as split


class RQ2SplitTests(unittest.TestCase):
    def test_8207_synthetic_source_coverage_and_reproduction(self):
        source = [f"images/train/synthetic_{i:05d}.jpg" for i in range(8207)]
        counts = {name: [1, 0] for name in source}
        edges = [(source[0], source[1]), (source[1], source[2]), (source[7], source[8])]
        first = split.partition(source, edges, counts, seed=24209199)
        second = split.partition(source, list(reversed(edges)), counts, seed=24209199)
        self.assertEqual(first, second)
        train, dev, groups = first
        self.assertEqual((len(train), len(dev)), (7386, 821))
        self.assertEqual(len(set(train + dev)), 8207)
        self.assertFalse(set(train) & set(dev))
        self.assertEqual(set(train + dev), set(source))
        self.assertIn(source[:3], groups)
        split.validate(source, train, dev, groups)
        train_set, dev_set = set(train), set(dev)
        self.assertEqual(train, [n for n in source if n in train_set])
        self.assertEqual(dev, [n for n in source if n in dev_set])

    def test_connected_groups_override_exact_target(self):
        names = [str(i) for i in range(8)]
        edges = [(names[i], names[i + 1]) for i in (0, 1, 2, 4, 5, 6)]
        train, dev, groups = split.partition(names, edges, {n: [1] for n in names}, target_dev=3)
        self.assertEqual((len(train), len(dev)), (4, 4))
        split.validate(names, train, dev, groups)

    def test_input_duplicates_and_incomplete_counts_rejected(self):
        with self.assertRaises(ValueError):
            split.partition(["a", "a", "b"], [], {"a": [1], "b": [1]}, target_dev=1)
        with self.assertRaises(ValueError):
            split.partition(["a", "b"], [], {"a": [1]}, target_dev=1)
        with self.assertRaises(ValueError):
            split.components(["a", "b"], [("a", "missing")])

    def test_validation_rejects_overlap_missing_and_split_groups(self):
        source = ["a", "b", "c", "d"]
        for train, dev in ((["a", "b", "c"], ["c", "d"]), (["a"], ["c", "d"]),
                           (["a", "c"], ["b", "d"])):
            with self.subTest(train=train), self.assertRaises(ValueError):
                split.validate(source, train, dev, [["a", "b"], ["c"], ["d"]])

    def test_source_labels_only_and_original_physical_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "labels/val"
            folder.mkdir(parents=True)
            (folder / "moved.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n")
            (folder / "reserved.txt").write_text("MUST NOT BE READ")
            counts, digest = split.count_source_labels(["images/val/moved.jpg"], root, classes=2)
            self.assertEqual(counts, {"images/val/moved.jpg": [2, 1]})
            self.assertEqual(len(digest), 64)

    def test_manifest_bytes_hashes_counts_and_freeze_protection(self):
        source = [f"images/train/{i:02d}.jpg" for i in range(30)]
        counts = {name: [1, i % 3] for i, name in enumerate(source)}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reports = []
            for name in ("first", "second"):
                report = split.freeze(root / name, source, [], {0: "A", 1: "B"}, counts,
                                      {"synthetic_manifest": "fixture"}, "label_fixture", "git_fixture", target_dev=3)
                reports.append(report)
                output = root / name / split.OUTPUT
                for file, digest in report["manifest_sha256"].items():
                    data = (output / file).read_bytes()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
                    self.assertNotIn(b"\r", data)
                self.assertEqual(json.loads((output / "verification_report.json").read_text())["split_seed"], 24209199)
                statistics = report["counts"]
                self.assertEqual(statistics["source"]["images"], 30)
                for c in range(2):
                    for metric in ("image_count", "annotation_count"):
                        self.assertEqual(statistics["source"]["classes"][c][metric],
                                         statistics["detector_train"]["classes"][c][metric] +
                                         statistics["development_calibration"]["classes"][c][metric])
            self.assertEqual(reports[0]["manifest_sha256"], reports[1]["manifest_sha256"])
            self.assertEqual(reports[0]["counts"], reports[1]["counts"])
            with self.assertRaises(FileExistsError):
                split.freeze(root / "first", source, [], {0: "A", 1: "B"}, counts, {}, "", "", target_dev=3)


if __name__ == "__main__":
    unittest.main()
