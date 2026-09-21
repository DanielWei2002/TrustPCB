"""Synthetic Data partition logic tests only; no real dataset labels or images."""

import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustpcb import rq2_data_partition as split


class RQ2SplitTests(unittest.TestCase):
    def test_imbalanced_class_frequencies_keep_ten_percent_shares(self):
        source, counts = [], {}
        for c, size in enumerate((800, 200, 100, 100)):
            for i in range(size):
                name = f"images/train/c{c}_{i:04d}.jpg"
                source.append(name)
                vector = [0] * 4
                vector[c] = 1 if i % 2 else 5
                counts[name] = vector
        train, dev, groups = split.partition(source, [], counts, target_dev=120)
        split.validate(source, train, dev, groups)
        self.assertEqual((len(train), len(dev)), (1080, 120))
        balance = split.class_balance(source, dev, counts, dict(enumerate("ABCD")))
        for row in balance["classes"]:
            self.assertLessEqual(row["absolute_image_deviation_pp"], 1.0)
            self.assertLessEqual(row["absolute_annotation_deviation_pp"], 1.0)

    def test_multilabel_swaps_improve_greedy_allocation(self):
        rng = random.Random(93)  # Fixed synthetic fixture, not an experimental split seed.
        source = [str(i) for i in range(100)]
        counts = {name: [rng.randrange(4) if rng.random() < 0.5 else 0 for _ in range(3)]
                  for name in source}
        diagnostics = {}
        train, dev, groups = split.partition(source, [], counts, target_dev=10, diagnostics=diagnostics)
        self.assertEqual((len(train), len(dev)), (90, 10))
        self.assertGreater(diagnostics["accepted_swaps"], 0)
        self.assertLess(diagnostics["final_mean_squared_deviation_pp2"],
                        diagnostics["initial_mean_squared_deviation_pp2"])
        history = diagnostics["balance_error_history_pp2"]
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(history, history[1:])))
        self.assertTrue(diagnostics["local_optimum_reached"])
        self.assertEqual(split.partition(source, [], counts, target_dev=10), (train, dev, groups))

    def test_balance_report_percentages_and_zero_count_policy(self):
        source = ["a", "b", "c", "d"]
        counts = {"a": [3, 0], "b": [1, 0], "c": [0, 0], "d": [0, 0]}
        report = split.class_balance(source, ["a"], counts, {0: "A", 1: "absent"})
        a, absent = report["classes"]
        self.assertEqual(a["source_image_count"], 2)
        self.assertEqual(a["dev_image_count"], 1)
        self.assertEqual(a["dev_image_percentage_of_source"], 50)
        self.assertEqual(a["dev_annotation_percentage_of_source"], 75)
        self.assertEqual(a["absolute_image_deviation_pp"], 40)
        self.assertEqual(a["absolute_annotation_deviation_pp"], 65)
        self.assertIsNone(absent["dev_image_percentage_of_source"])
        self.assertIsNone(absent["absolute_annotation_deviation_pp"])
        self.assertEqual(report["summary"]["mean_squared_deviation_pp2"], (40**2 + 65**2) / 2)

    def test_rejected_split_is_preserved_and_compared_using_same_error(self):
        source = [f"images/train/{i:03d}.jpg" for i in range(200)]
        counts = {name: ([1, 0] if i < 100 else [0, 3]) for i, name in enumerate(source)}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rejected = root / split.REJECTED_OUTPUT
            rejected.mkdir(parents=True)
            payloads = {"detector_train_images.txt": "\n".join(source[20:]) + "\n",
                        "development_calibration_images.txt": "\n".join(source[:20]) + "\n",
                        "verification_report.json": '{"status": "original"}\n'}
            for name, contents in payloads.items():
                (rejected / name).write_bytes(contents.encode())
            report = split.freeze(root, source, [], {0: "A", 1: "B"}, counts, {}, "labels", "git", target_dev=20)
            self.assertEqual(report["counts"]["detector_train"]["images"], 180)
            self.assertEqual(report["counts"]["development_calibration"]["images"], 20)
            comparison = report["rejected_split_comparison"]
            self.assertGreater(comparison["mean_squared_error_reduction_pp2"], 0)
            self.assertEqual(report["class_balance"]["summary"]["mean_squared_deviation_pp2"], 0)
            for name, contents in payloads.items():
                self.assertEqual((rejected / name).read_bytes(), contents.encode())
            self.assertTrue((root / split.OUTPUT / "verification_report.json").is_file())

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
