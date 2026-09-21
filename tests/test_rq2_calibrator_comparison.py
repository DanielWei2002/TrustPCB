"""Small synthetic fixtures only; never scientific development data or final fitting."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trustpcb import rq2_calibrator_comparison as comparison


def synthetic_population(n=10):
    images = [f"images/train/synthetic_{i}.jpg" for i in range(n)]
    # Mixed labels at each raw score ensure a finite, nonseparable tiny fit.
    rows = [{"image": image, "prediction_id": f"{image}#{j}", "raw_confidence": p,
             "correct_iou50": y}
            for image in images for j, (p, y) in enumerate(((.2, 0), (.2, 1), (.8, 0), (.8, 1)), 1)]
    return images, rows


class FoldAndFitTests(unittest.TestCase):
    def test_deterministic_balanced_complete_grouped_folds(self):
        images, _ = synthetic_population(20)
        edges = [(images[0], images[1]), (images[1], images[2])]
        folds = comparison.assign_folds(images, edges)
        self.assertEqual(folds, comparison.assign_folds(images, list(reversed(edges))))
        self.assertEqual([r["image"] for r in folds], images)
        self.assertEqual(len({r["image"] for r in folds}), len(images))
        lookup = {r["image"]: r["fold_id"] for r in folds}
        self.assertEqual(len({lookup[i] for i in images[:3]}), 1)
        self.assertEqual([sum(r["fold_id"] == f for r in folds) for f in range(5)], [4] * 5)

    def test_folds_do_not_depend_on_correctness_or_prediction_counts(self):
        images, rows = synthetic_population()
        before = comparison.assign_folds(images, [])
        for row in rows:
            row["correct_iou50"] = 1 - row["correct_iou50"]
            row["correct_at_iou75"] = "ignored"
        rows.extend(rows[:5])
        self.assertEqual(before, comparison.assign_folds(images, []))

    def test_cross_fit_every_prediction_once_same_folds_all_methods(self):
        images, rows = synthetic_population()
        folds = comparison.assign_folds(images, [])
        records = []
        def spy(method, p, y):
            records.append((method, len(p), y.copy()))
            return comparison.fit_calibrator(method, p, y)
        values, row_folds, fits = comparison.cross_fit(images, rows, folds, spy)
        self.assertEqual(len(fits), 20)
        self.assertEqual(len(records), 20)
        for method in comparison.METHODS:
            self.assertEqual(values[method].shape, (len(rows),))
            self.assertTrue(np.isfinite(values[method]).all())
        for image in images:
            indices = [i for i, r in enumerate(rows) if r["image"] == image]
            self.assertEqual(len(set(row_folds[indices])), 1)
        for fold in range(5):
            items = fits[fold * 4:fold * 4 + 4]
            self.assertEqual({r["held_out_predictions"] for r in items}, {8})
            self.assertEqual({r["training_predictions"] for r in items}, {32})

    def test_held_out_labels_never_enter_fit(self):
        images, rows = synthetic_population()
        folds = comparison.assign_folds(images, [])
        lookup = {r["image"]: r["fold_id"] for r in folds}
        calls = []
        def spy(method, p, y):
            fold = len(calls) // 4
            expected = [r["correct_iou50"] for r in rows if lookup[r["image"]] != fold]
            self.assertEqual(y.tolist(), expected)
            calls.append(method)
            return {"method": "Isotonic", "x": [0, 1], "y": [0, 1]}
        comparison.cross_fit(images, rows, folds, spy)

    def test_temperature_mapping_positive_monotonic(self):
        self.check_monotonic("Temperature")

    def test_platt_mapping_positive_monotonic(self):
        self.check_monotonic("Platt")

    def test_beta_mapping_constraints_monotonic(self):
        self.check_monotonic("Beta")

    def test_isotonic_nondecreasing_support_and_ties(self):
        self.check_monotonic("Isotonic")
        model = comparison.fit_calibrator("Isotonic", [.2, .2, .5, .8], [0, 1, 0, 1])
        p = comparison.apply_calibrator(model, [0, .2, .5, .8, 1])
        self.assertEqual(p[0], p[1])
        self.assertEqual(p[-1], p[-2])
        np.testing.assert_allclose(p[:3], [1 / 3] * 3)

    def check_monotonic(self, method):
        p = [.1, .1, .3, .3, .6, .6, .9, .9]
        y = [0, 1, 0, 0, 0, 1, 1, 1]
        model = comparison.fit_calibrator(method, p, y)
        values = comparison.apply_calibrator(model, np.linspace(0, 1, 51))
        self.assertTrue(np.all(np.diff(values) >= 0))
        self.assertTrue(np.all((values >= 0) & (values <= 1)))
        if method in ("Temperature", "Platt"):
            self.assertGreater(model["parameters"][0], 0)
        if method == "Beta":
            self.assertTrue(all(v >= 0 for v in model["parameters"][:2]))

    def test_single_class_fold_fails_without_fallback(self):
        for method in comparison.METHODS:
            with self.assertRaises(ValueError):
                comparison.fit_calibrator(method, [.1, .9], [1, 1])


class MetricsAndBootstrapTests(unittest.TestCase):
    def test_nll_brier_ece_match_existing_definition(self):
        p, y = [.01, .1, .3, .8, 1.], [0, 1, 0, 1, 0]
        result = comparison.metric_summary(p, y)
        existing = comparison.analysis.metrics([{"confidence": c, "correct": target} for c, target in zip(p, y)], "correct")
        self.assertAlmostEqual(result["nll"], existing["negative_log_likelihood"])
        self.assertAlmostEqual(result["brier"], existing["brier"])
        self.assertAlmostEqual(result["ece"], existing["ece"])
        np.testing.assert_allclose([b["count"] for b in result["reliability_bins"]], [b["count"] for b in existing["reliability_bins"]])
        self.assertAlmostEqual(comparison.metric_summary([.2, .8], [0, 1])["nll"], -math.log(.8))
        self.assertAlmostEqual(comparison.metric_summary([.2, .8], [0, 1])["brier"], .04)

    def test_bootstrap_whole_images_prediction_weighting_and_determinism(self):
        images = ["a", "b", "empty"]
        rows = [{"image": "a", "correct_iou50": 0}] + [{"image": "b", "correct_iou50": 1}] * 3
        # Use two nonempty images for this exact resampling comparison.
        images = images[:2]
        values = {m: np.array([.1, .4, .5, .8]) for m in comparison.ALL_METHODS}
        values["Temperature"] = np.array([.2, .5, .6, .9])
        pairs, meta = comparison.cluster_bootstrap(images, rows, values, resamples=32)
        self.assertEqual((pairs, meta), comparison.cluster_bootstrap(images, rows, values, resamples=32))
        draws = np.random.Generator(np.random.PCG64(comparison.SEED)).integers(0, 2, size=(32, 2), dtype=np.int64)
        deltas = []
        for draw in draws:
            expanded = [j for i in draw for j, r in enumerate(rows) if r["image"] == images[i]]
            target = [rows[j]["correct_iou50"] for j in expanded]
            delta = comparison.metric_summary(values["Temperature"][expanded], target)["nll"] - comparison.metric_summary(values["Raw"][expanded], target)["nll"]
            deltas.append(delta)
        actual = next(r for r in pairs if (r["competitor"], r["reference"], r["metric"]) == ("Temperature", "Raw", "nll"))
        np.testing.assert_allclose([actual["ci_low"], actual["ci_high"]], np.percentile(deltas, [2.5, 97.5]))
        identical = next(r for r in pairs if (r["competitor"], r["reference"], r["metric"]) == ("Platt", "Beta", "brier"))
        self.assertEqual((identical["difference"], identical["ci_low"], identical["ci_high"]), (0., 0., 0.))


class SelectionTests(unittest.TestCase):
    def fixture(self):
        metrics = {m: {"nll": .4, "brier": .2, "ece": 999} for m in comparison.METHODS}
        metrics["Raw"] = {"nll": .8, "brier": .3}
        pairs = [{"competitor": a, "reference": b, "metric": metric, "difference": 0., "ci_low": -.1, "ci_high": .1}
                 for a in comparison.ALL_METHODS for b in comparison.ALL_METHODS if a != b for metric in ("nll", "brier")]
        return metrics, pairs

    def exclude(self, pairs, competitor, reference, metric):
        for row in pairs:
            if (row["competitor"], row["reference"], row["metric"]) == (competitor, reference, metric):
                row.update(ci_low=.01, ci_high=.1)

    def test_nll_hierarchy_ignores_ece(self):
        metrics, pairs = self.fixture()
        metrics["Beta"]["nll"] = .1
        for method in ("Temperature", "Platt", "Isotonic"):
            self.exclude(pairs, method, "Beta", "nll")
        result = comparison.select_method(metrics, pairs)
        self.assertEqual(result["selected_method"], "Beta")
        self.assertEqual(set(result["selected_minus_raw"]), {"nll", "brier"})
        self.assertFalse(result["final_fit_performed"])

    def test_brier_secondary_within_nll_set(self):
        metrics, pairs = self.fixture()
        metrics["Beta"]["nll"] = .1
        self.exclude(pairs, "Isotonic", "Beta", "nll")
        metrics["Platt"]["brier"] = .1
        metrics["Isotonic"]["brier"] = .001  # excluded by NLL, cannot win
        self.exclude(pairs, "Temperature", "Platt", "brier")
        self.exclude(pairs, "Beta", "Platt", "brier")
        self.assertEqual(comparison.select_method(metrics, pairs)["selected_method"], "Platt")

    def test_simplicity_order_and_ci_endpoint_includes_zero(self):
        metrics, pairs = self.fixture()
        for row in pairs:
            row["ci_low"] = 0.
        self.assertEqual(comparison.select_method(metrics, pairs)["selected_method"], "Temperature")
        metrics["Isotonic"]["nll"] = .1
        for method in ("Temperature", "Platt"):
            self.exclude(pairs, method, "Isotonic", "nll")
        self.assertEqual(comparison.select_method(metrics, pairs)["selected_method"], "Beta")

    def test_raw_baseline_safeguard_including_exact_tie(self):
        metrics, pairs = self.fixture()
        metrics["Raw"]["nll"] = .4
        result = comparison.select_method(metrics, pairs)
        self.assertIsNone(result["selected_method"])
        self.assertEqual(result["status"], "scientific_review_required")


class InputIsolationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = [f"images/train/synthetic_{i}.jpg" for i in range(821)]
        manifest = self.root / comparison.raw.DEVELOPMENT
        manifest.parent.mkdir(parents=True)
        manifest.write_text("\n".join(self.images) + "\n")
        raw_bytes = manifest.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(raw_bytes).hexdigest()
        for mock in (patch.object(comparison.raw, "MANIFEST_SHA", digest),
                     patch.object(comparison, "EXPECTED_PREDICTIONS", 4)):
            mock.start()
            self.addCleanup(mock.stop)
        self.source = self.root / comparison.analysis.OUTPUT
        self.source.mkdir(parents=True)
        pair_file = self.root / comparison.partition.PAIRS
        pair_file.parent.mkdir(parents=True)
        pair_file.write_text("train_file,val_file\nsynthetic_0.jpg,synthetic_1.jpg\nsynthetic_1.jpg,synthetic_2.jpg\n")
        self.write_input()

    def write_input(self, score=.01, label75="not-read", image=None):
        table = self.source / "labelled_predictions.csv"
        with table.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=("image", "source_row", "confidence", "correct_at_iou50", "correct_at_iou75"))
            writer.writeheader()
            for i in range(4):
                writer.writerow({"image": image or self.images[i], "source_row": i + 1, "confidence": score,
                                 "correct_at_iou50": i % 2, "correct_at_iou75": label75})
        metadata = {"status": "complete", "inclusion_threshold": .01, "primary_iou": .5,
                    "checkpoint_epoch": 72, "checkpoint_sha256": comparison.raw.CHECKPOINT_SHA,
                    "manifest_sha256_lf": comparison.raw.MANIFEST_SHA,
                    "output_sha256": {table.name: comparison.rq1._sha(table)}}
        comparison.rq1._write_json(self.source / "provenance.json", metadata)

    def test_sensitivity_field_not_used_and_all_images_kept(self):
        images, rows, edges, _ = comparison.load_inputs(self.root)
        self.assertEqual(len(images), 821)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(edges), 2)
        before = comparison.assign_folds(images, edges)
        self.write_input(label75="changed-sensitivity-label")
        new_images, new_rows, new_edges, _ = comparison.load_inputs(self.root)
        self.assertEqual(rows, new_rows)
        self.assertEqual(before, comparison.assign_folds(new_images, new_edges))

    def test_no_refiltering_and_test_image_isolation(self):
        self.write_input(score=.009)
        with self.assertRaises(ValueError):
            comparison.load_inputs(self.root)
        self.write_input(image="images/val/held-out-final-test.jpg")
        with self.assertRaises(ValueError):
            comparison.load_inputs(self.root)

    def test_malformed_population_and_hash_rejected(self):
        table = self.source / "labelled_predictions.csv"
        table.write_text(table.read_text() + "changed")
        with self.assertRaises(ValueError):
            comparison.load_inputs(self.root)
        self.write_input()
        with patch.object(comparison, "EXPECTED_PREDICTIONS", 7131):
            with self.assertRaises(ValueError):
                comparison.load_inputs(self.root)

    def test_group_crossing_development_boundary_rejected(self):
        (self.root / comparison.partition.PAIRS).write_text("train_file,val_file\nsynthetic_0.jpg,outside.jpg\n")
        with self.assertRaises(ValueError):
            comparison.load_inputs(self.root)

    def test_windows_real_execution_blocked(self):
        with patch.object(os, "name", "nt"):
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                comparison.main(["--dicc"])


class OutputLifecycleTests(unittest.TestCase):
    def test_synthetic_outputs_and_existing_directory_protection(self):
        images, rows = synthetic_population()
        bootstrap = comparison.cluster_bootstrap
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(comparison, "load_inputs", return_value=(images, rows, [], {})), \
                 patch.object(comparison.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"}), \
                 patch.object(comparison, "cluster_bootstrap", side_effect=lambda *args: bootstrap(*args, resamples=32)), \
                 patch.object(comparison, "RESAMPLES", 32):
                output = comparison.run(root)
                provenance = comparison.rq1._read_json(output / "provenance.json")
                self.assertEqual(provenance["status"], "complete")
                self.assertFalse(provenance["final_fit_performed"])
                self.assertEqual(provenance["bootstrap"]["resamples"], 32)
                with (output / "oof_calibrated_predictions.csv").open() as file:
                    self.assertEqual(len(list(csv.DictReader(file))), len(rows) * 4)
                self.assertTrue((output / "reliability_comparison.png").read_bytes().startswith(b"\x89PNG"))
                for name, digest in provenance["output_sha256"].items():
                    self.assertEqual(comparison.rq1._sha(output / name), digest)
                with self.assertRaises(FileExistsError):
                    comparison.run(root)

    def test_incomplete_output_never_reused(self):
        images, rows = synthetic_population()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(comparison, "load_inputs", return_value=(images, rows, [], {})), \
                 patch.object(comparison.rq1, "git_provenance", return_value={"git_dirty": False}), \
                 patch.object(comparison, "cross_fit", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    comparison.run(root)
                output = root / comparison.OUTPUT
                self.assertEqual(comparison.rq1._read_json(output / "provenance.json")["status"], "incomplete")
                with self.assertRaises(FileExistsError):
                    comparison.run(root)


if __name__ == "__main__":
    unittest.main()
