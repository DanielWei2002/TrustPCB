"""Tiny synthetic CV only; full grid enumeration is text/arithmetic, not a run."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustpcb import rq2_weighted_fusion as fusion


def fixture():
    images = [f"synthetic_{i}" for i in range(10)]
    rows = [{"image": image, "correct_iou50": y, "correct_at_iou75": y,
             "calibrated_confidence": .2 + .6*y, "raw_confidence": .2 + .6*y,
             "class_consistency": .8 - .6*y, "localisation_stability": .5}
            for image in images for y in (0, 1)]
    folds = fusion.comparison.assign_folds(images, [(images[0], images[1])])
    return images, rows, folds


def result(weights, ap=.6, roc=.7):
    return {**dict(zip(fusion.WEIGHT_KEYS, weights)), "mean_fold_auprc": ap, "mean_fold_auroc": roc}


class GridSelectionTests(unittest.TestCase):
    def test_exact_integer_grid_vertices_and_sum(self):
        grid = fusion.grid()
        self.assertEqual(len(grid), 5151)
        self.assertEqual(len(set(grid)), 5151)
        self.assertTrue(all(sum(w) == 100 and min(w) >= 0 for w in grid))
        for weights in ((100, 0, 0), (0, 100, 0), (0, 0, 100), (33, 33, 34), (70, 10, 20)):
            self.assertIn(weights, grid)

    def test_invalid_weights_and_count_rejected(self):
        for weights in ((-1, 0, 101), (30, 30, 30), (.5, .5, 0), (100, 0)):
            with self.assertRaises(ValueError):
                fusion.validate_weights(weights)
        with patch.object(fusion, "CANDIDATE_COUNT", 3):
            with self.assertRaisesRegex(ValueError, "5151"):
                fusion.grid()

    def test_auprc_primary_over_auroc(self):
        selected = fusion.select([result((100, 0, 0), .8, .5), result((0, 100, 0), .7, 1)])
        self.assertEqual(selected["weights"]["w_conf"], 1)

    def test_auroc_tie_break_and_absolute_tolerance(self):
        selected = fusion.select([result((100, 0, 0), .8, .5), result((0, 100, 0), .8 - 5e-13, .9)])
        self.assertEqual(selected["weights"]["w_class"], 1)
        selected = fusion.select([result((100, 0, 0), .8, .5), result((0, 100, 0), .8 - 2e-12, .9)])
        self.assertEqual(selected["weights"]["w_conf"], 1)

    def test_equal_weight_distance(self):
        selected = fusion.select([result((100, 0, 0)), result((33, 33, 34))])
        self.assertEqual(selected["integer_percentages"]["localisation_percent"], 34)

    def test_final_lexicographic_conf_then_localisation(self):
        selected = fusion.select([result((33, 34, 33)), result((33, 33, 34)), result((34, 33, 33))])
        self.assertEqual(selected["integer_percentages"]["conf_percent"], 34)
        selected = fusion.select([result((33, 34, 33)), result((33, 33, 34))])
        self.assertEqual(selected["integer_percentages"]["localisation_percent"], 34)


class CVTests(unittest.TestCase):
    def test_deterministic_grouped_image_folds(self):
        images, rows, folds = fixture()
        self.assertEqual(folds, fusion.comparison.assign_folds(images, [(images[1], images[0])]))
        lookup = {f["image"]: f["fold_id"] for f in folds}
        self.assertEqual(lookup[images[0]], lookup[images[1]])
        self.assertEqual([f["image"] for f in folds], images)
        self.assertEqual(len(lookup), 10)
        self.assertEqual(len({lookup[r["image"]] for r in rows}), 5)

    def test_fold_metrics_and_secondary_cannot_influence_candidates(self):
        images, rows, folds = fixture()
        candidates = [(100, 0, 0), (0, 100, 0), (0, 0, 100)]
        before = fusion.evaluate_candidates(images, rows, folds, candidates)
        self.assertEqual(before[0]["mean_fold_auprc"], 1)
        self.assertEqual(before[1]["mean_fold_auroc"], 0)
        changed = copy.deepcopy(rows)
        for row in changed:
            row["correct_at_iou75"] = "must never be parsed in selection"
        self.assertEqual(before, fusion.evaluate_candidates(images, changed, folds, candidates))
        self.assertEqual(fusion.select(before)["weights"]["w_conf"], 1)

    def test_invalid_single_class_fold_stops_without_reassignment(self):
        images, rows, folds = fixture()
        for row in rows:
            row["correct_iou50"] = 1
        with self.assertRaisesRegex(ValueError, "both IoU50"):
            fusion.evaluate_candidates(images, rows, folds, [(100, 0, 0)])

    def test_selected_weights_unchanged_for_secondary(self):
        _, rows, _ = fixture()
        selected = fusion.select([result((70, 10, 20))])
        before = copy.deepcopy(selected)
        for row in rows:
            row["correct_at_iou75"] = 1 - row["correct_iou50"]
        primary, secondary = fusion.report_metrics(rows, selected)
        self.assertEqual(selected, before)
        self.assertEqual(primary[-1]["auroc"], 1)
        self.assertEqual(secondary[-1]["auroc"], 0)
        self.assertEqual(secondary[-1]["analysis_label"], fusion.sensitivity.LABEL)

    def test_foreign_image_rejected(self):
        images, rows, folds = fixture()
        rows[0]["image"] = "images/val/final-test.jpg"
        with self.assertRaisesRegex(ValueError, "development-only"):
            fusion.evaluate_candidates(images, rows, folds, [(100, 0, 0)])

    def test_loader_preserves_primary_population_guards(self):
        for reason in ("7131 population mismatch", "foreign final-test input", "821 image count"):
            with patch.object(fusion.risk, "load_inputs", side_effect=ValueError(reason)):
                with self.assertRaisesRegex(ValueError, reason):
                    fusion.load_inputs("synthetic")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images, self.rows, _ = fixture()
        for mocked in (patch.object(fusion, "load_inputs", return_value=(self.images, self.rows, [], {"synthetic": "hash"})),
                       patch.object(fusion.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"})):
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_synthetic_orchestration_primary_projection_isolation_provenance(self):
        sentinel = self.root / fusion.risk.OUTPUT / "method_metrics.csv"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_bytes(b"immutable synthetic equal-weight results")
        digest = fusion.rq1._sha(sentinel)
        def synthetic_results(images, rows, folds, candidates):
            self.assertTrue(all("correct_at_iou75" not in r for r in rows))
            return [result(w) for w in candidates]  # no full-grid scientific evaluation
        with patch.object(fusion, "evaluate_candidates", side_effect=synthetic_results), \
             patch.object(fusion.comparison, "fit_calibrator", side_effect=AssertionError("fitting forbidden")), \
             patch.object(fusion.comparison.raw, "_predict", side_effect=AssertionError("inference forbidden")):
            output = fusion.run(self.root)
        self.assertEqual(fusion.rq1._sha(sentinel), digest)
        info = fusion.rq1._read_json(output / "provenance.json")
        self.assertEqual(info["actual_candidate_count"], 5151)
        self.assertEqual(info["seed"], 24209199)
        self.assertIn("Proposal-specified", info["purpose"])
        self.assertEqual(info["selected_weights"]["integer_percentages"]["conf_percent"], 34)
        self.assertFalse(info["selected_weights"]["iou75_used_for_selection"])
        for name, expected in info["output_sha256"].items():
            self.assertEqual(fusion.rq1._sha(output / name), expected)
        with self.assertRaises(FileExistsError):
            fusion.run(self.root)

    def test_incomplete_output_blocks_retry(self):
        with patch.object(fusion, "evaluate_candidates", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                fusion.run(self.root)
        self.assertEqual(fusion.rq1._read_json(self.root / fusion.OUTPUT / "provenance.json")["status"], "incomplete")
        with self.assertRaises(FileExistsError):
            fusion.run(self.root)

    def test_windows_gate_before_data_access(self):
        with patch.object(fusion.os, "name", "nt"), patch.object(fusion, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                fusion.main(["--dicc"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
