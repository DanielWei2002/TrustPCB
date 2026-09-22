"""Synthetic secondary-target fixtures only; primary outputs remain immutable."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustpcb import rq2_risk_sensitivity as sensitivity
import test_rq2_risk_evaluation as primary_tests

p = sensitivity.primary


def population():
    images, rows = primary_tests.population()
    for row, secondary in zip(rows, [0, 0, 0, 1]):
        row["correct_at_iou75"] = str(secondary)
    return images, rows


class TargetMetricTests(unittest.TestCase):
    def test_secondary_inversion_preserves_primary_and_input(self):
        _, rows = population()
        before = copy.deepcopy(rows)
        adapted = sensitivity.sensitivity_rows(rows)
        self.assertEqual(rows, before)
        self.assertEqual([r["incorrect_iou75"] for r in adapted], [1, 1, 1, 0])
        self.assertEqual([r["incorrect_prediction"] for r in adapted], [1, 1, 1, 0])
        self.assertEqual([r["correct_iou50"] for r in adapted], [r["correct_iou50"] for r in rows])

    def test_missing_nonbinary_and_conflicting_aliases_fail(self):
        for row in ({}, {"correct_at_iou75": ""}, {"correct_at_iou75": None},
                    {"correct_at_iou75": "nan"}, {"correct_at_iou75": .75},
                    {"correct_at_iou75": 1, "correct_iou75": 0}):
            with self.assertRaises(ValueError):
                sensitivity.sensitivity_rows([row])
        self.assertEqual(sensitivity.sensitivity_rows([{"correct_iou75": 1}])[0]["incorrect_iou75"], 0)

    def test_all_eight_risks_unchanged_by_secondary_labels(self):
        _, rows = population()
        original = p.risk_scores(rows)
        adapted = p.risk_scores(sensitivity.sensitivity_rows(rows))
        self.assertEqual(tuple(adapted), p.METHODS)
        for method in p.METHODS:
            np.testing.assert_array_equal(original[method], adapted[method])

    def test_same_draws_as_primary_deterministic_paired_and_invalid(self):
        images, rows = population()
        adapted = sensitivity.sensitivity_rows(rows)
        scores = p.risk_scores(adapted)
        primary_samples, primary_info = p.bootstrap(images, rows, scores, replicates=32)
        samples, info = p.bootstrap(images, adapted, scores, replicates=32, seed=sensitivity.SEED)
        repeated, repeated_info = p.bootstrap(images, adapted, scores, replicates=32, seed=sensitivity.SEED)
        self.assertEqual(info, repeated_info)
        self.assertEqual(info["draw_indices_sha256"], primary_info["draw_indices_sha256"])
        np.testing.assert_equal(samples, repeated)
        np.testing.assert_equal(samples[:, 0], samples[:, 1])
        self.assertTrue(np.isnan(samples[:, 0, 0]).any())
        self.assertFalse(np.array_equal(primary_samples, samples, equal_nan=True))
        draws = np.random.Generator(np.random.PCG64(sensitivity.SEED)).integers(0, len(images), size=(32, len(images)), dtype=np.int64)
        for index, draw in enumerate(draws):
            ids = [i for image_id in draw for i, row in enumerate(adapted) if row["image"] == images[image_id]]
            result = p.metrics(scores[p.METHODS[0]][ids], [adapted[i]["incorrect_iou75"] for i in ids])
            np.testing.assert_allclose(samples[index, 0], [np.nan if result[k] is None else result[k] for k in ("auroc", "auprc")], equal_nan=True)

    def test_metrics_direction_and_pairwise_delta_sign(self):
        _, rows = population()
        rows = sensitivity.sensitivity_rows(rows)
        y = np.array([r["incorrect_iou75"] for r in rows])
        scores = {m: y.astype(float) for m in p.METHODS}
        scores[p.METHODS[0]] = 1 - y
        samples = np.ones((3, 8, 2))
        samples[:, 0, 0] = 0
        methods, differences = p.evaluation_tables(rows, scores, samples)
        self.assertEqual(methods[-1]["auroc"], 1.)
        self.assertEqual(methods[-1]["auprc"], 1.)
        self.assertEqual(differences[0]["observed_delta_auroc"], 1.)
        self.assertEqual(differences[0]["bootstrap_mean"], 1.)
        self.assertEqual([(r["method_A"], r["method_B"]) for r in differences], list(p.PAIRS))


class InputTests(unittest.TestCase):
    def test_reuses_primary_population_and_final_test_guards(self):
        images, rows = population()
        with patch.object(p, "load_inputs", return_value=(images, rows, {"synthetic": "hash"})) as loader:
            result = sensitivity.load_inputs(Path("synthetic"))
        loader.assert_called_once_with(Path("synthetic"))
        self.assertEqual(result[2], {"synthetic": "hash"})
        for reason in ("7131 population mismatch", "foreign final-test image", "missing score"):
            with patch.object(p, "load_inputs", side_effect=ValueError(reason)):
                with self.assertRaisesRegex(ValueError, reason):
                    sensitivity.load_inputs(Path("synthetic"))

    def test_real_text_validator_before_secondary_adaptation(self):
        images = [f"synthetic_{i}" for i in range(821)]
        _, rows = population()
        for index, row in enumerate(rows):
            row["image"] = images[index]
        validated = p.validate_rows(images, rows, expected_count=4)
        self.assertEqual(len(sensitivity.sensitivity_rows(validated)), 4)
        with self.assertRaises(ValueError):
            p.validate_rows(images, rows)
        rows[0]["image"] = "images/val/final-test.jpg"
        with self.assertRaisesRegex(ValueError, "foreign"):
            p.validate_rows(images, rows, expected_count=4)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        images, rows = population()
        for mocked in (patch.object(p, "load_inputs", return_value=(images, rows, {p.INPUT: "synthetic"})),
                       patch.object(sensitivity.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"}),
                       patch.object(sensitivity, "REPLICATES", 32)):
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_outputs_labels_primary_isolation_provenance_and_overwrite(self):
        primary = self.root / p.OUTPUT
        primary.mkdir(parents=True)
        sentinel = primary / "method_metrics.csv"
        sentinel.write_bytes(b"immutable synthetic primary evidence")
        before = sensitivity.rq1._sha(sentinel)
        with patch.object(p, "run", side_effect=AssertionError("primary run forbidden")):
            output = sensitivity.run(self.root)
        self.assertEqual(sensitivity.rq1._sha(sentinel), before)
        self.assertEqual(list(primary.iterdir()), [sentinel])
        info = sensitivity.rq1._read_json(output / "provenance.json")
        self.assertEqual(info["analysis_label"], sensitivity.LABEL)
        self.assertEqual(info["sensitivity_correctness_iou"], .75)
        self.assertEqual(info["primary_correctness_iou"], .5)
        self.assertEqual(info["risk_formulas"], p.FORMULAS)
        self.assertEqual(info["metric_definitions"]["target"], sensitivity.TARGET)
        self.assertEqual(info["status"], "complete")
        for name, digest in info["output_sha256"].items():
            self.assertEqual(sensitivity.rq1._sha(output / name), digest)
        summary = sensitivity.rq1._read_json(output / "summary.json")
        self.assertEqual(summary["incorrect"], 3)
        self.assertEqual(summary["incorrect_prevalence"], .75)
        self.assertEqual(len(p.read_csv(output / "method_metrics.csv")), 8)
        self.assertEqual(len(p.read_csv(output / "bootstrap_metrics.csv")), 32 * 8)
        with self.assertRaises(FileExistsError):
            sensitivity.run(self.root)

    def test_failure_leaves_incomplete_directory_and_blocks_retry(self):
        with patch.object(p, "bootstrap", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                sensitivity.run(self.root)
        self.assertEqual(sensitivity.rq1._read_json(self.root / sensitivity.OUTPUT / "provenance.json")["status"], "incomplete")
        with self.assertRaises(FileExistsError):
            sensitivity.run(self.root)

    def test_windows_gate_precedes_real_analysis(self):
        with patch.object(sensitivity.os, "name", "nt"), patch.object(sensitivity, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                sensitivity.main(["--dicc"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
