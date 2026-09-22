"""Synthetic development ranking fixtures only; no research run or image access."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustpcb import rq2_risk_evaluation as risk
import test_rq2_final_calibrator as final_tests


def population():
    images = ["synthetic_a", "synthetic_b", "synthetic_zero"]
    rows = [{"image": image, "prediction_id": str(i), "raw_confidence": p,
             "calibrated_confidence": p, "class_consistency": 1 - p, "localisation_stability": .5,
             "correct_iou50": correct, "incorrect_prediction": 1 - correct, "class_id": i % 2}
            for i, (image, p, correct) in enumerate(((images[0], .2, 0), (images[1], .4, 1),
                                                    (images[1], .7, 0), (images[1], .9, 1)))]
    return images, rows


class FormulaMetricTests(unittest.TestCase):
    def test_all_eight_equal_weight_formulas(self):
        row = {"raw_confidence": .2, "calibrated_confidence": .4, "class_consistency": .7, "localisation_stability": .9}
        result = risk.risk_scores([row])
        self.assertEqual(tuple(result), risk.METHODS)
        np.testing.assert_allclose([result[m][0] for m in risk.METHODS], [.8, .6, .3, .1, .45, .35, .2, 1 / 3])

    def test_auroc_direction_ties_and_ap_positive_class(self):
        self.assertEqual(risk.metrics([.9, .1], [1, 0]), {"auroc": 1., "auprc": 1.})
        self.assertEqual(risk.metrics([.1, .9], [1, 0]), {"auroc": 0., "auprc": .5})
        self.assertEqual(risk.metrics([.5, .5], [1, 0]), {"auroc": .5, "auprc": .5})
        result = risk.metrics([.9, .8, .7, .6], [1, 0, 1, 0])
        self.assertEqual(result["auroc"], .75)
        self.assertAlmostEqual(result["auprc"], (1 + 2 / 3) / 2)

    def test_undefined_metrics_and_all_positive_ap(self):
        self.assertEqual(risk.metrics([1, 2], [0, 0]), {"auroc": None, "auprc": None})
        self.assertEqual(risk.metrics([1, 2], [1, 1]), {"auroc": None, "auprc": 1.})
        self.assertEqual(risk.metrics([], []), {"auroc": None, "auprc": None})

    def test_frequency_weights_equal_expanded_predictions(self):
        scores, labels, weights = [.8, .8, .2], [1, 0, 0], [3, 2, 4]
        actual = risk.weighted_metrics(risk.ranking_plan(scores, labels), weights)[0]
        expanded = risk.metrics(np.repeat(scores, weights), np.repeat(labels, weights))
        np.testing.assert_allclose(actual, [expanded["auroc"], expanded["auprc"]])
        self.assertAlmostEqual(actual[0], (4 + 1) / 6)
        self.assertAlmostEqual(actual[1], 3 / 5)

    def test_interval_linear_percentiles_ignore_nan_without_redraw(self):
        result = risk.interval([0, 1, 2, 3, np.nan])
        self.assertEqual(result["valid_replicates"], 4)
        self.assertEqual(result["invalid_replicates"], 1)
        self.assertAlmostEqual(result["lower_95"], .075)
        self.assertAlmostEqual(result["upper_95"], 2.925)
        self.assertIsNone(risk.interval([np.nan])["bootstrap_mean"])


class BootstrapTests(unittest.TestCase):
    def test_paired_cluster_draws_multiplicity_seed_and_zero_image(self):
        images, rows = population()
        scores = risk.risk_scores(rows)
        samples, metadata = risk.bootstrap(images, rows, scores, replicates=32)
        again, again_metadata = risk.bootstrap(images, rows, scores, replicates=32)
        np.testing.assert_equal(samples, again)
        self.assertEqual(metadata, again_metadata)
        np.testing.assert_equal(samples[:, 0], samples[:, 1])
        draws = np.random.Generator(np.random.PCG64(24209199)).integers(0, 3, size=(32, 3), dtype=np.int64)
        self.assertTrue(any(len(set(draw)) < len(draw) for draw in draws))
        for index, draw in enumerate(draws):
            expanded_ids = [i for sampled in draw for i, row in enumerate(rows) if row["image"] == images[sampled]]
            for j, method in enumerate(risk.METHODS):
                values = risk.metrics(scores[method][expanded_ids], [rows[i]["incorrect_prediction"] for i in expanded_ids])
                expected = [np.nan if values[key] is None else values[key] for key in ("auroc", "auprc")]
                np.testing.assert_allclose(samples[index, j], expected, equal_nan=True)
        self.assertTrue(np.isnan(samples[:, 0, 0]).any())
        self.assertEqual(metadata["sampled_images_per_replicate"], 3)

    def test_invalid_replicates_reported_per_metric(self):
        images, rows = population()
        scores = risk.risk_scores(rows)
        samples, _ = risk.bootstrap(images, rows, scores, replicates=32)
        methods, pairs = risk.evaluation_tables(rows, scores, samples)
        for row in methods:
            self.assertEqual(row["auroc_valid_replicates"] + row["auroc_invalid_replicates"], 32)
            self.assertEqual(row["auprc_valid_replicates"] + row["auprc_invalid_replicates"], 32)
            self.assertGreater(row["auroc_invalid_replicates"], 0)
        self.assertEqual(len(pairs), 5)

    def test_pairwise_sign_A_minus_B(self):
        _, rows = population()
        y = np.array([r["incorrect_prediction"] for r in rows])
        scores = {method: y.astype(float) for method in risk.METHODS}
        scores[risk.METHODS[0]] = 1 - y
        samples = np.ones((4, 8, 2))
        samples[:, 0, 0] = 0
        _, pairs = risk.evaluation_tables(rows, scores, samples)
        self.assertEqual(pairs[0]["observed_delta_auroc"], 1)
        self.assertEqual(pairs[0]["bootstrap_mean"], 1)
        self.assertEqual(pairs[0]["lower_95"], 1)


class DiagnosticTests(unittest.TestCase):
    def test_descriptives_population_std_quantiles(self):
        _, rows = population()
        desc, _, _ = risk.diagnostics(rows, risk.risk_scores(rows))
        row = next(r for r in desc if r["signal"] == "calibrated_confidence" and r["group"] == "incorrect")
        self.assertEqual(row["count"], 2)
        self.assertAlmostEqual(row["mean"], .45)
        self.assertAlmostEqual(row["standard_deviation"], .25)
        self.assertAlmostEqual(row["q1"], .325)
        self.assertAlmostEqual(row["median"], .45)
        self.assertAlmostEqual(row["q3"], .575)

    def test_spearman_constant_and_per_class_undefined(self):
        _, rows = population()
        _, correlations, classes = risk.diagnostics(rows, risk.risk_scores(rows))
        self.assertAlmostEqual(correlations[0]["spearman_rho"], -1)
        self.assertIsNone(correlations[1]["spearman_rho"])
        self.assertEqual(len(classes), 8)
        self.assertTrue(all(r["auroc"] is None for r in classes))

    def test_tied_spearman_and_absent_class_optional(self):
        _, rows = population()
        for index, row in enumerate(rows):
            row["calibrated_confidence"] = row["class_consistency"] = index // 2
            row.pop("class_id")
        _, correlations, classes = risk.diagnostics(rows, risk.risk_scores(rows))
        self.assertAlmostEqual(correlations[0]["spearman_rho"], 1)
        self.assertEqual(classes, [])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.images = [f"synthetic_{i}" for i in range(821)]
        _, self.rows = population()
        for i, row in enumerate(self.rows):
            row["image"] = self.images[i]

    def validate(self):
        return risk.validate_rows(self.images, self.rows, expected_count=4)

    def test_target_inversion(self):
        self.assertEqual([r["incorrect_prediction"] for r in self.validate()], [1, 0, 1, 0])

    def test_ranges_missing_and_binary_labels(self):
        for key in ("raw_confidence", *risk.SIGNALS):
            previous = self.rows[0][key]
            for value in (None, "", float("nan"), float("inf"), -.1, 1.1):
                self.rows[0][key] = value
                with self.assertRaises(ValueError):
                    self.validate()
            self.rows[0][key] = previous
        self.rows[0]["correct_iou50"] = .5
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_and_final_test_isolation(self):
        self.rows[1]["prediction_id"] = self.rows[0]["prediction_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.validate()
        self.rows[1]["prediction_id"] = "unique"
        self.rows[1]["image"] = "images/val/final-test.jpg"
        with self.assertRaisesRegex(ValueError, "foreign"):
            self.validate()

    def test_population_count_and_reference_floor(self):
        with self.assertRaises(ValueError):
            risk.validate_rows(self.images, self.rows)
        self.rows[0]["raw_confidence"] = .009
        with self.assertRaises(ValueError):
            self.validate()
        self.rows[0]["raw_confidence"] = .2
        self.images.pop()
        with self.assertRaises(ValueError):
            self.validate()


class InputBindingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = final_tests.FinalCalibratorTests("test_unreviewed_selection_rejected_before_reading")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.execute()
        self.root = self.fixture.root
        mocked = patch.object(risk, "PREDICTION_COUNT", 40)
        mocked.start()
        self.addCleanup(mocked.stop)
        hashes = risk.final.load_inputs(self.root, approve_beta_selection=True)[3]
        folder = self.root / risk.final.OUTPUT
        for name in ("final_calibrator.json", "calibrated_development_predictions.csv", "fit_summary.json", "provenance.json"):
            hashes[f"{risk.final.OUTPUT}/{name}"] = risk.rq1._sha(folder / name)
        rows = risk.read_csv(folder / "calibrated_development_predictions.csv")
        for row in rows:
            row.update(class_consistency="0.8", localisation_stability="0.6")
        self.stability = self.root / risk.STABILITY
        self.stability.mkdir(parents=True)
        table = self.stability / "prediction_stability_scores.csv"
        risk.final.comparison.write_csv(table, rows)
        risk.rq1._write_json(self.stability / "provenance.json", {"status": "complete", "input_sha256": hashes,
            "output_sha256": {table.name: risk.rq1._sha(table)}})

    def test_complete_binding_without_checkpoint_or_images(self):
        images, rows, hashes = risk.load_inputs(self.root)
        self.assertEqual(len(images), 821)
        self.assertEqual(len(rows), 40)
        self.assertIn(risk.INPUT, hashes)
        self.assertFalse((self.root / "images").exists())

    def test_table_tampering_rejected(self):
        table = self.root / risk.INPUT
        table.write_text(table.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            risk.load_inputs(self.root)

    def test_reordered_rows_rejected_with_updated_hash(self):
        table = self.root / risk.INPUT
        rows = risk.read_csv(table)[::-1]
        table.unlink()
        risk.final.comparison.write_csv(table, rows)
        provenance = risk.rq1._read_json(self.stability / "provenance.json")
        provenance["output_sha256"][table.name] = risk.rq1._sha(table)
        risk.rq1._write_json(self.stability / "provenance.json", provenance)
        with self.assertRaisesRegex(ValueError, "identity/order"):
            risk.load_inputs(self.root)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        images, rows = population()
        for mocked in (patch.object(risk, "load_inputs", return_value=(images, rows, {risk.INPUT: "synthetic"})),
                       patch.object(risk.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"}),
                       patch.object(risk, "REPLICATES", 32)):
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_outputs_provenance_and_overwrite(self):
        output = risk.run(self.root)
        info = risk.rq1._read_json(output / "provenance.json")
        self.assertEqual(info["risk_formulas"], risk.FORMULAS)
        self.assertEqual(info["bootstrap_seed"], 24209199)
        self.assertEqual(info["status"], "complete")
        self.assertEqual(info["input_sha256"][risk.INPUT], "synthetic")
        for name, digest in info["output_sha256"].items():
            self.assertEqual(risk.rq1._sha(output / name), digest)
        summary = risk.rq1._read_json(output / "summary.json")
        self.assertEqual(summary["incorrect_prevalence"], .5)
        self.assertFalse(summary["weights_tuned"])
        self.assertFalse(summary["threshold_selected"])
        self.assertEqual(len(risk.read_csv(output / "bootstrap_metrics.csv")), 32 * 8)
        with self.assertRaises(FileExistsError):
            risk.run(self.root)

    def test_incomplete_output_blocks_retry(self):
        with patch.object(risk, "bootstrap", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                risk.run(self.root)
        self.assertEqual(risk.rq1._read_json(self.root / risk.OUTPUT / "provenance.json")["status"], "incomplete")
        with self.assertRaises(FileExistsError):
            risk.run(self.root)

    def test_windows_gate_before_input(self):
        with patch.object(risk.os, "name", "nt"), patch.object(risk, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                risk.main(["--dicc"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
