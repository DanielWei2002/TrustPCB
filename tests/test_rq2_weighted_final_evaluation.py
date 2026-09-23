"""Synthetic text and ranking fixtures only; never real final-test analysis."""

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustpcb import rq2_weighted_final_evaluation as evaluation


def fixture():
    images = ["synthetic_a", "synthetic_b", "zero_predictions"]
    rows = [{"image": image, "prediction_id": str(i), "raw_confidence": .2 + .6*y,
             "calibrated_confidence": .2 + .6*y, "class_consistency": .8 - .6*y,
             "localisation_stability": .5, "correct_iou50": y, "correct_iou75": 0,
             "incorrect_iou75": 1}
            for i, (image, y) in enumerate(((images[0], 0), (images[1], 1), (images[1], 0)))]
    return images, rows


def selected():
    return {"weights": {"w_conf": .89, "w_class": .01, "w_localisation": .10},
            "integer_percentages": dict(zip(evaluation.fusion.WEIGHT_KEYS, (89, 1, 10))), "iou75_used_for_selection": False}


class ValidationTests(unittest.TestCase):
    def test_frozen_weights_and_mismatch(self):
        evaluation.validate_weights(selected())
        for key in ("w_conf", "w_class", "w_localisation"):
            record = selected()
            record["weights"][key] += .01
            with self.assertRaises(ValueError):
                evaluation.validate_weights(record)

    def test_selected_file_matches_evidence_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / evaluation.SELECTED
            path.parent.mkdir(parents=True)
            evaluation.rq1._write_json(path, selected())
            blob = path.read_bytes()
            with patch.object(evaluation.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=blob)) as git:
                identity = evaluation.selected_identity(root)
            self.assertEqual(identity["evidence_commit"], evaluation.EVIDENCE_COMMIT)
            self.assertEqual(identity["sha256"], evaluation.rq1._sha(path))
            self.assertIn(evaluation.EVIDENCE_COMMIT, git.call_args.args[0][-1])
            with patch.object(evaluation.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=b"different")):
                with self.assertRaisesRegex(ValueError, "evidence commit"):
                    evaluation.selected_identity(root)

    def test_population_counts_and_invalid_signals(self):
        _, rows = fixture()
        images = [f"synthetic_{i}" for i in range(2052)]
        for i, row in enumerate(rows):
            row["image"] = images[i]
        with self.assertRaisesRegex(ValueError, "17840"):
            evaluation.validate_population(images, rows)
        with patch.object(evaluation, "PREDICTION_COUNT", 3):
            self.assertEqual(len(evaluation.validate_population(images, rows)), 3)
            for field, value in (("calibrated_confidence", float("nan")), ("raw_confidence", .009), ("correct_iou75", .5)):
                changed = copy.deepcopy(rows)
                changed[0][field] = value
                with self.assertRaises(ValueError):
                    evaluation.validate_population(images, changed)
            rows[0]["image"] = "development_image"
            with self.assertRaisesRegex(ValueError, "non-final-test"):
                evaluation.validate_population(images, rows)

    def test_duplicate_identity_rejected(self):
        images = [f"synthetic_{i}" for i in range(2052)]
        _, rows = fixture()
        for r in rows:
            r["image"] = images[0]
            r["prediction_id"] = "duplicate"
        with patch.object(evaluation, "PREDICTION_COUNT", 3):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                evaluation.validate_population(images, rows)


class MetricTests(unittest.TestCase):
    def test_exact_weighted_score_and_equal_baseline(self):
        _, rows = fixture()
        scores = evaluation.scores_for(rows)
        self.assertAlmostEqual(scores["weighted_trustpcb_risk"][0], .89*.8 + .01*.2 + .10*.5)
        self.assertEqual(scores["equal_weight_trustpcb_risk"][0], evaluation.risk.risk_scores(rows)["trustpcb_risk"][0])

    def test_shared_deterministic_bootstrap_and_secondary(self):
        images, rows = fixture()
        scores = evaluation.scores_for(rows)
        before = copy.deepcopy(rows)
        with patch.object(evaluation, "REPLICATES", 32):
            first = evaluation.evaluate(images, rows, scores)
            repeated = evaluation.evaluate(images, rows, scores)
            secondary = evaluation.evaluate(images, rows, scores, secondary=True)
        self.assertEqual(first, repeated)
        self.assertEqual(first[2]["draw_indices_sha256"], secondary[2]["draw_indices_sha256"])
        self.assertEqual(rows, before)
        self.assertTrue(all(r["auroc"] is None for r in secondary[0]))
        self.assertTrue(all(r["auroc_invalid_replicates"] == 32 for r in secondary[0]))
        self.assertEqual(len(first[1]), 4)
        self.assertTrue(all(r["method_A"] == "weighted_trustpcb_risk" for r in first[1]))
        self.assertTrue(all(r["analysis_label"] == evaluation.sensitivity.LABEL for r in secondary[1]))
        for pair in secondary[1]:
            if pair["metric"] == "auroc":
                self.assertIsNone(pair["observed_difference"])
                self.assertEqual(pair["invalid_replicates"], 32)

    def test_paired_metrics_share_samples_and_difference_signs(self):
        images, rows = fixture()
        # Weighted ranking lies between the two references for both metrics.
        scores = dict(zip(evaluation.METHODS, (np.array([.9, .1, .8]),
                                              np.array([.2, .9, .1]),
                                              np.array([.9, .8, .1]))))
        samples = np.array([[[.9, .8], [.1, .2], [.6, .5]],
                            [[.7, .6], [.2, .3], [.5, .4]],
                            [[np.nan, .6], [.2, np.nan], [.5, .4]],
                            [[.7, .6], [.2, .3], [np.nan, np.nan]]])
        with patch.object(evaluation.risk, "bootstrap", return_value=(samples, {"draw_indices_sha256": "shared"})) as bootstrap:
            methods, pairs, info = evaluation.evaluate(images, rows, scores)
        bootstrap.assert_called_once()
        self.assertEqual(bootstrap.call_args.kwargs["replicates"], 10000)
        self.assertEqual(bootstrap.call_args.kwargs["seed"], 24209199)
        self.assertEqual(info["draw_indices_sha256"], "shared")
        observed = {r["method"]: r for r in methods}
        for pair in pairs:
            j = evaluation.METHODS.index(pair["method_B"])
            k = ("auroc", "auprc").index(pair["metric"])
            self.assertAlmostEqual(pair["observed_difference"],
                observed[pair["method_A"]][pair["metric"]] - observed[pair["method_B"]][pair["metric"]])
            self.assertEqual(np.sign(pair["observed_difference"]), -1 if j == 0 else 1)
            valid = [0, 1] if (j, k) in ((0, 0), (1, 1)) else [0, 1, 2]
            differences = samples[valid, 2, k] - samples[valid, j, k]
            self.assertAlmostEqual(pair["bootstrap_mean"], differences.mean())
            self.assertAlmostEqual(pair["lower_95"], np.percentile(differences, 2.5))
            self.assertAlmostEqual(pair["upper_95"], np.percentile(differences, 97.5))
            self.assertEqual(pair["valid_replicates"], len(valid))
            self.assertEqual(pair["invalid_replicates"], 4-len(valid))

    def test_bootstrap_subset_reuses_identical_draws_and_image_multiplicity(self):
        images, rows = fixture()
        rows = [{**r, "incorrect_prediction": 1-r["correct_iou50"]} for r in rows]
        scores = evaluation.scores_for(rows)
        sample, info = evaluation.risk.bootstrap(images, rows, scores, methods=evaluation.METHODS, replicates=16)
        expanded_scores = {m: scores[evaluation.METHODS[0]] for m in evaluation.risk.METHODS}
        original, old_info = evaluation.risk.bootstrap(images, rows, expanded_scores, replicates=16)
        self.assertEqual(info["draw_indices_sha256"], old_info["draw_indices_sha256"])
        np.testing.assert_equal(sample[:, 0], original[:, 0])
        draws = np.random.Generator(np.random.PCG64(24209199)).integers(0, 3, size=(16, 3), dtype=np.int64)
        for i, draw in enumerate(draws):
            ids = [j for k in draw for j, row in enumerate(rows) if row["image"] == images[k]]
            result = evaluation.risk.metrics(scores[evaluation.METHODS[2]][ids], [rows[j]["incorrect_prediction"] for j in ids])
            np.testing.assert_allclose(sample[i, 2], [np.nan if result[m] is None else result[m] for m in ("auroc", "auprc")], equal_nan=True)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        images, rows = fixture()
        for mocked in (patch.object(evaluation, "load_inputs", return_value=(images, rows, {"synthetic": "hash"}, {"evidence_commit": evaluation.EVIDENCE_COMMIT})),
                       patch.object(evaluation.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"}),
                       patch.object(evaluation, "REPLICATES", 32)):
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_output_isolation_provenance_and_forbidden_operations(self):
        sentinel = self.root / evaluation.final.OUTPUT / "synthetic_immutable.txt"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_bytes(b"unchanged final results")
        digest = evaluation.rq1._sha(sentinel)
        with patch.object(evaluation.fusion, "select", side_effect=AssertionError("no selection")), \
             patch.object(evaluation.fusion, "grid", side_effect=AssertionError("no search")), \
             patch.object(evaluation.final.calibration, "fit_calibrator", side_effect=AssertionError("no fit")), \
             patch.object(evaluation.final.raw, "_predict", side_effect=AssertionError("no inference")), \
             patch.object(evaluation.final.stability, "_predict", side_effect=AssertionError("no transforms")):
            output = evaluation.run(self.root)
        self.assertEqual(evaluation.rq1._sha(sentinel), digest)
        provenance = evaluation.rq1._read_json(output / "provenance.json")
        self.assertEqual(provenance["test_role"], evaluation.final.ROLE)
        self.assertEqual(provenance["weights"], selected()["weights"])
        self.assertEqual(provenance["status"], "complete")
        for prefix in ("primary", "sensitivity_iou75"):
            pairs = evaluation.risk.read_csv(output / f"{prefix}_pairwise_differences.csv")
            self.assertEqual(len(pairs), 4)
            self.assertEqual({r["metric"] for r in pairs}, {"auroc", "auprc"})
        for key in evaluation.NO_ACTIONS:
            self.assertIs(provenance[key], False)
        for name, value in provenance["output_sha256"].items():
            self.assertEqual(evaluation.rq1._sha(output / name), value)
        with self.assertRaises(FileExistsError):
            evaluation.run(self.root)

    def test_incomplete_protection(self):
        with patch.object(evaluation, "evaluate", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                evaluation.run(self.root)
        self.assertEqual(evaluation.rq1._read_json(self.root / evaluation.OUTPUT / "provenance.json")["status"], "incomplete")
        with self.assertRaises(FileExistsError):
            evaluation.run(self.root)

    def test_windows_gate_before_loading(self):
        with patch.object(evaluation.os, "name", "nt"), patch.object(evaluation, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                evaluation.main(["--dicc"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
