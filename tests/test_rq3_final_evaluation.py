"""Synthetic final-evaluator tests; never read real test predictions."""

import copy
from contextlib import ExitStack
import hashlib
import importlib
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from trustpcb.rq3 import final_evaluation as evaluation


class PackageCompatibilityTests(unittest.TestCase):
    def check_order(self, names):
        script = f'''
import importlib, sys
from unittest.mock import patch
first, second = [importlib.import_module(n) for n in {names!r}]
assert first is second
assert first.__name__ == 'trustpcb.rq3.final_evaluation'
for owner, observer in ((first, second), (second, first)):
    with patch.object(owner, 'load_inputs', return_value='synthetic') as load:
        assert observer.load_inputs(None) == 'synthetic'
        load.assert_called_once_with(None)
    with patch.dict(owner.FLAGS, synthetic=False):
        assert observer.FLAGS['synthetic'] is False
assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules
'''
        subprocess.run([sys.executable, "-B", "-c", script], check=True, capture_output=True)

    def test_legacy_first(self):
        self.check_order(("trustpcb.rq3_final_evaluation", "trustpcb.rq3.final_evaluation"))

    def test_canonical_first(self):
        self.check_order(("trustpcb.rq3.final_evaluation", "trustpcb.rq3_final_evaluation"))

    def test_canonical_aliases_and_review_patch_delegation(self):
        from trustpcb import rq1, rq3_selective_review as legacy_review
        from trustpcb.rq3 import selective_review
        from trustpcb.rq2 import weighted_final_evaluation
        self.assertIs(evaluation.rq1, rq1)
        self.assertIs(evaluation.review, selective_review)
        self.assertIs(evaluation.frozen, weighted_final_evaluation)
        self.assertIs(evaluation.review.fusion, evaluation.frozen.fusion)
        self.assertIs(evaluation.review.risk, evaluation.frozen.risk)
        images, rows, thresholds = fixture()
        for owner in (legacy_review, selective_review):
            with patch.object(owner, "prepare", wraps=owner.prepare) as prepare, \
                 patch.object(owner, "curve", wraps=owner.curve) as curve, \
                 patch.object(owner, "referral_metrics", wraps=owner.referral_metrics) as referral, \
                 patch.object(owner, "threshold_for", side_effect=AssertionError("no retuning")):
                evaluation.observed_tables(images, rows, owner.scores_for(rows), thresholds)
            prepare.assert_called_once()
            self.assertEqual(curve.call_count, 5)
            self.assertEqual(referral.call_count, 15)

    def test_frozen_constants_and_identity_delegation(self):
        self.assertEqual((evaluation.IMAGE_COUNT, evaluation.PREDICTION_COUNT), (2052, 17840))
        self.assertEqual((evaluation.SEED, evaluation.REPLICATES), (24209199, 10000))
        self.assertEqual(evaluation.THRESHOLDS, "runs/rq3/development_selective_review/development_thresholds.json")
        self.assertEqual(evaluation.OUTPUT, "runs/rq3/final_evaluation")
        self.assertEqual(evaluation.frozen.final.MANIFEST, "configs/datasets/similarity_aware_val_images.txt")
        self.assertEqual(evaluation.frozen.final.MANIFEST_SHA, "a2f246bf5be60cad5fac7f985ed91888bddb43601b73331f39d02e180075bc8e")
        self.assertEqual(evaluation.frozen.final.PARTITION_REPORT, "data/splits/rq2_train_development_split/verification_report.json")
        self.assertEqual(evaluation.frozen.WEIGHTS, (89, 1, 10))
        self.assertEqual(evaluation.review.fusion.WEIGHT_KEYS, ("conf_percent", "class_percent", "localisation_percent"))
        self.assertEqual(evaluation.review.METHODS, ("raw_confidence_risk", "calibrated_confidence_risk",
            "class_consistency_risk", "localisation_stability_risk", "weighted_trustpcb_risk"))
        with patch.object(evaluation.frozen, "validate_population", return_value="checked") as validate:
            self.assertEqual(evaluation.validate_population([], []), "checked")
            validate.assert_called_once_with([], [])

    def test_both_cli_help_forms(self):
        for module in ("trustpcb.rq3_final_evaluation", "trustpcb.rq3.final_evaluation"):
            result = subprocess.run([sys.executable, "-B", "-m", module, "--help"],
                                    check=True, capture_output=True, text=True)
            self.assertIn("--dicc", result.stdout)

    def test_legacy_cli_dispatches_canonical_main(self):
        with patch.object(evaluation, "main") as main:
            runpy.run_path(str(Path(__file__).resolve().parents[1]/"src/trustpcb/rq3_final_evaluation.py"),
                           run_name="__main__")
        main.assert_called_once_with()

    def test_parser_dispatch_and_missing_dicc_guard(self):
        root = Path("synthetic")
        with patch.object(evaluation.os, "name", "posix"), \
             patch.object(evaluation, "find_project_root", return_value=root) as find, \
             patch.object(evaluation, "run") as run, patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "--dicc required"):
                evaluation.main([])
            find.assert_not_called()
            run.assert_not_called()
            evaluation.main(["--dicc"])
            run.assert_called_once_with(root)


def fixture():
    images = ["synthetic_a", "synthetic_b", "synthetic_empty"]
    rows = [{"image": image, "prediction_id": str(i), "raw_confidence": confidence,
             "calibrated_confidence": confidence, "class_consistency": .7,
             "localisation_stability": .8, "correct_iou50": correct, "correct_iou75": 0,
             "trustpcb_risk": 999} for i, (image, confidence, correct) in enumerate(
                 ((images[0], .1, 0), (images[0], .7, 1), (images[1], .5, 0)))]
    thresholds = {m: [{"budget_percent": b, "threshold": t, "nominal_crops": 821*b//100,
                       "achieved_crops": 0, "achieved_workload": 0.0}
                      for b, t in ((5, .5), (10, .4), (20, .2))] for m in evaluation.review.METHODS}
    return images, rows, thresholds


def save_thresholds(root):
    thresholds = fixture()[2]
    path = root/evaluation.THRESHOLDS
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.rq1._write_json(path, {"thresholds": thresholds, "comparison": ">",
        "derived_from": "development scores only", "test_retuning_permitted": False})
    provenance = {"status": "complete", "development_images": 821, "prediction_count": 7131,
        "weights": [.89, .01, .10], "budgets_percent": [5, 10, 20], "selected_weight_identity": {"synthetic": "weight"},
        "output_sha256": {path.name: evaluation.rq1._sha(path)}}
    evaluation.rq1._write_json(path.parent/"provenance.json", provenance)
    return thresholds


class ValidationTests(unittest.TestCase):
    def test_original_threshold_evidence_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = save_thresholds(root)
            original_read = evaluation.rq1._read_json
            with patch.object(evaluation.rq1, "_read_json", wraps=original_read) as read:
                self.assertEqual(evaluation.load_thresholds(root)[0], expected)
            self.assertEqual([call.args[0] for call in read.call_args_list],
                             [root/evaluation.DEVELOPMENT/"provenance.json", root/evaluation.THRESHOLDS])
            # A reporting extension cannot substitute for missing original provenance.
            provenance = root/evaluation.DEVELOPMENT/"provenance.json"
            extension = root/evaluation.DEVELOPMENT/"reporting_extension"
            extension.mkdir()
            (extension/"provenance.json").write_bytes(provenance.read_bytes())
            provenance.unlink()
            with self.assertRaises(FileNotFoundError):
                evaluation.load_thresholds(root)

    def test_incomplete_development_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_thresholds(root)
            path = root/evaluation.DEVELOPMENT/"provenance.json"
            record = evaluation.rq1._read_json(path)
            record["status"] = "incomplete"
            evaluation.rq1._write_json(path, record)
            with self.assertRaisesRegex(ValueError, "completed frozen"):
                evaluation.load_thresholds(root)

    def test_exact_saved_thresholds_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            thresholds = save_thresholds(root)
            loaded, hashes, _ = evaluation.load_thresholds(root)
            self.assertEqual(loaded, thresholds)
            self.assertEqual(hashes[evaluation.THRESHOLDS], evaluation.rq1._sha(root/evaluation.THRESHOLDS))

    def test_modified_threshold_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_thresholds(root)
            with (root/evaluation.THRESHOLDS).open("a") as file:
                file.write(" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                evaluation.load_thresholds(root)

    def test_wrong_protocol_and_budget_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for change in ("comparison", "budget"):
                save_thresholds(root)
                path = root/evaluation.THRESHOLDS
                saved = evaluation.rq1._read_json(path)
                if change == "comparison":
                    saved["comparison"] = ">="
                else:
                    saved["thresholds"][evaluation.review.METHODS[0]][0]["budget_percent"] = 6
                evaluation.rq1._write_json(path, saved)
                provenance_path = path.parent/"provenance.json"
                provenance = evaluation.rq1._read_json(provenance_path)
                provenance["output_sha256"][path.name] = evaluation.rq1._sha(path)
                evaluation.rq1._write_json(provenance_path, provenance)
                with self.assertRaises(ValueError):
                    evaluation.load_thresholds(root)

    def test_population_counts_and_fields(self):
        images = [f"synthetic_{i}" for i in range(2052)]
        template = fixture()[1][0]
        rows = [{**template, "image": images[i % 2052], "prediction_id": str(i)} for i in range(17840)]
        self.assertEqual(len(evaluation.validate_population(images, rows)), 17840)
        for bad_images, bad_rows in ((images[:-1], rows), (images, rows[:-1]),
                (images, [{**rows[0], "image": "foreign"}, *rows[1:]])):
            with self.assertRaises(ValueError):
                evaluation.validate_population(bad_images, bad_rows)

    def test_mismatched_weight_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_thresholds(root)
            images, rows, _ = fixture()
            with patch.object(evaluation.frozen, "load_inputs", return_value=(images, rows, {}, {"different": "weight"})), \
                 patch.object(evaluation, "validate_population", return_value=rows):
                with self.assertRaisesRegex(ValueError, "weight identity"):
                    evaluation.load_inputs(root)


class CalculationTests(unittest.TestCase):
    def test_legacy_equal_weight_field_cannot_supply_weighted_score(self):
        _, rows, _ = fixture()
        scores = evaluation.review.scores_for(rows)
        expected = [(.89*(1-r["calibrated_confidence"])+.01*(1-r["class_consistency"])+.10*(1-r["localisation_stability"])) for r in rows]
        np.testing.assert_allclose(scores["weighted_trustpcb_risk"], expected)
        changed = [{**r, "trustpcb_risk": -999} for r in rows]
        np.testing.assert_array_equal(scores["weighted_trustpcb_risk"], evaluation.review.scores_for(changed)["weighted_trustpcb_risk"])

    def test_strict_referral_empty_denominator_and_no_test_workload_forcing(self):
        images, rows, thresholds = fixture()
        before = copy.deepcopy(thresholds)
        with patch.object(evaluation.review, "threshold_for", side_effect=AssertionError("no threshold derivation")), \
             patch.object(evaluation.review, "tables", side_effect=AssertionError("no development tables")):
            curves, areas, crops, referrals = evaluation.observed_tables(images, rows, evaluation.review.scores_for(rows), thresholds)
        record = next(r for r in referrals if r["method"] == "raw_confidence_risk" and r["budget_percent"] == 5)
        self.assertEqual(record["threshold"], .5)
        self.assertEqual(record["referred_crops"], 1)  # Equal .5 risk stays automatic.
        self.assertAlmostEqual(record["workload"], 1/3)  # May exceed nominal 5% on test.
        self.assertEqual(record["automatic_predictions"], 1)
        self.assertEqual(record["automatic_incorrect"], 1)
        self.assertEqual(record["remaining_automatic_error_rate"], 1)
        self.assertEqual(record["development_nominal_crops"], 41)
        self.assertEqual(record["development_achieved_crops"], 0)
        self.assertEqual(record["development_achieved_workload"], 0)
        self.assertEqual(list(curves[0]), ["method", "coverage", "retained_predictions", "retained_incorrect",
                                           "empirical_risk", "max_retained_score"])
        self.assertEqual(list(areas[0]), ["method", "aurc"])
        self.assertEqual(list(crops[0]), ["method", "image", "prediction_count", "incorrect_predictions",
                                        "crop_risk", "referral_eligible"])
        self.assertEqual(len(crops), 3*5)
        self.assertTrue(all(r["crop_risk"] is None and not r["referral_eligible"] for r in crops if r["image"] == "synthetic_empty"))
        self.assertEqual(len(areas), 5)
        self.assertEqual(thresholds, before)
        self.assertTrue(curves)

    def test_shared_whole_crop_bootstrap_with_fixed_thresholds(self):
        images, rows, thresholds = fixture()
        scores = evaluation.review.scores_for(rows)
        with patch.object(evaluation.review, "threshold_for", side_effect=AssertionError("no rederivation")):
            intervals, info = evaluation.review.bootstrap(images, rows, scores, thresholds, replicates=13, seed=evaluation.SEED)
        draws = np.random.Generator(np.random.PCG64(24209199)).integers(0, 3, size=(13, 3), dtype=np.int64)
        self.assertEqual(info["draw_indices_sha256"], hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest())
        # First crop carries two predictions; both must move together in every draw.
        retained = [float(np.count_nonzero(draw == 1)) for draw in draws]
        expected = evaluation.review.risk.interval(retained)
        record = next(r for r in intervals if r["method"] == "raw_confidence_risk" and r["budget_percent"] == 5 and r["metric"] == "automatic_predictions")
        for key, value in expected.items():
            self.assertEqual(record[key], value)
        self.assertEqual(len(info["paired_differences"]), 14)
        self.assertFalse(info["thresholds_rederived"])

    def test_iou75_does_not_change_primary_outputs(self):
        images, rows, thresholds = fixture()
        changed = [{**r, "correct_iou75": 1-r["correct_iou75"]} for r in rows]
        self.assertEqual(evaluation.observed_tables(images, rows, evaluation.review.scores_for(rows), thresholds),
                         evaluation.observed_tables(images, changed, evaluation.review.scores_for(changed), thresholds))


class WorkflowTests(unittest.TestCase):
    def test_outputs_provenance_and_no_forbidden_operations(self):
        images, rows, thresholds = fixture()
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as forbidden:
            root = Path(tmp)
            evidence = root/"runs/rq3/development_selective_review/synthetic_evidence.txt"
            evidence.parent.mkdir(parents=True)
            evidence.write_bytes(b"frozen")
            from trustpcb.rq3 import computational_efficiency
            from trustpcb.rq2 import transformation_stability, calibrator_comparison
            from trustpcb.rq2 import confidence_distribution
            for module, names in ((evaluation.review, ("tables", "run", "extend_reporting")),
                    (calibrator_comparison, ("apply_calibrator", "run")),
                    (transformation_stability, ("transform", "_predict", "run")),
                    (confidence_distribution, ("_predict",)),
                    (evaluation.frozen, ("run",)),
                    (computational_efficiency, ("run", "measure", "adapter"))):
                for name in names:
                    forbidden.enter_context(patch.object(module, name, side_effect=AssertionError(f"forbidden {name}")))
            with patch.object(evaluation, "load_inputs", return_value=(images, rows, {evaluation.THRESHOLDS: "synthetic_sha"}, {"evidence": "frozen"}, thresholds)), \
                 patch.object(evaluation.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"}), \
                 patch.object(evaluation, "REPLICATES", 5), \
                 patch.object(evaluation.review, "threshold_for", side_effect=AssertionError("no threshold derivation")), \
                 patch.object(evaluation.review.fusion, "select", side_effect=AssertionError("no weight selection")), \
                 patch.object(evaluation.review.risk.final.comparison, "fit_calibrator", side_effect=AssertionError("no fitting")):
                output = evaluation.run(root)
                with self.assertRaises(FileExistsError):
                    evaluation.run(root)
            self.assertEqual(evidence.read_bytes(), b"frozen")
            self.assertEqual({p.name for p in output.iterdir()}, {"prediction_risk_coverage.csv", "aurc.csv", "crop_risks.csv",
                "crop_referral_metrics.csv", "bootstrap_intervals.csv", "paired_bootstrap_differences.csv",
                "bootstrap_summary.json", "summary.json", "provenance.json"})
            provenance = evaluation.rq1._read_json(output/"provenance.json")
            self.assertEqual(provenance["status"], "complete")
            self.assertEqual(provenance["experiment"], "rq3_final_evaluation")
            self.assertEqual(provenance["correctness_definition"], "same-class one-to-one IoU >= 0.50")
            self.assertEqual(provenance["test_role"], evaluation.ROLE)
            self.assertEqual(provenance["applied_thresholds"], thresholds)
            self.assertEqual(provenance["development_threshold_sha256"], "synthetic_sha")
            self.assertEqual(provenance["bootstrap_seed"], 24209199)
            self.assertEqual(provenance["git_commit"], "synthetic")
            for flag in evaluation.FLAGS:
                self.assertIs(provenance[flag], False)
            for name, digest in provenance["output_sha256"].items():
                self.assertEqual(evaluation.rq1._sha(output/name), digest)

    def test_failure_leaves_incomplete_and_blocks_retry(self):
        images, rows, thresholds = fixture()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(evaluation, "load_inputs", return_value=(images, rows, {evaluation.THRESHOLDS: "sha"}, {}, thresholds)), \
             patch.object(evaluation.rq1, "git_provenance", return_value={"git_dirty": False}), \
             patch.object(evaluation.review, "bootstrap", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                evaluation.run(tmp)
            self.assertEqual(evaluation.rq1._read_json(Path(tmp)/evaluation.OUTPUT/"provenance.json")["status"], "incomplete")
            with self.assertRaises(FileExistsError):
                evaluation.run(tmp)

    def test_dirty_inputs_and_windows_gate(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(evaluation.rq1, "git_provenance", return_value={"git_dirty": True}), \
             patch.object(evaluation, "load_inputs") as load:
            with self.assertRaises(RuntimeError):
                evaluation.run(tmp)
            load.assert_not_called()
        with patch.object(evaluation.os, "name", "nt"), patch.object(evaluation, "run") as run:
            with self.assertRaises(RuntimeError):
                evaluation.main(["--dicc"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
