"""Tiny CPU-only fixtures; no real development/test evaluation."""

import copy
import importlib
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from trustpcb.rq3 import selective_review as review


class PackageCompatibilityTests(unittest.TestCase):
    def check_order(self, names):
        script = f'''
import importlib, sys
from unittest.mock import patch
first, second = [importlib.import_module(n) for n in {names!r}]
from trustpcb import rq3_final_evaluation as final
assert first is second is final.review
assert first.__name__ == 'trustpcb.rq3.selective_review'
for source, target in ((first, second), (second, first)):
    with patch.object(source, 'scores_for', return_value='synthetic') as score:
        assert target.scores_for([]) == final.review.scores_for([]) == 'synthetic'
        assert score.call_count == 2
    with patch.dict(source.DEFINITIONS, synthetic='shared'):
        assert target.DEFINITIONS['synthetic'] == final.review.DEFINITIONS['synthetic'] == 'shared'
assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules
'''
        subprocess.run([sys.executable, "-B", "-c", script], check=True, capture_output=True)

    def test_legacy_first(self):
        self.check_order(("trustpcb.rq3_selective_review", "trustpcb.rq3.selective_review"))

    def test_canonical_first(self):
        self.check_order(("trustpcb.rq3.selective_review", "trustpcb.rq3_selective_review"))

    def test_canonical_dependencies_and_direct_binding(self):
        from trustpcb.rq2 import risk_evaluation, weighted_fusion, weighted_final_evaluation
        self.assertIs(review.risk, risk_evaluation)
        self.assertIs(review.fusion, weighted_fusion)
        self.assertIs(review.selected_identity, weighted_final_evaluation.selected_identity)
        original = review.selected_identity
        with patch.object(weighted_final_evaluation, "selected_identity") as source:
            self.assertIs(review.selected_identity, original)
            self.assertIsNot(review.selected_identity, source)
        with patch.object(review, "selected_identity", return_value={"synthetic": "identity"}) as selected, \
             patch.object(review.risk, "load_inputs", return_value=(["a"], [], {})) as load, \
             patch.object(weighted_final_evaluation, "load_inputs", side_effect=AssertionError("no test population")):
            self.assertEqual(review.load_inputs(Path("synthetic")), (["a"], [], {}, {"synthetic": "identity"}))
            selected.assert_called_once_with(Path("synthetic"))
            load.assert_called_once_with(Path("synthetic"))

    def test_downstream_all_consumed_interfaces_and_bidirectional_patches(self):
        from trustpcb import rq3_final_evaluation as final
        legacy = importlib.import_module("trustpcb.rq3_selective_review")
        self.assertIs(final.review, review)
        names = ("OUTPUT", "SEED", "REPLICATES", "BUDGETS", "METHODS", "REFERRAL_BOOTSTRAP_METRICS",
                 "DEFINITIONS", "scores_for", "prepare", "curve", "threshold_for", "referral_metrics",
                 "tables", "bootstrap", "load_inputs", "extend_reporting", "run", "main",
                 "risk", "fusion", "rq1", "selected_identity")
        for name in names:
            self.assertIs(getattr(final.review, name), getattr(legacy, name))
        for owner, observer in ((review, final.review), (final.review, legacy)):
            for name in ("METHODS", "BUDGETS", "DEFINITIONS"):
                marker = object()
                with patch.object(owner, name, marker):
                    self.assertIs(getattr(observer, name), marker)
            for name in ("scores_for", "prepare", "curve", "referral_metrics", "bootstrap"):
                with patch.object(owner, name, return_value="synthetic") as helper:
                    self.assertEqual(getattr(observer, name)("fixture"), "synthetic")
                    helper.assert_called_once_with("fixture")
        images, rows = fixture()
        scores = review.scores_for(rows)
        thresholds = review.tables(images, rows, scores)[-1]
        with patch.object(legacy, "threshold_for", side_effect=AssertionError("no test retuning")), \
             patch.object(review, "prepare", wraps=review.prepare) as prepare, \
             patch.object(review, "curve", wraps=review.curve) as curve, \
             patch.object(review, "referral_metrics", wraps=review.referral_metrics) as referral:
            final.observed_tables(images, rows, scores, thresholds)
        prepare.assert_called_once()
        self.assertEqual(curve.call_count, 5)
        self.assertEqual(referral.call_count, 15)

    def test_both_cli_help_forms(self):
        for name in ("trustpcb.rq3_selective_review", "trustpcb.rq3.selective_review"):
            result = subprocess.run([sys.executable, "-B", "-m", name, "--help"],
                                    check=True, capture_output=True, text=True)
            for option in ("--dicc", "--bootstrap", "--extend-report"):
                self.assertIn(option, result.stdout)

    def test_legacy_execution_dispatches_canonical_main(self):
        with patch.object(review, "main") as main:
            runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/trustpcb/rq3_selective_review.py"),
                           run_name="__main__")
        main.assert_called_once_with()

    def test_parser_bootstrap_and_extension_dispatch(self):
        root = Path("synthetic")
        for flags in ([], ["--bootstrap"], ["--extend-report"], ["--extend-report", "--bootstrap"]):
            with self.subTest(flags=flags), patch.object(review.os, "name", "posix"), \
                 patch.object(review, "find_project_root", return_value=root), \
                 patch.object(review, "run") as run, patch.object(review, "extend_reporting") as extend, \
                 patch("builtins.print"):
                review.main(["--dicc", *flags])
                if "--extend-report" in flags:
                    extend.assert_called_once_with(root)
                    run.assert_not_called()
                else:
                    run.assert_called_once_with(root, with_bootstrap="--bootstrap" in flags)
                    extend.assert_not_called()

    def test_missing_dicc_blocks_before_root_access(self):
        with patch.object(review.os, "name", "posix"), patch.object(review, "find_project_root") as root:
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                review.main(["--bootstrap"])
        root.assert_not_called()

    def test_frozen_population_and_inclusion_validator_identity(self):
        self.assertEqual((review.risk.IMAGE_COUNT, review.risk.PREDICTION_COUNT), (821, 7131))
        images = [f"synthetic_{i}" for i in range(821)]
        template = fixture()[1][0]
        rows = [{**template, "image": images[i % 821], "prediction_id": str(i), "raw_confidence": .01}
                for i in range(7131)]
        self.assertEqual(len(review.risk.validate_rows(images, rows)), 7131)
        for bad_images, bad_rows in ((images[:-1], rows), (images, rows[:-1]),
                                    (images, [{**rows[0], "raw_confidence": .009}, *rows[1:]])):
            with self.assertRaises(ValueError):
                review.risk.validate_rows(bad_images, bad_rows)

    def test_frozen_defaults_and_method_order(self):
        self.assertEqual(review.bootstrap.__kwdefaults__, {"replicates": 10000, "seed": 24209199})
        self.assertEqual(review.run.__kwdefaults__, {"with_bootstrap": False})
        self.assertEqual(review.BUDGETS, (5, 10, 20))
        self.assertEqual(review.METHODS, ("raw_confidence_risk", "calibrated_confidence_risk",
            "class_consistency_risk", "localisation_stability_risk", "weighted_trustpcb_risk"))
        self.assertEqual(review.fusion.WEIGHT_KEYS, ("conf_percent", "class_percent", "localisation_percent"))


def fixture():
    images = [f"synthetic_{i}" for i in range(20)]
    rows = [{"image": images[i//2], "prediction_id": str(i), "raw_confidence": .1+i/25,
             "calibrated_confidence": .2+i/30, "class_consistency": .3+i/40,
             "localisation_stability": .4+i/50, "correct_iou50": i % 2}
            for i in range(10)]
    return images, rows


class CalculationTests(unittest.TestCase):
    def test_score_direction_and_frozen_formula(self):
        _, rows = fixture()
        scores = review.scores_for(rows)
        for key, field in zip(review.METHODS[:4], ("raw_confidence", *review.risk.SIGNALS)):
            np.testing.assert_allclose(scores[key], [1-r[field] for r in rows])
        np.testing.assert_allclose(scores[review.METHODS[-1]],
            [.89*(1-r["calibrated_confidence"])+.01*(1-r["class_consistency"])+.10*(1-r["localisation_stability"]) for r in rows])

    def test_curve_ties_error_rates_and_aurc(self):
        plan = (np.array([0, 1, 2]), np.array([1, 2]), np.array([.1, .8]))
        records, area = review.curve(plan, np.array([0, 1, 1]), np.ones(3))
        self.assertEqual(len(records), 3)
        self.assertIsNone(records[0]["empirical_risk"])
        self.assertAlmostEqual(records[1]["coverage"], 2/3)
        self.assertEqual(records[1]["empirical_risk"], .5)
        self.assertAlmostEqual(area, (2/3)*.5+(1/3)*(2/3))

    def test_empty_curve_and_zero_multiplicity_groups(self):
        plan = (np.array([0, 1]), np.array([0, 1]), np.array([.1, .9]))
        records, area = review.curve(plan, np.array([0, 1]), np.array([0, 2]))
        self.assertEqual(len(records), 2)
        self.assertEqual(area, 1)
        self.assertIsNone(review.curve(plan, np.array([0, 1]), np.zeros(2))[1])

    def test_crop_max_and_empty_accounting(self):
        images, rows = fixture()
        scores = review.scores_for(rows)
        _, _, counts, _, crops, _ = review.prepare(images, rows, scores)
        self.assertEqual(len(counts), 20)
        for method in review.METHODS:
            self.assertEqual(crops[method][0], max(scores[method][:2]))
            self.assertTrue(np.isneginf(crops[method][5:]).all())
        _, _, referrals, inventory, _ = review.tables(images, rows, scores)
        self.assertTrue(all(r["crop_risk"] is None and not r["referral_eligible"]
                            for r in inventory if r["prediction_count"] == 0))
        for r in referrals:
            self.assertEqual(r["workload"], r["referred_crops"]/20)
            self.assertLessEqual(r["referred_crops"], r["nominal_crops"])

    def test_strict_threshold_ties_and_nominal_821_counts(self):
        values = np.r_[.9, .8, .8, .7, np.full(16, -np.inf)]
        result = review.threshold_for(values, 10)
        self.assertEqual(result["nominal_crops"], 2)
        self.assertEqual(result["threshold"], .8)
        self.assertEqual(result["achieved_crops"], 1)
        self.assertEqual(review.threshold_for(np.zeros(821), 5)["nominal_crops"], 41)
        self.assertEqual(review.threshold_for(np.zeros(821), 10)["nominal_crops"], 82)
        self.assertEqual(review.threshold_for(np.zeros(821), 20)["nominal_crops"], 164)
        self.assertEqual(review.threshold_for(np.zeros(821), 20)["achieved_crops"], 0)
        self.assertEqual(review.threshold_for(np.full(20, -np.inf), 20)["achieved_crops"], 0)
        self.assertEqual(review.threshold_for(np.r_[.1, np.full(19, -np.inf)], 20)["achieved_crops"], 1)

    def test_thresholds_independent_of_labels_and_order(self):
        images, rows = fixture()
        before = review.tables(images, rows, review.scores_for(rows))[-1]
        changed = [{**r, "correct_iou50": 1-r["correct_iou50"]} for r in reversed(rows)]
        self.assertEqual(before, review.tables(list(reversed(images)), changed, review.scores_for(changed))[-1])

    def test_error_capture_random_expectation_and_undefined(self):
        result = review.referral_metrics(np.array([.9, .3, -np.inf]), np.array([3, 1, 0]),
            np.array([2, 1, 0]), .3, np.ones(3))
        self.assertEqual(result["referred_crops"], 1)
        self.assertAlmostEqual(result["workload"], 1/3)
        self.assertAlmostEqual(result["error_capture"], 2/3)
        self.assertEqual(result["random_expected_error_capture"], .5)
        self.assertEqual(result["random_expected_captured_incorrect"], 1.5)
        empty = review.referral_metrics(np.array([-np.inf]), np.array([0]), np.array([0]), -1, np.ones(1))
        self.assertEqual(empty["referred_crops"], 0)
        self.assertIsNone(empty["error_capture"])

    def test_foreign_duplicate_and_invalid_signals(self):
        images, rows = fixture()
        for altered in ([{**rows[0], "image": "foreign"}], [rows[0], rows[0]], [{**rows[0], "correct_iou50": .2}]):
            with self.assertRaises(ValueError):
                review.prepare(images, altered, review.scores_for(altered))
        scores = review.scores_for(rows)
        scores[review.METHODS[0]][0] = np.nan
        with self.assertRaises(ValueError):
            review.prepare(images, rows, scores)

    def test_automatic_error_counts_reduction_and_multiplicity(self):
        result = review.referral_metrics(np.array([.9, .3, -np.inf]), np.array([3, 2, 0]),
            np.array([2, 0, 0]), .3, np.array([2, 3, 1]))
        self.assertEqual(result["automatic_predictions"], 6)
        self.assertEqual(result["automatic_incorrect"], 0)
        self.assertEqual(result["remaining_automatic_error_rate"], 0)
        self.assertAlmostEqual(result["no_referral_error_rate"], 4/12)
        self.assertAlmostEqual(result["absolute_error_reduction"], 4/12)
        none = review.referral_metrics(np.array([.9]), np.array([3]), np.array([2]), 1, np.ones(1))
        self.assertEqual(none["absolute_error_reduction"], 0)
        all_referred = review.referral_metrics(np.array([.9]), np.array([3]), np.array([2]), 0, np.ones(1))
        self.assertEqual(all_referred["automatic_predictions"], 0)
        self.assertIsNone(all_referred["remaining_automatic_error_rate"])
        self.assertIsNone(all_referred["absolute_error_reduction"])
        worse = review.referral_metrics(np.array([.9, .1]), np.array([1, 1]), np.array([0, 1]), .5, np.ones(2))
        self.assertEqual(worse["absolute_error_reduction"], -.5)

    def test_paired_bootstrap_matches_explicit_shared_samples(self):
        images, rows = fixture()
        rows = [{**r, "image": images[i], "correct_iou50": [0, 1, 0][i]} for i, r in enumerate(rows[:3])]
        scores = {m: np.array([.9, .1, .8]) for m in review.METHODS}
        scores[review.METHODS[0]] = np.array([.1, .9, .2])
        scores[review.METHODS[1]] = np.array([.9, .2, .1])
        thresholds = review.tables(images, rows, scores)[-1]
        with patch.object(review, "threshold_for", side_effect=AssertionError("no retuning")):
            intervals, info = review.bootstrap(images, rows, scores, thresholds, replicates=17)
        self.assertEqual(len(info["paired_differences"]), 14)
        draws = np.random.Generator(np.random.PCG64(review.SEED)).integers(0, len(images), size=(17, len(images)))
        for pair in info["paired_differences"]:
            differences = []
            observed = []
            for draw in [np.arange(len(images)), *draws]:
                values = []
                for method in (pair["method_A"], pair["method_B"]):
                    copied = [dict(r, prediction_id=f"{j}:{r['prediction_id']}") for j, image_index in enumerate(draw)
                              for r in rows if r["image"] == images[image_index]]
                    copied_scores = {m: np.array([scores[m][int(r["prediction_id"].split(":")[-1])] for r in copied])
                                     for m in review.METHODS}
                    # Explicit duplicated crops/predictions, including empty crop copies.
                    copied_images = [f"copy_{i}" for i in range(len(draw))]
                    for r in copied:
                        r["image"] = copied_images[int(r["prediction_id"].split(":")[0])]
                    _, labels, counts, errors, crops, plans = review.prepare(copied_images, copied, copied_scores)
                    if pair["metric"] == "aurc":
                        value = review.curve(plans[method], labels, np.ones(len(copied)))[1]
                    else:
                        threshold = next(t["threshold"] for t in thresholds[method] if t["budget_percent"] == pair["budget_percent"])
                        value = review.referral_metrics(crops[method], counts, errors, threshold, np.ones(len(draw)))[pair["metric"]]
                    values.append(value)
                delta = values[0]-values[1] if all(v is not None for v in values) else np.nan
                if not observed:
                    observed.append(delta)
                else:
                    differences.append(delta)
            expected = review.risk.interval(differences)
            for key, value in expected.items():
                if value is None:
                    self.assertIsNone(pair[key])
                else:
                    self.assertAlmostEqual(pair[key], value)
            if np.isfinite(observed[0]):
                self.assertAlmostEqual(pair["observed_difference"], observed[0])
            else:
                self.assertIsNone(pair["observed_difference"])
        raw_pairs = [r for r in info["paired_differences"] if r["method_B"] == review.METHODS[0]]
        self.assertLess(next(r for r in raw_pairs if r["metric"] == "aurc")["observed_difference"], 0)
        self.assertGreater(next(r for r in raw_pairs if r["metric"] == "error_capture" and r["budget_percent"] == 5)["observed_difference"], 0)
        self.assertLess(next(r for r in raw_pairs if r["metric"] == "remaining_automatic_error_rate" and r["budget_percent"] == 5)["observed_difference"], 0)
        self.assertTrue(any(r["invalid_replicates"] for r in info["paired_differences"]))
        self.assertTrue(any(r["metric"] == "absolute_error_reduction" for r in intervals))

    def test_bootstrap_determinism_shared_draws_fixed_thresholds(self):
        images, rows = fixture()
        scores = review.scores_for(rows)
        thresholds = review.tables(images, rows, scores)[-1]
        before = copy.deepcopy(thresholds)
        with patch.object(review, "threshold_for", side_effect=AssertionError("no retuning")):
            a = review.bootstrap(images, rows, scores, thresholds, replicates=12)
            b = review.bootstrap(images, rows, scores, thresholds, replicates=12)
        self.assertEqual(a, b)
        self.assertEqual(thresholds, before)
        self.assertFalse(a[1]["thresholds_rederived"])
        self.assertEqual(a[1]["seed"], 24209199)
        import hashlib
        draws = np.random.Generator(np.random.PCG64(24209199)).integers(0, 20, size=(12, 20), dtype=np.int64)
        self.assertEqual(a[1]["draw_indices_sha256"], hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest())
        self.assertTrue(all(r["valid_replicates"]+r["invalid_replicates"] == 12 for r in a[0]))

    def test_bootstrap_all_invalid_capture(self):
        images, rows = fixture()
        rows = [{**r, "correct_iou50": 1} for r in rows]
        scores = review.scores_for(rows)
        thresholds = review.tables(images, rows, scores)[-1]
        intervals, _ = review.bootstrap(images, rows, scores, thresholds, replicates=3)
        for row in intervals:
            if "error_capture" in row["metric"]:
                self.assertEqual(row["invalid_replicates"], 3)
                self.assertIsNone(row["bootstrap_mean"])


class WorkflowTests(unittest.TestCase):
    def test_loader_uses_development_only_and_missing_inputs_fail(self):
        images, rows = fixture()
        with patch.object(review, "selected_identity", return_value={"synthetic": "identity"}), \
             patch.object(review.risk, "load_inputs", return_value=(images, rows, {})) as load:
            self.assertEqual(review.load_inputs("synthetic")[:2], (images, rows))
            load.assert_called_once_with("synthetic")
        with patch.object(review, "selected_identity", return_value={}), \
             patch.object(review.risk, "load_inputs", side_effect=FileNotFoundError("missing frozen artifacts")):
            with self.assertRaises(FileNotFoundError):
                review.load_inputs("synthetic")

    def test_output_provenance_determinism_and_overwrite_protection(self):
        images, rows = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentinel = root/"runs/rq2/synthetic_evidence.json"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b"immutable")
            with patch.object(review, "load_inputs", return_value=(images, rows, {"synthetic": "hash"}, {"evidence": "hash"})), \
                 patch.object(review.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"}), \
                 patch.object(review.fusion, "select", side_effect=AssertionError("no selection")), \
                 patch.object(review.risk.final.comparison, "fit_calibrator", side_effect=AssertionError("no fit")):
                output = review.run(root)
                with self.assertRaises(FileExistsError):
                    review.run(root)
            self.assertEqual(sentinel.read_bytes(), b"immutable")
            provenance = review.rq1._read_json(output/"provenance.json")
            self.assertEqual(provenance["status"], "complete")
            self.assertEqual(provenance["experiment"], "rq3_development_selective_review")
            self.assertFalse(provenance["test_population_accessed"])
            self.assertFalse(provenance["bootstrap_requested"])
            for name, digest in provenance["output_sha256"].items():
                self.assertEqual(review.rq1._sha(output/name), digest)
            thresholds = review.rq1._read_json(output/"development_thresholds.json")
            self.assertEqual(thresholds["comparison"], ">")
            self.assertFalse(thresholds["test_retuning_permitted"])
            curves = review.risk.read_csv(output/"prediction_risk_coverage.csv")
            self.assertEqual(list(curves[0]), ["method", "coverage", "retained_predictions",
                "retained_incorrect", "empirical_risk", "max_retained_score"])
            inventory = review.risk.read_csv(output/"crop_risks.csv")
            self.assertEqual(list(inventory[0]), ["method", "image", "prediction_count",
                "incorrect_predictions", "crop_risk", "referral_eligible"])
            self.assertEqual([(r["method"], r["image"]) for r in inventory],
                             [(method, image) for method in review.METHODS for image in images])
            referrals = review.risk.read_csv(output/"crop_referral_metrics.csv")
            self.assertEqual([(r["method"], int(r["budget_percent"])) for r in referrals],
                             [(method, budget) for method in review.METHODS for budget in review.BUDGETS])
            self.assertEqual(list(referrals[0]), ["method", "budget_percent", "nominal_crops", "threshold",
                "achieved_crops", "achieved_workload", "referred_crops", "workload", "referred_predictions",
                "captured_incorrect", "error_capture", "referred_error_rate", "automatic_predictions",
                "automatic_incorrect", "no_referral_error_rate", "remaining_automatic_error_rate",
                "absolute_error_reduction", "random_expected_error_capture",
                "random_expected_captured_incorrect", "random_expected_referred_predictions"])

    def test_partial_and_dirty_input_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(review.rq1, "git_provenance", return_value={"git_dirty": True}):
                with self.assertRaises(RuntimeError):
                    review.run(root)
            (root/review.OUTPUT).mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                review.run(root)

    def test_reporting_extension_preserves_evidence_and_saved_thresholds(self):
        images, rows = fixture()
        scores = review.scores_for(rows)
        thresholds = review.tables(images, rows, scores)[-1]
        intervals, info = review.bootstrap(images, rows, scores, thresholds, replicates=3)
        info["replicates"] = 10000  # Tiny synthetic stand-in for completed metadata.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(review, "load_inputs", return_value=(images, rows, {"synthetic": "hash"}, {"evidence": "hash"})), \
                 patch.object(review.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"}):
                original = review.run(root)
                review.rq1._write_json(original/"bootstrap_summary.json", info)
                source = review.rq1._read_json(original/"provenance.json")
                source.update(bootstrap_requested=True, bootstrap_replicates=10000)
                source["output_sha256"]["bootstrap_summary.json"] = review.rq1._sha(original/"bootstrap_summary.json")
                review.rq1._write_json(original/"provenance.json", source)
                before = {p.name: p.read_bytes() for p in original.iterdir()}
                with patch.object(review, "threshold_for", side_effect=AssertionError("saved thresholds only")), \
                     patch.object(review, "bootstrap", return_value=(intervals, info)) as boot:
                    output = review.extend_reporting(root)
                self.assertEqual(boot.call_args.args[3], thresholds)
                for name, content in before.items():
                    self.assertEqual((original/name).read_bytes(), content)
                self.assertEqual(review.rq1._read_json(output/"provenance.json")["status"], "complete")
                self.assertEqual(review.rq1._read_json(output/"provenance.json")["experiment"],
                                 "rq3_development_reporting_extension")
                self.assertEqual(len(review.risk.read_csv(output/"paired_bootstrap_differences.csv")), 14)
                pairs = review.risk.read_csv(output/"paired_bootstrap_differences.csv")
                self.assertEqual(list(pairs[0]), ["method_A", "method_B", "budget_percent", "metric",
                    "observed_difference", "bootstrap_mean", "lower_95", "upper_95",
                    "valid_replicates", "invalid_replicates"])
                with self.assertRaises(FileExistsError):
                    review.extend_reporting(root)

    def test_reporting_extension_rejects_different_draws(self):
        images, rows = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(review, "load_inputs", return_value=(images, rows, {}, {})), \
                 patch.object(review.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"}):
                original = review.run(root)
                review.rq1._write_json(original/"bootstrap_summary.json", {"seed": review.SEED, "replicates": 10000, "draw_indices_sha256": "original"})
                source = review.rq1._read_json(original/"provenance.json")
                source.update(bootstrap_requested=True, bootstrap_replicates=10000)
                source["output_sha256"]["bootstrap_summary.json"] = review.rq1._sha(original/"bootstrap_summary.json")
                review.rq1._write_json(original/"provenance.json", source)
                with patch.object(review, "bootstrap", return_value=([], {"seed": review.SEED, "replicates": 10000, "draw_indices_sha256": "different"})):
                    with self.assertRaisesRegex(ValueError, "draws differ"):
                        review.extend_reporting(root)
                self.assertEqual(review.rq1._read_json(original/"reporting_extension/provenance.json")["status"], "incomplete")

    def test_windows_gate(self):
        with patch.object(review.os, "name", "nt"), patch.object(review, "run") as run:
            with self.assertRaises(RuntimeError):
                review.main(["--dicc"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
