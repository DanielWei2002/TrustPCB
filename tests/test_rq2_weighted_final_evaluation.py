"""Synthetic text and ranking fixtures only; never real final-test analysis."""

import copy
import csv
import importlib
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustpcb.rq2 import weighted_final_evaluation as evaluation


class PackageCompatibilityTests(unittest.TestCase):
    def test_fresh_import_orders_identity_and_bidirectional_patches(self):
        names = ("trustpcb.rq2_weighted_final_evaluation", "trustpcb.rq2.weighted_final_evaluation")
        for order in (names, names[::-1]):
            with self.subTest(order=order):
                script = f"""
import importlib, sys
from unittest.mock import patch
a, b = [importlib.import_module(name) for name in {order!r}]
assert a is b
for source, target in ((a, b), (b, a)):
    with patch.object(source, 'load_inputs', return_value='synthetic') as loader:
        assert target.load_inputs(None) == 'synthetic'
        loader.assert_called_once_with(None)
assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules
"""
                subprocess.run([sys.executable, "-B", "-c", script], check=True, capture_output=True)

    def test_canonical_dependencies_and_frozen_constants(self):
        for alias, name in (("risk", "risk_evaluation"), ("fusion", "weighted_fusion"),
                            ("final", "final_evaluation"), ("sensitivity", "risk_sensitivity")):
            self.assertIs(getattr(evaluation, alias), importlib.import_module(f"trustpcb.rq2.{name}"))
        self.assertIs(evaluation.resolve_input, importlib.import_module("trustpcb.rq2.artifact_paths").resolve_input)
        self.assertEqual(evaluation.WEIGHTS, (89, 1, 10))
        self.assertEqual(evaluation.fusion.WEIGHT_KEYS, ("conf_percent", "class_percent", "localisation_percent"))
        self.assertEqual(evaluation.risk.SIGNALS, ("calibrated_confidence", "class_consistency", "localisation_stability"))
        self.assertEqual((evaluation.SEED, evaluation.REPLICATES, evaluation.PREDICTION_COUNT), (24209199, 10000, 17840))
        self.assertEqual(evaluation.OUTPUT, "runs/rq2/weighted_final_evaluation")
        self.assertEqual(evaluation.EVIDENCE_COMMIT, "74510277056c72114dc421bd900ddb5dfbc91933")

    def test_rq3_final_loader_and_validator_through_both_names(self):
        from trustpcb import rq3_final_evaluation as consumer
        legacy = importlib.import_module("trustpcb.rq2_weighted_final_evaluation")
        self.assertIs(consumer.frozen, evaluation)
        images, rows = fixture()
        identity = {"synthetic": "identity"}
        for source in (legacy, evaluation):
            with patch.object(source, "validate_population", return_value=rows) as validator:
                self.assertIs(consumer.validate_population(images, rows), rows)
                validator.assert_called_once_with(images, rows)
            with patch.object(source, "load_inputs", return_value=(images, rows, {"input": "hash"}, identity)) as loader, \
                 patch.object(source, "validate_population", return_value=rows), \
                 patch.object(consumer, "load_thresholds", return_value=({}, {"threshold": "hash"}, {"selected_weight_identity": identity})):
                self.assertEqual(consumer.load_inputs(Path("synthetic")),
                                 (images, rows, {"input": "hash", "threshold": "hash"}, identity, {}))
                loader.assert_called_once_with(Path("synthetic"))
        with patch.object(consumer.frozen, "WEIGHTS", (1, 2, 97)):
            self.assertEqual(legacy.WEIGHTS, (1, 2, 97))
            self.assertEqual(evaluation.WEIGHTS, (1, 2, 97))

    def test_rq3_direct_symbol_imports_share_live_globals_and_evidence_guards(self):
        from trustpcb import rq3_selective_review as review
        from trustpcb import rq3_computational_efficiency as efficiency
        legacy = importlib.import_module("trustpcb.rq2_weighted_final_evaluation")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / evaluation.SELECTED
            path.parent.mkdir(parents=True)
            evaluation.rq1._write_json(path, selected())
            for consumer in (review, efficiency):
                self.assertIs(consumer.selected_identity, evaluation.selected_identity)
                self.assertIs(consumer.selected_identity.__globals__, evaluation.__dict__)
                for source in (legacy, evaluation):
                    with patch.object(source, "resolve_input", return_value=path) as resolver, \
                         patch.object(source.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=path.read_bytes())):
                        self.assertEqual(consumer.selected_identity(root)["sha256"], evaluation.rq1._sha(path))
                        resolver.assert_called_once_with(root, evaluation.SELECTED)
                    with patch.dict(consumer.selected_identity.__globals__, {"EVIDENCE_COMMIT": "synthetic"}):
                        self.assertEqual(source.EVIDENCE_COMMIT, "synthetic")
                    with patch.object(source.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=b"mismatch")):
                        with self.assertRaisesRegex(ValueError, "evidence commit"):
                            consumer.selected_identity(root)

    def test_both_module_help_commands(self):
        for name in ("trustpcb.rq2_weighted_final_evaluation", "trustpcb.rq2.weighted_final_evaluation"):
            result = subprocess.run([sys.executable, "-B", "-m", name, "--help"],
                                    check=True, capture_output=True, text=True)
            self.assertIn("--dicc", result.stdout)

    def test_legacy_execution_dispatches_canonical_main(self):
        with patch.object(evaluation, "main") as main:
            runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/trustpcb/rq2_weighted_final_evaluation.py"),
                           run_name="__main__")
        main.assert_called_once_with()

    def test_parser_dispatch_and_missing_flag(self):
        root = Path("synthetic")
        with patch.object(evaluation.os, "name", "posix"), \
             patch.object(evaluation, "find_project_root", return_value=root) as find, \
             patch.object(evaluation, "run", return_value="synthetic_output") as run, patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                evaluation.main([])
            find.assert_not_called()
            run.assert_not_called()
            evaluation.main(["--dicc"])
            run.assert_called_once_with(root)


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
            git.assert_called_once_with(
                ["git", "show", f"{evaluation.EVIDENCE_COMMIT}:{evaluation.SELECTED}"],
                cwd=root, check=True, capture_output=True)
            self.assertNotIn("-C", git.call_args.args[0])
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


class ArtifactAlignmentTests(unittest.TestCase):
    """Exercise the real text loader using only three synthetic prediction rows."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.folder = self.root / evaluation.final.OUTPUT
        (self.folder / "primary_iou50").mkdir(parents=True)
        self.images = [f"synthetic_{i}" for i in range(2052)]
        _, self.rows = fixture()
        for i, row in enumerate(self.rows):
            row["image"] = self.images[i]
        self.labels = copy.deepcopy(self.rows)
        for row in self.rows:
            row["legacy_extra_risk"] = .123
        self.provenance = {"status": "complete", "test_role": evaluation.final.ROLE,
            "test_images": 2052, "test_manifest": evaluation.final.MANIFEST,
            "test_manifest_sha256_lf": evaluation.final.MANIFEST_SHA, "reference_threshold": .01,
            "checkpoint_identity": {"sha256": evaluation.final.raw.CHECKPOINT_SHA},
            "beta_parameters": list(evaluation.final.BETA)}
        for relative in (evaluation.final.MANIFEST, evaluation.final.PARTITION_REPORT,
                         *(s[0] for s in evaluation.final.raw.rq2_detector_training.MANIFESTS.values())):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic manifest bytes\n", encoding="utf-8")
        for mocked in (patch.object(evaluation, "selected_identity", return_value={"sha256": "synthetic"}),
                       patch.object(evaluation.final, "frozen_test_images", return_value=self.images),
                       patch.object(evaluation, "PREDICTION_COUNT", 3)):
            mocked.start()
            self.addCleanup(mocked.stop)
        self.save()

    def save(self):
        # Rewrite only this test's temporary fixtures to simulate rehashed tampering.
        for name, rows in (("prediction_stability_scores.csv", self.rows),
                           ("labelled_predictions.csv", self.labels)):
            with (self.folder / name).open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        evaluation.rq1._write_json(self.folder / "primary_iou50/summary.json",
            {"test_images": 2052, "prediction_count": 3, "test_role": evaluation.final.ROLE})
        self.provenance["output_sha256"] = {name: evaluation.rq1._sha(self.folder / name) for name in
            ("prediction_stability_scores.csv", "labelled_predictions.csv", "primary_iou50/summary.json")}
        evaluation.rq1._write_json(self.folder / "provenance.json", self.provenance)

    def test_load_preserves_row_order_labels_and_signals(self):
        images, rows, hashes, identity = evaluation.load_inputs(self.root)
        self.assertEqual(images, self.images)
        self.assertEqual([r["prediction_id"] for r in rows], [r["prediction_id"] for r in self.rows])
        for actual, expected in zip(rows, self.rows):
            for key in ("raw_confidence", *evaluation.risk.SIGNALS, "correct_iou50", "incorrect_iou75"):
                self.assertEqual(actual[key], expected[key])
        self.assertEqual(identity, {"sha256": "synthetic"})
        self.assertIn(f"{evaluation.final.OUTPUT}/labelled_predictions.csv", hashes)

    def test_rehashed_reordered_or_missing_rows_fail(self):
        original = copy.deepcopy(self.labels)
        for labels in (original[::-1], original[:-1]):
            self.labels = labels
            self.save()
            with self.assertRaisesRegex(ValueError, "fields/order"):
                evaluation.load_inputs(self.root)

    def test_rehashed_label_confidence_or_stability_misalignment_fails(self):
        original = copy.deepcopy(self.labels)
        for key in ("correct_iou50", "calibrated_confidence", "class_consistency", "localisation_stability"):
            self.labels = copy.deepcopy(original)
            self.labels[0][key] = 1 - self.labels[0][key] if key != "localisation_stability" else .6
            self.save()
            with self.assertRaisesRegex(ValueError, "fields/order"):
                evaluation.load_inputs(self.root)

    def test_hash_and_completion_guards(self):
        with (self.folder / "labelled_predictions.csv").open("a") as file:
            file.write("tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            evaluation.load_inputs(self.root)
        self.provenance["status"] = "incomplete"
        self.save()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            evaluation.load_inputs(self.root)


class MetricTests(unittest.TestCase):
    def test_exact_weighted_score_and_equal_baseline(self):
        _, rows = fixture()
        scores = evaluation.scores_for(rows)
        self.assertAlmostEqual(scores["weighted_trustpcb_risk"][0], .89*.8 + .01*.2 + .10*.5)
        self.assertEqual(scores["equal_weight_trustpcb_risk"][0], evaluation.risk.risk_scores(rows)["trustpcb_risk"][0])

    def test_component_order_endpoints_and_no_clipping_or_imputation(self):
        _, rows = fixture()
        self.assertEqual(tuple(evaluation.scores_for(rows)), evaluation.METHODS)
        for values, expected in (((0., 1., 1.), .89), ((1., 0., 1.), .01),
                                 ((1., 1., 0.), .10), ((1., 1., 1.), 0.), ((0., 0., 0.), 1.)):
            row = {**rows[0], **dict(zip(evaluation.risk.SIGNALS, values))}
            self.assertEqual(evaluation.scores_for([row])[evaluation.METHODS[2]][0], expected)
        for value in (-.1, 1.1, float("nan"), None):
            with self.assertRaises(ValueError):
                evaluation.scores_for([{**rows[0], "class_consistency": value}])
        missing = dict(rows[0])
        del missing["class_consistency"]
        with self.assertRaises(KeyError):
            evaluation.scores_for([missing])

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
        self.assertEqual(provenance["experiment"], "rq2_weighted_final_evaluation")
        self.assertEqual(provenance["risk_formula"], evaluation.FORMULA)
        self.assertEqual({p.name for p in output.iterdir()}, {
            "primary_method_metrics.csv", "primary_pairwise_differences.csv",
            "sensitivity_iou75_method_metrics.csv", "sensitivity_iou75_pairwise_differences.csv",
            "summary.json", "provenance.json"})
        for prefix in ("primary", "sensitivity_iou75"):
            pairs = evaluation.risk.read_csv(output / f"{prefix}_pairwise_differences.csv")
            self.assertEqual(len(pairs), 4)
            self.assertEqual({r["metric"] for r in pairs}, {"auroc", "auprc"})
            self.assertEqual(list(pairs[0]), ["analysis_label", "method_A", "method_B", "metric",
                "observed_difference", "bootstrap_mean", "lower_95", "upper_95", "valid_replicates", "invalid_replicates"])
            self.assertEqual([(r["method_B"], r["metric"]) for r in pairs],
                [(method, metric) for method in evaluation.METHODS[:2] for metric in ("auroc", "auprc")])
            methods = evaluation.risk.read_csv(output / f"{prefix}_method_metrics.csv")
            self.assertEqual([r["method"] for r in methods], list(evaluation.METHODS))
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
