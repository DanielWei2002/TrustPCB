"""Tiny synthetic CV only; full grid enumeration is text/arithmetic, not a run."""

import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
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


class FusionCompatibilityTests(unittest.TestCase):
    def python(self, args):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), env.get("PYTHONPATH", "")))
        result = subprocess.run([sys.executable, "-B", *args], cwd=root, env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def imports(self, first, second):
        self.python(["-c", f"""
import importlib
import sys
from pathlib import Path
from unittest.mock import patch
first = importlib.import_module({first!r})
second = importlib.import_module({second!r})
from trustpcb import rq2_weighted_fusion as legacy
from trustpcb.rq2 import weighted_fusion as canonical
from trustpcb.rq2 import risk_evaluation, risk_sensitivity, calibrator_comparison
from trustpcb import rq2_final_evaluation as evaluation, rq2_weighted_final_evaluation as weighted
from trustpcb import rq3_selective_review as review, rq3_computational_efficiency as efficiency
from trustpcb import rq3_final_evaluation as rq3
assert first is second is legacy is canonical
assert sys.modules["trustpcb.rq2_weighted_fusion"] is canonical
assert sys.modules["trustpcb.rq2.weighted_fusion"] is canonical
assert Path(canonical.__file__).resolve() == Path("src/trustpcb/rq2/weighted_fusion.py").resolve()
assert canonical.risk is risk_evaluation
assert canonical.sensitivity is risk_sensitivity
assert canonical.comparison is calibrator_comparison
assert evaluation.risk is canonical.risk
assert evaluation.sensitivity is canonical.sensitivity
assert evaluation.calibration is canonical.comparison
consumers = (weighted.fusion, review.fusion, efficiency.fusion,
             rq3.frozen.fusion, rq3.review.fusion)
assert all(item is canonical for item in consumers)
for name in ("OUTPUT", "SEED", "CANDIDATE_COUNT", "TOLERANCE", "PURPOSE", "WEIGHT_KEYS",
             "HIERARCHY", "validate_weights", "grid", "component_matrix", "weighted_risk",
             "load_inputs", "evaluate_candidates", "select", "report_metrics", "run", "main",
             "risk", "comparison", "sensitivity"):
    assert getattr(legacy, name) is getattr(canonical, name)
assert (canonical.SEED, canonical.CANDIDATE_COUNT, canonical.TOLERANCE) == (24209199, 5151, 1e-12)
assert canonical.WEIGHT_KEYS == ("conf_percent", "class_percent", "localisation_percent")
assert canonical.OUTPUT == "runs/rq2/weighted_fusion"
assert weighted.WEIGHTS == (89, 1, 10)
assert weighted.SELECTED == canonical.OUTPUT + "/selected_weights.json"
for owner, observer in ((legacy, canonical), (canonical, legacy)):
    with patch.object(owner, "WEIGHT_KEYS", ("synthetic",)):
        assert all(item.WEIGHT_KEYS is observer.WEIGHT_KEYS for item in consumers)
    for name in ("component_matrix", "weighted_risk", "select"):
        with patch.object(owner, name, return_value="synthetic") as mocked:
            for item in consumers:
                assert getattr(item, name) is getattr(observer, name) is mocked
                assert getattr(item, name)("fixture") == "synthetic"
            assert mocked.call_count == len(consumers)
from trustpcb import rq2_risk_evaluation as old_risk, rq2_calibrator_comparison as old_comparison
for owner in (old_risk, risk_evaluation):
    with patch.object(owner, "metrics", return_value="synthetic") as mocked:
        assert canonical.risk.metrics("fixture") == "synthetic"
        assert evaluation.risk.metrics is mocked
for owner in (old_comparison, calibrator_comparison):
    for name in ("write_csv", "assign_folds"):
        with patch.object(owner, name, return_value="synthetic") as mocked:
            assert getattr(canonical.comparison, name)("fixture") == "synthetic"
            assert getattr(evaluation.calibration, name) is mocked
assert "torch" not in sys.modules and "ultralytics" not in sys.modules
"""])

    def test_legacy_first(self):
        self.imports("trustpcb.rq2_weighted_fusion", "trustpcb.rq2.weighted_fusion")

    def test_canonical_first(self):
        self.imports("trustpcb.rq2.weighted_fusion", "trustpcb.rq2_weighted_fusion")

    def test_legacy_help(self):
        self.assertIn("--dicc", self.python(["-m", "trustpcb.rq2_weighted_fusion", "--help"]))

    def test_canonical_help(self):
        self.assertIn("--dicc", self.python(["-m", "trustpcb.rq2.weighted_fusion", "--help"]))

    def test_legacy_main_dispatch(self):
        self.python(["-c", """
import runpy
from unittest.mock import patch
from trustpcb.rq2 import weighted_fusion
with patch.object(weighted_fusion, "main") as main:
    runpy.run_path("src/trustpcb/rq2_weighted_fusion.py", run_name="__main__")
    main.assert_called_once_with()
"""])

    def test_parser_dispatch_mock_only(self):
        with patch.object(fusion, "os", SimpleNamespace(name="posix")), \
                patch.object(fusion, "find_project_root", return_value="synthetic-root"), \
                patch.object(fusion, "run", return_value="synthetic") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            fusion.main(["--dicc"])
        run.assert_called_once_with("synthetic-root")

    def test_missing_dicc_blocks_before_access(self):
        with patch.object(fusion, "os", SimpleNamespace(name="posix")), \
                patch.object(fusion, "find_project_root") as root, patch.object(fusion, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                fusion.main([])
            root.assert_not_called()
            run.assert_not_called()


class GridSelectionTests(unittest.TestCase):
    def test_components_formula_boundaries_and_invalid_signals(self):
        keys = ("calibrated_confidence", "class_consistency", "localisation_stability")
        rows = [dict(zip(keys, values)) for values in ((0, 0, 0), (1, 1, 1), (.2, .4, .6))]
        matrix = fusion.component_matrix(rows)
        np.testing.assert_allclose(matrix, [[1, 1, 1], [0, 0, 0], [.8, .6, .4]])
        np.testing.assert_allclose(fusion.weighted_risk(matrix, (89, 1, 10)), [1, 0, .758])
        for weights, index in (((100, 0, 0), 0), ((0, 100, 0), 1), ((0, 0, 100), 2)):
            np.testing.assert_array_equal(fusion.weighted_risk(matrix, weights), matrix[:, index])
        for bad in (None, float("nan"), float("inf"), -.1, 1.1):
            with self.assertRaises(ValueError):
                fusion.component_matrix([{**rows[0], keys[0]: bad}])
        with self.assertRaises(KeyError):
            fusion.component_matrix([{}])

    def test_exact_integer_grid_vertices_and_sum(self):
        grid = fusion.grid()
        self.assertEqual(len(grid), 5151)
        self.assertEqual(len(set(grid)), 5151)
        self.assertEqual(grid, sorted(grid))
        self.assertEqual(grid[:2], [(0, 0, 100), (0, 1, 99)])
        self.assertEqual(grid[-1], (100, 0, 0))
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
    def test_loader_inventory_and_hash_alignment(self):
        images, rows, _ = fixture()
        hashes = {"synthetic": "digest"}
        with patch.object(fusion.risk, "load_inputs", return_value=(images, rows, hashes)), \
                patch.object(fusion.comparison, "load_inputs", return_value=(images, [], [], hashes)):
            loaded = fusion.load_inputs("synthetic")
        self.assertIs(loaded[0], images)
        self.assertIs(loaded[1], rows)
        for linked, binding in ((images[::-1], hashes), (images, {"synthetic": "changed"})):
            with patch.object(fusion.risk, "load_inputs", return_value=(images, rows, hashes)), \
                    patch.object(fusion.comparison, "load_inputs", return_value=(linked, [], [], binding)):
                with self.assertRaisesRegex(ValueError, "Similarity inventory"):
                    fusion.load_inputs("synthetic")

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
        for reason in ("7131 population mismatch", "foreign final-test input", "821 image count",
                       "Duplicate prediction identity", "missing score"):
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
        self.assertEqual(info["experiment"], "rq2_weighted_fusion")
        records = fusion.risk.read_csv(output / "candidate_weights.csv")
        self.assertEqual([tuple(int(r[k]) for k in fusion.WEIGHT_KEYS) for r in records], fusion.grid())
        selected = fusion.rq1._read_json(output / "selected_weights.json")
        self.assertEqual(selected, info["selected_weights"])
        reports = fusion.risk.read_csv(output / "development_metrics.csv")
        self.assertEqual([r["method"] for r in reports], [
            "calibrated_confidence_risk", "equal_weight_trustpcb_risk", "selected_weighted_trustpcb_risk"])
        self.assertEqual(list(reports[0]), ["analysis_label", "method", "auroc", "auprc",
                                          "prediction_count", "incorrect", "incorrect_prevalence"])
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
