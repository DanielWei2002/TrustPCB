"""Synthetic secondary-target fixtures only; primary outputs remain immutable."""

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
from trustpcb import rq2_risk_sensitivity as sensitivity
import test_rq2_risk_evaluation as primary_tests

p = sensitivity.primary


def population():
    images, rows = primary_tests.population()
    for row, secondary in zip(rows, [0, 0, 0, 1]):
        row["correct_at_iou75"] = str(secondary)
    return images, rows


class SensitivityCompatibilityTests(unittest.TestCase):
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
from trustpcb import rq2_risk_sensitivity as legacy, rq2_risk_evaluation as old_primary
from trustpcb.rq2 import risk_sensitivity as canonical, risk_evaluation as primary
from trustpcb import rq2_weighted_fusion as fusion, rq2_final_evaluation as evaluation
from trustpcb import rq2_weighted_final_evaluation as weighted
from trustpcb import rq3_selective_review as review, rq3_computational_efficiency as efficiency
from trustpcb import rq3_final_evaluation as rq3
assert first is second is legacy is canonical
assert sys.modules["trustpcb.rq2_risk_sensitivity"] is canonical
assert sys.modules["trustpcb.rq2.risk_sensitivity"] is canonical
assert Path(canonical.__file__).resolve() == Path("src/trustpcb/rq2/risk_sensitivity.py").resolve()
assert canonical.primary is primary is old_primary
consumers = (fusion.sensitivity, evaluation.sensitivity, weighted.sensitivity,
             review.fusion.sensitivity, efficiency.fusion.sensitivity,
             rq3.frozen.sensitivity, rq3.review.fusion.sensitivity)
assert all(item is canonical for item in consumers)
for name in ("OUTPUT", "LABEL", "TARGET", "CORRECTNESS", "SEED", "REPLICATES",
             "DEFINITIONS", "sensitivity_rows", "load_inputs", "run", "main", "primary"):
    assert getattr(legacy, name) is getattr(canonical, name)
for name in ("sensitivity_rows", "load_inputs", "run", "main"):
    assert getattr(canonical, name).__globals__ is canonical.__dict__
assert (canonical.SEED, canonical.REPLICATES) == (24209199, 10000)
assert canonical.OUTPUT == "runs/rq2/risk_sensitivity_iou75"
assert canonical.LABEL == "Secondary IoU>=0.75 sensitivity analysis"
assert canonical.TARGET == "incorrect_iou75 = 1 - frozen IoU75 correctness"
assert canonical.CORRECTNESS == "same-class one-to-one IoU >= 0.75; existing secondary labels only"
assert canonical.DEFINITIONS == dict(primary.DEFINITIONS, target=canonical.TARGET)
for owner, observer in ((legacy, canonical), (canonical, legacy)):
    with patch.object(owner, "LABEL", "synthetic"):
        assert all(item.LABEL == observer.LABEL == "synthetic" for item in consumers)
    with patch.dict(owner.DEFINITIONS, synthetic="shared"):
        assert all(item.DEFINITIONS["synthetic"] == "shared" for item in consumers)
    with patch.object(owner, "sensitivity_rows", return_value="synthetic") as mocked:
        for item in consumers:
            assert item.sensitivity_rows is observer.sensitivity_rows is mocked
            assert item.sensitivity_rows("fixture") == "synthetic"
        assert mocked.call_count == len(consumers)
for owner in (old_primary, primary):
    for name in ("load_inputs", "risk_scores", "bootstrap", "evaluation_tables", "nullable"):
        with patch.object(owner, name, return_value="synthetic") as mocked:
            for item in consumers:
                assert getattr(item.primary, name) is mocked
                assert getattr(item.primary, name)("fixture") == "synthetic"
            assert mocked.call_count == len(consumers)
assert "torch" not in sys.modules and "ultralytics" not in sys.modules
"""])

    def test_legacy_first(self):
        self.imports("trustpcb.rq2_risk_sensitivity", "trustpcb.rq2.risk_sensitivity")

    def test_canonical_first(self):
        self.imports("trustpcb.rq2.risk_sensitivity", "trustpcb.rq2_risk_sensitivity")

    def test_legacy_help(self):
        self.assertIn("--dicc", self.python(["-m", "trustpcb.rq2_risk_sensitivity", "--help"]))

    def test_canonical_help(self):
        self.assertIn("--dicc", self.python(["-m", "trustpcb.rq2.risk_sensitivity", "--help"]))

    def test_legacy_main_dispatch(self):
        self.python(["-c", """
import runpy
from unittest.mock import patch
from trustpcb.rq2 import risk_sensitivity
with patch.object(risk_sensitivity, "main") as main:
    runpy.run_path("src/trustpcb/rq2_risk_sensitivity.py", run_name="__main__")
    main.assert_called_once_with()
"""])

    def test_parser_dispatch_mock_only(self):
        with patch.object(sensitivity, "os", SimpleNamespace(name="posix")), \
                patch.object(sensitivity, "find_project_root", return_value="synthetic-root"), \
                patch.object(sensitivity, "run", return_value="synthetic") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            sensitivity.main(["--dicc"])
        run.assert_called_once_with("synthetic-root")

    def test_missing_dicc_blocks_before_access(self):
        with patch.object(sensitivity, "os", SimpleNamespace(name="posix")), \
                patch.object(sensitivity, "find_project_root") as root, patch.object(sensitivity, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                sensitivity.main([])
            root.assert_not_called()
            run.assert_not_called()


class TargetMetricTests(unittest.TestCase):
    def test_empty_agreeing_aliases_and_exact_row_preservation(self):
        self.assertEqual(sensitivity.sensitivity_rows([]), [])
        _, rows = population()
        for row in rows:
            row["correct_iou75"] = int(row["correct_at_iou75"])
        before = copy.deepcopy(rows)
        adapted = sensitivity.sensitivity_rows(rows)
        self.assertEqual(adapted, sensitivity.sensitivity_rows(rows))
        for original, result in zip(rows, adapted):
            self.assertIsNot(original, result)
            self.assertEqual({k: result[k] for k in original if k != "incorrect_prediction"},
                             {k: v for k, v in original.items() if k != "incorrect_prediction"})
        self.assertEqual(rows, before)

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
        duplicates = copy.deepcopy(rows)
        duplicates[1]["prediction_id"] = duplicates[0]["prediction_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            p.validate_rows(images, duplicates, expected_count=4)
        with self.assertRaises(ValueError):
            p.validate_rows(images, rows[:-1], expected_count=4)
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
        self.assertEqual(info["experiment"], "rq2_risk_sensitivity_iou75")
        for name, digest in info["output_sha256"].items():
            self.assertEqual(sensitivity.rq1._sha(output / name), digest)
        summary = sensitivity.rq1._read_json(output / "summary.json")
        self.assertEqual(summary["incorrect"], 3)
        self.assertEqual(summary["incorrect_prevalence"], .75)
        self.assertEqual(len(p.read_csv(output / "method_metrics.csv")), 8)
        self.assertEqual(len(p.read_csv(output / "bootstrap_metrics.csv")), 32 * 8)
        methods = p.read_csv(output / "method_metrics.csv")
        self.assertEqual([r["method"] for r in methods], list(p.METHODS))
        audit = p.read_csv(output / "bootstrap_metrics.csv")
        self.assertEqual(list(audit[0]), ["analysis_label", "replicate", "method", "auroc",
                                        "auprc", "auroc_valid", "auprc_valid"])
        self.assertEqual([(r["replicate"], r["method"]) for r in audit],
                         [(str(i), method) for i in range(32) for method in p.METHODS])
        self.assertTrue(all(r["analysis_label"] == sensitivity.LABEL for r in audit))
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
