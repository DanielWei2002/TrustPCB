"""Tiny synthetic pixels, boxes and mocked predictions only. No detector imports."""

import contextlib
import csv
import io
import math
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
from trustpcb import rq2_transformation_stability as stability
import test_rq2_final_calibrator as final_tests


def reference(identifier="synthetic#1", box=(20, 20, 40, 40), cls=0):
    return {"prediction_id": identifier, "class_id": cls, "box": list(box)}


def detection(box=(20, 20, 40, 40), cls=0, confidence=.005):
    return {"box": list(box), "class_id": cls, "confidence": confidence}


class StabilityCompatibilityTests(unittest.TestCase):
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
from trustpcb import rq2_transformation_stability as legacy
from trustpcb.rq2 import transformation_stability as canonical
from trustpcb.rq2 import final_calibrator, confidence_distribution, artifact_paths
from trustpcb.rq2 import calibrator_comparison, calibration_analysis, data_partition, detector_training
from trustpcb import rq2_final_evaluation as evaluation, rq2_weighted_final_evaluation as weighted
from trustpcb import rq2_risk_evaluation as risk, rq2_risk_sensitivity as sensitivity
from trustpcb import rq2_weighted_fusion as fusion
from trustpcb import rq3_computational_efficiency as efficiency, rq3_selective_review as review
from trustpcb import rq3_final_evaluation as rq3
assert first is second is legacy is canonical
assert sys.modules["trustpcb.rq2_transformation_stability"] is canonical
assert sys.modules["trustpcb.rq2.transformation_stability"] is canonical
assert Path(canonical.__file__).resolve() == Path("src/trustpcb/rq2/transformation_stability.py").resolve()
assert canonical.final is final_calibrator
assert canonical.raw is confidence_distribution
assert canonical.resolve_input is artifact_paths.resolve_input
assert canonical.raw.rq2_detector_training is detector_training
assert canonical.final.comparison is calibrator_comparison
assert canonical.final.comparison.analysis is calibration_analysis
assert canonical.final.comparison.partition is data_partition
consumers = (evaluation.stability, weighted.final.stability, efficiency.stability,
             rq3.frozen.final.stability)
assert all(item is canonical for item in consumers)
for name in ("OUTPUT", "FAMILIES", "SPECS", "RULES", "rectangle", "map_points",
             "forward_matrix", "transform", "evaluable", "signed_area", "polygon_area",
             "polygon_intersection", "polygon_iou", "assign", "match_transform",
             "aggregate", "load_inputs", "_predict", "run", "main", "final", "raw"):
    assert getattr(legacy, name) is getattr(canonical, name)
assert evaluation.run.__defaults__[1] is canonical._predict
# Risk workflows consume stability artifacts, not a transformation module import.
assert risk.STABILITY == canonical.OUTPUT == "runs/rq2/transformation_stability"
assert sensitivity.primary is risk
assert fusion.risk is risk
assert weighted.risk is risk
assert review.risk is risk
assert rq3.review.risk is risk
for owner, observer in ((legacy, canonical), (canonical, legacy)):
    with patch.object(owner, "SPECS", ("synthetic",)):
        assert all(item.SPECS is observer.SPECS for item in consumers)
    with patch.dict(owner.RULES, synthetic="shared"):
        assert all(item.RULES["synthetic"] == "shared" for item in consumers)
    for name in ("transform", "match_transform", "aggregate", "load_inputs", "_predict"):
        with patch.object(owner, name, return_value="synthetic") as mocked:
            for item in consumers:
                assert getattr(item, name) is getattr(observer, name) is mocked
                assert getattr(item, name)("fixture") == "synthetic"
            assert mocked.call_count == len(consumers)
# The downstream default is intentionally bound at definition time, as before.
assert evaluation.run.__defaults__[1] is canonical._predict
assert "torch" not in sys.modules and "ultralytics" not in sys.modules
"""])

    def test_legacy_first(self):
        self.imports("trustpcb.rq2_transformation_stability", "trustpcb.rq2.transformation_stability")

    def test_canonical_first(self):
        self.imports("trustpcb.rq2.transformation_stability", "trustpcb.rq2_transformation_stability")

    def test_legacy_help(self):
        self.assertIn("--dicc", self.python(["-m", "trustpcb.rq2_transformation_stability", "--help"]))

    def test_canonical_help(self):
        self.assertIn("--dicc", self.python(["-m", "trustpcb.rq2.transformation_stability", "--help"]))

    def test_legacy_main_dispatch(self):
        self.python(["-c", """
import runpy
from unittest.mock import patch
from trustpcb.rq2 import transformation_stability
with patch.object(transformation_stability, "main") as main:
    runpy.run_path("src/trustpcb/rq2_transformation_stability.py", run_name="__main__")
    main.assert_called_once_with()
"""])

    def test_parser_dispatch_mock_only(self):
        root, dataset = Path("synthetic-root"), Path("synthetic-dataset")
        with patch.object(stability, "os", SimpleNamespace(name="posix")), \
                patch.object(stability, "find_project_root", return_value=root), \
                patch.object(stability, "load_paths", return_value=SimpleNamespace(project_root=root, dataset_root=dataset)), \
                patch.object(stability, "run", return_value="synthetic") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            stability.main(["--dicc"])
        run.assert_called_once_with(root, dataset, stability._predict)

    def test_missing_dicc_blocks_before_access(self):
        with patch.object(stability, "os", SimpleNamespace(name="posix")), \
                patch.object(stability, "find_project_root") as root, patch.object(stability, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                stability.main([])
            root.assert_not_called()
            run.assert_not_called()


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(12 * 14 * 3, dtype=np.uint8).reshape(12, 14, 3)

    def test_exact_frozen_eleven_specs(self):
        self.assertEqual(len(stability.SPECS), 11)
        self.assertEqual([s["id"] for s in stability.SPECS], [
            "brightness_0.90", "brightness_1.10", "contrast_0.90", "contrast_1.10",
            "blur_0.6", "rotation_-2", "rotation_+2", "translation_x_-0.02",
            "translation_x_+0.02", "translation_y_-0.02", "translation_y_+0.02"])
        self.assertEqual([s["factor"] for s in stability.SPECS[:4]], [.9, 1.1, .9, 1.1])
        self.assertEqual(stability.SPECS[4]["sigma"], .6)
        self.assertEqual([s["degrees"] for s in stability.SPECS[5:7]], [-2, 2])
        self.assertEqual([(s["axis"], s["fraction"]) for s in stability.SPECS[7:]], [("x", -.02), ("x", .02), ("y", -.02), ("y", .02)])

    def test_brightness_both_factors_clipping(self):
        for spec in stability.SPECS[:2]:
            expected = np.rint(np.clip(self.image.astype(float) * spec["factor"], 0, 255)).astype(np.uint8)
            np.testing.assert_array_equal(stability.transform(self.image, spec)[0], expected)

    def test_contrast_both_factors_original_scalar_mean(self):
        for spec in stability.SPECS[2:4]:
            values = self.image.astype(float)
            expected = np.rint(np.clip(values.mean() + spec["factor"] * (values - values.mean()), 0, 255)).astype(np.uint8)
            np.testing.assert_array_equal(stability.transform(self.image, spec)[0], expected)

    def test_gaussian_sigma_and_channel_isolation(self):
        pixels = np.zeros((11, 11, 3), dtype=np.uint8)
        pixels[5, 5, 0] = 255
        output = stability.transform(pixels, stability.SPECS[4])[0]
        axis = np.arange(-2, 3)
        kernel = np.exp(-axis ** 2 / (2 * .6 ** 2))
        kernel /= kernel.sum()
        expected = np.rint(255 * np.outer(kernel, kernel)).astype(np.uint8)
        np.testing.assert_array_equal(output[3:8, 3:8, 0], expected)
        self.assertFalse(output[:, :, 1:].any())

    def test_all_transforms_deterministic_same_dimensions(self):
        for spec in stability.SPECS:
            first = stability.transform(self.image, spec)
            second = stability.transform(self.image, spec)
            np.testing.assert_array_equal(first[0], second[0])
            self.assertEqual(first[0].shape, self.image.shape)
            self.assertEqual(first[0].dtype, np.uint8)

    def test_affine_matrices_inverse_and_centre(self):
        corners = stability.rectangle([2, 3, 7, 8])
        for spec in stability.SPECS[5:]:
            matrix = stability.forward_matrix(spec, 100, 50)
            np.testing.assert_allclose(stability.map_points(stability.map_points(corners, matrix), np.linalg.inv(matrix)), corners, atol=1e-13)
            if spec["family"] == "rotation":
                np.testing.assert_allclose(stability.map_points([[50, 25]], matrix), [[50, 25]])
            else:
                axis = 0 if spec["axis"] == "x" else 1
                self.assertEqual(matrix[axis, 2], spec["fraction"] * (100 if axis == 0 else 50))

    def test_translation_pixels_and_channel_median_border(self):
        pixels = np.full((100, 100, 3), [10, 20, 30], dtype=np.uint8)
        pixels[40, 40] = [200, 150, 100]
        for spec in stability.SPECS[7:]:
            output, matrix, median = stability.transform(pixels, spec)
            self.assertEqual(median, [10, 20, 30])
            x, y = stability.map_points([[40.5, 40.5]], matrix)[0] - .5
            np.testing.assert_array_equal(output[int(y), int(x)], [200, 150, 100])
            np.testing.assert_array_equal(output[0, 0], [10, 20, 30])

    def test_evaluability_bounds_rotation_translation_photometric(self):
        for spec in stability.SPECS:
            matrix = stability.forward_matrix(spec, 100, 100)
            geometric = spec["family"] in ("rotation", "translation")
            self.assertTrue(stability.evaluable([20, 20, 40, 40], matrix, 100, 100, geometric))
            self.assertEqual(stability.evaluable([0, 0, 100, 100], matrix, 100, 100, geometric), not geometric)
        self.assertTrue(stability.evaluable([0, 0, 100, 100], np.eye(3), 100, 100, True))

    def test_rotation_pixel_centres_agree_with_forward_geometry(self):
        pixels = np.zeros((101, 101, 3), dtype=np.uint8)
        pixels[50, 80] = 255
        yy, xx = np.indices(pixels.shape[:2])
        for spec in stability.SPECS[5:7]:
            output, matrix, _ = stability.transform(pixels, spec)
            mass = output[:, :, 0].astype(float)
            actual = [np.sum((xx + .5) * mass) / mass.sum(), np.sum((yy + .5) * mass) / mass.sum()]
            expected = stability.map_points([[80.5, 50.5]], matrix)[0]
            np.testing.assert_allclose(actual, expected, atol=.025)


class PolygonTests(unittest.TestCase):
    def test_area_intersection_and_winding(self):
        first, second = stability.rectangle([0, 0, 2, 2]), stability.rectangle([1, 1, 3, 3])
        self.assertEqual(stability.polygon_area(first), 4)
        self.assertEqual(stability.polygon_area(stability.polygon_intersection(first, second)), 1)
        self.assertAlmostEqual(stability.polygon_iou(first, second), 1 / 7)
        self.assertAlmostEqual(stability.polygon_iou(first[::-1], second[::-1]), 1 / 7)

    def test_identity_disjoint_touching_degenerate(self):
        square = stability.rectangle([0, 0, 2, 2])
        self.assertEqual(stability.polygon_iou(square, square), 1)
        for box in ([3, 3, 5, 5], [2, 0, 4, 2], [1, 1, 1, 1]):
            self.assertEqual(stability.polygon_iou(square, stability.rectangle(box)), 0)

    def test_rotated_quadrilateral_analytic_iou_not_enclosing_box(self):
        square = stability.rectangle([-1, -1, 1, 1])
        angle = math.pi / 4
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        polygon = stability.map_points(square, rotation)
        self.assertAlmostEqual(stability.polygon_iou(square, polygon), 1 / math.sqrt(2))
        enclosing = stability.rectangle([*polygon.min(axis=0), *polygon.max(axis=0)])
        self.assertNotAlmostEqual(stability.polygon_iou(square, polygon), stability.polygon_iou(square, enclosing))

    def test_containment_symmetry_and_translation_invariance(self):
        square = stability.rectangle([0, 0, 10, 10])
        self.assertEqual(stability.polygon_iou(square, stability.rectangle([2, 2, 4, 4])), .04)
        for degrees in (-35, -2, 2, 35):
            polygon = stability.map_points(stability.rectangle([1, 1, 12, 8]),
                stability.forward_matrix({"family": "rotation", "degrees": degrees}, 20, 20))
            overlap = stability.polygon_iou(square, polygon)
            self.assertAlmostEqual(overlap, stability.polygon_iou(polygon, square))
            self.assertAlmostEqual(overlap, stability.polygon_iou(square + [100, -80], polygon + [100, -80]))


class MatchingTests(unittest.TestCase):
    def test_empty_references_and_invalid_iou(self):
        mapped, audits = stability.match_transform([], [detection()], stability.SPECS[0], 100, 100)
        self.assertEqual(len(mapped), 1)
        self.assertEqual(audits, [])
        self.assertEqual(stability.assign(np.empty((0, 2))), {})
        for values in ([[float("nan")]], [[-.01]], [[1.01]], [1.]):
            with self.assertRaises(ValueError):
                stability.assign(values)

    def test_no_evaluable_family_has_missing_scores(self):
        audits = [{"transform_id": spec["id"], "family": spec["family"], "evaluable": False,
                   "status": "not_evaluable", "class_consistency_value": None,
                   "localisation_stability_value": None} for spec in stability.SPECS]
        scores = stability.aggregate(audits)
        self.assertIsNone(scores["class_consistency"])
        self.assertIsNone(scores["localisation_stability"])
        self.assertEqual(scores["n_evaluable_families"], 0)
        self.assertEqual(scores["n_matched_transforms"], 0)

    def test_threshold_cardinality_before_iou(self):
        self.assertEqual(stability.assign([[1., .5], [.5, .49]]), {0: 1, 1: 0})
        result = stability.assign([[1., .5, 0], [0, 1., .5], [.5, 0, 0]])
        self.assertEqual(len(result), 3)
        self.assertEqual(stability.assign([[.5, .499999]]), {0: 0})

    def test_secondary_total_iou_one_to_one_and_ties(self):
        self.assertEqual(stability.assign([[.7, .8], [.9, .6]]), {0: 1, 1: 0})
        matrix = np.ones((3, 2))
        first = stability.assign(matrix)
        self.assertEqual(len(set(first.values())), 2)
        for _ in range(5):
            self.assertEqual(stability.assign(matrix), first)
        self.assertEqual(stability.assign(np.empty((2, 0))), {})

    def test_class_agnostic_and_below_reference_floor_eligible(self):
        refs = [reference()]
        for cls in (0, 1):
            _, audits = stability.match_transform(refs, [detection(cls=cls, confidence=.001)], stability.SPECS[0], 100, 100)
            self.assertEqual(audits[0]["status"], "matched")
            self.assertEqual(audits[0]["class_consistency_value"], int(cls == 0))
            self.assertEqual(audits[0]["localisation_stability_value"], 1)

    def test_unmatched_zero_and_unevaluable_missing(self):
        _, audits = stability.match_transform([reference(box=[0, 0, 100, 100]), reference("inside")], [], stability.SPECS[5], 100, 100)
        self.assertEqual(audits[0]["status"], "not_evaluable")
        self.assertIsNone(audits[0]["class_consistency_value"])
        self.assertIsNone(audits[0]["localisation_stability_value"])
        self.assertEqual(audits[1]["class_consistency_value"], 0)
        self.assertEqual(audits[1]["localisation_stability_value"], 0)

    def test_class_and_confidence_cannot_change_tied_spatial_assignment(self):
        refs = [reference("first"), reference("second")]
        first, audit1 = stability.match_transform(refs, [detection(cls=1), detection(cls=0)], stability.SPECS[0], 100, 100)
        second, audit2 = stability.match_transform(refs, [detection(cls=0, confidence=.9), detection(cls=1)], stability.SPECS[0], 100, 100)
        self.assertEqual([r["detection_id"] for r in audit1], [r["detection_id"] for r in audit2])
        self.assertEqual([r["class_id"] for r in first], [1, 0])
        self.assertEqual([r["class_id"] for r in second], [0, 1])

    def test_exact_inverse_polygon_preserved(self):
        spec = stability.SPECS[6]
        mapped, _ = stability.match_transform([reference()], [detection()], spec, 100, 100)
        expected = stability.map_points(stability.rectangle([20, 20, 40, 40]), np.linalg.inv(stability.forward_matrix(spec, 100, 100)))
        np.testing.assert_allclose(mapped[0]["polygon"], expected)
        self.assertNotEqual(mapped[0]["polygon"][0][1], mapped[0]["polygon"][1][1])

    def test_family_balance_translation_not_fourfold_weight(self):
        audits = []
        for spec in stability.SPECS:
            value = int(spec["family"] == "translation")
            audits.append({"transform_id": spec["id"], "family": spec["family"], "evaluable": True,
                           "status": "matched" if value else "unmatched", "class_consistency_value": value,
                           "localisation_stability_value": float(value)})
        scores = stability.aggregate(audits)
        self.assertEqual(scores["class_consistency"], .2)
        self.assertEqual(scores["localisation_stability"], .2)
        self.assertEqual(scores["n_evaluable_transforms"], 11)
        self.assertEqual(scores["n_matched_transforms"], 4)
        self.assertEqual(scores["n_evaluable_families"], 5)
        for row in audits:
            if row["family"] == "rotation":
                row.update(evaluable=False, status="not_evaluable", class_consistency_value=None, localisation_stability_value=None)
        scores = stability.aggregate(audits)
        self.assertEqual(scores["class_consistency"], .25)
        self.assertIsNone(scores["class_consistency_rotation"])
        self.assertEqual(scores["n_evaluable_transforms"], 9)
        self.assertEqual(scores["n_evaluable_families"], 4)


class InputTests(unittest.TestCase):
    def setUp(self):
        # Reuse an existing tiny synthetic provenance fixture, never real outputs.
        self.fixture = final_tests.FinalCalibratorTests("test_unreviewed_selection_rejected_before_reading")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        table = self.fixture.analysis / "labelled_predictions.csv"
        with table.open() as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            row.update(class_name=stability.raw.rq2_detector_training.NAMES[int(row["class_id"])], x1="20", y1="20", x2="40", y2="40")
        table.unlink()
        stability.final.comparison.write_csv(table, rows)
        self.fixture.bind_analysis()
        self.fixture.bind_comparison()
        self.fixture.execute()
        checkpoint = self.root / stability.raw.CHECKPOINT
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic selected checkpoint")
        mocked = patch.object(stability.raw, "CHECKPOINT_SHA", stability.rq1._sha(checkpoint))
        mocked.start()
        self.addCleanup(mocked.stop)
        # Source identity was frozen before checkpoint fixture; mock only identity
        # verification's expected hash while leaving source validation unchanged.
        self.checkpoint = checkpoint

    def load(self):
        # Calibration source provenance references this same synthetic checkpoint.
        # Update the binding before final fitting is unnecessary: validation uses
        # original frozen identifier in source metadata, so isolate that read only.
        with patch.object(stability.raw, "checkpoint_identity", return_value={"sha256": stability.rq1._sha(self.checkpoint), "epoch": 72}):
            original_sha = stability.final.comparison.rq1._read_json(self.fixture.analysis / "provenance.json")["checkpoint_sha256"]
            with patch.object(stability.raw, "CHECKPOINT_SHA", original_sha):
                return stability.load_inputs(self.root)

    def test_valid_final_provenance_reference_order_and_identity(self):
        images, refs, hashes, _ = self.load()
        self.assertEqual(len(images), 821)
        self.assertEqual(len(refs), 40)
        self.assertEqual([int(r["record"]["source_row"]) for r in refs], list(range(1, 41)))
        self.assertIn(f"{stability.final.OUTPUT}/final_calibrator.json", hashes)

    def test_changed_calibrator_or_table_rejected(self):
        artifact = self.root / stability.final.OUTPUT / "final_calibrator.json"
        artifact.write_text(artifact.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load()

    def test_checkpoint_real_identity_checker_rejects_change(self):
        stability.raw.checkpoint_identity(self.root)
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Checkpoint differs"):
            stability.raw.checkpoint_identity(self.root)

    def test_foreign_reference_rejected_even_with_rebound_table_hash(self):
        folder = self.root / stability.final.OUTPUT
        path = folder / "calibrated_development_predictions.csv"
        path.write_text(path.read_text().replace("images/train/synthetic_0.jpg", "images/val/final-test.jpg"))
        provenance = stability.rq1._read_json(folder / "provenance.json")
        provenance["output_sha256"][path.name] = stability.rq1._sha(path)
        stability.rq1._write_json(folder / "provenance.json", provenance)
        with self.assertRaisesRegex(ValueError, "identity/order"):
            self.load()


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = ["images/train/synthetic.jpg"]
        self.refs = [{**reference("synthetic#2"), "image": self.images[0], "record": {"prediction_id": "synthetic#2", "image": self.images[0]}},
                     {**reference("synthetic#1", (0, 0, 100, 100)), "image": self.images[0], "record": {"prediction_id": "synthetic#1", "image": self.images[0]}}]
        for mocked in (patch.object(stability, "load_inputs", return_value=(self.images, self.refs, {}, {"sha256": "synthetic"})),
                       patch.object(stability.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"}),
                       patch.object(stability.importlib.metadata, "version", side_effect=lambda name: stability.raw.VERSION if name == "ultralytics" else "synthetic")):
            mocked.start()
            self.addCleanup(mocked.stop)

    def predictor(self, root, dataset, images, metadata):
        for index, spec in enumerate(stability.SPECS):
            detections = [detection()] * (300 if index == 0 else 1)
            yield images[0], 100, 100, spec["id"], detections, {}

    def test_complete_outputs_saturation_order_audit_and_no_overwrite(self):
        output = stability.run(self.root, self.root, self.predictor)
        with (output / "prediction_stability_scores.csv").open() as file:
            scores = list(csv.DictReader(file))
        self.assertEqual([s["prediction_id"] for s in scores], ["synthetic#2", "synthetic#1"])
        with (output / "prediction_transform_matches.csv").open() as file:
            audits = list(csv.DictReader(file))
        self.assertEqual(list(audits[0]), [
            "image", "prediction_id", "transform_id", "family", "evaluable", "status",
            "detection_id", "mapped_iou", "class_consistency_value",
            "localisation_stability_value", "max_det_saturation"])
        with (output / "transformed_predictions.csv").open() as file:
            predictions = list(csv.DictReader(file))
        self.assertEqual(list(predictions[0]), [
            "image", "transform_id", "detection_id", "class_id", "confidence",
            "x1", "y1", "x2", "y2", "inverse_mapped_polygon", "max_det_saturation"])
        self.assertEqual([row["transform_id"] for row in audits],
                         [spec["id"] for spec in stability.SPECS for _ in self.refs])
        self.assertEqual(len(audits), 22)
        self.assertTrue(any(r["status"] == "not_evaluable" and r["localisation_stability_value"] == "" for r in audits))
        summary = stability.rq1._read_json(output / "summary.json")
        self.assertEqual(summary["max_det_saturated_views"], 1)
        provenance = stability.rq1._read_json(output / "provenance.json")
        self.assertEqual(provenance["status"], "complete")
        self.assertEqual(provenance["experiment"], "rq2_transformation_stability")
        self.assertEqual(provenance["rules"], stability.RULES)
        for name, digest in provenance["output_sha256"].items():
            self.assertEqual(stability.rq1._sha(output / name), digest)
        with self.assertRaises(FileExistsError):
            stability.run(self.root, self.root, self.predictor)

    def test_incomplete_and_foreign_result_block(self):
        def foreign(*args):
            yield "images/val/final-test.jpg", 100, 100, stability.SPECS[0]["id"], [], {}
        with self.assertRaisesRegex(RuntimeError, "Foreign"):
            stability.run(self.root, self.root, foreign)
        provenance = stability.rq1._read_json(self.root / stability.OUTPUT / "provenance.json")
        self.assertEqual(provenance["status"], "incomplete")
        with self.assertRaises(FileExistsError):
            stability.run(self.root, self.root, self.predictor)

    def test_missing_views_block(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            stability.run(self.root, self.root, lambda *args: iter(()))

    def test_windows_gate_prevents_real_adapter(self):
        with patch.object(stability.os, "name", "nt"), patch.object(stability, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                stability.main(["--dicc"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
