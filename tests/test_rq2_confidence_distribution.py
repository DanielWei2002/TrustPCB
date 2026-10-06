"""Confidence distribution tests: text and tiny synthetic predictions only, no ML imports."""

import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trustpcb import rq2_confidence_distribution as confidence


def row(image, score):
    return dict(zip(confidence.FIELDS, (image, 0, "SH", score, 1., 2., 3., 4.)))


class ConfidenceCompatibilityTests(unittest.TestCase):
    def run_python(self, arguments):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(REPO / "src"), environment.get("PYTHONPATH", "")])
        result = subprocess.run([sys.executable, "-B", *arguments], cwd=REPO,
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def check_import_order(self, first, second):
        self.run_python(["-c", f'''
import importlib
from pathlib import Path
import sys
from unittest.mock import patch
from types import SimpleNamespace
first = importlib.import_module({first!r})
second = importlib.import_module({second!r})
from trustpcb import rq2_confidence_distribution as legacy
from trustpcb.rq2 import confidence_distribution as canonical
from trustpcb.rq2 import detector_training, artifact_paths
from trustpcb import rq2_calibration_analysis as matching
from trustpcb import rq2_transformation_stability as stability
from trustpcb import rq2_final_evaluation as final
from trustpcb import rq3_computational_efficiency as efficiency
assert first is second is legacy is canonical
assert sys.modules["trustpcb.rq2_confidence_distribution"] is canonical
assert sys.modules["trustpcb.rq2.confidence_distribution"] is canonical
assert Path(canonical.__file__).resolve() == Path("src/trustpcb/rq2/confidence_distribution.py").resolve()
assert canonical.rq2_detector_training is detector_training
assert canonical.resolve_input is artifact_paths.resolve_input
assert canonical.reject_historical_output is artifact_paths.reject_historical_output
consumers = (matching.raw, stability.raw, final.raw, efficiency.raw)
assert all(consumer is canonical for consumer in consumers)
for name in ("CHECKPOINT", "CHECKPOINT_SHA", "DEVELOPMENT", "EPOCH", "FIELDS", "IMAGE_COUNT",
             "MANIFEST_SHA", "OUTPUT", "SETTINGS", "VERSION", "_predict", "checkpoint_identity",
             "development_images", "validate_row", "EDGES", "extract", "summarize",
             "find_project_root", "main"):
    assert getattr(legacy, name) is getattr(canonical, name)
for name in ("plan", "extract", "_predict", "checkpoint_identity", "development_images", "main"):
    assert getattr(canonical, name).__globals__ is canonical.__dict__
for owner, observer in ((legacy, canonical), (canonical, legacy)):
    with patch.object(owner, "SETTINGS", {{"conf": 0.001}}):
        owner.SETTINGS["synthetic"] = True
        assert observer.SETTINGS["synthetic"] is True
        assert all(consumer.SETTINGS is observer.SETTINGS for consumer in consumers)
    with patch.object(owner, "checkpoint_identity", return_value="synthetic") as identity:
        assert stability.raw.checkpoint_identity("fixture") == "synthetic"
        identity.assert_called_once_with("fixture")
        assert observer.checkpoint_identity is identity
    with patch.object(owner, "_predict", return_value=iter([("synthetic", [])])) as predict:
        with patch.object(final, "os", SimpleNamespace(name="posix")):
            assert list(final._original("root", "dataset", [], {{}})) == [("synthetic", [])]
        assert observer._predict is predict
        predict.assert_called_once()
assert "torch" not in sys.modules
assert "ultralytics" not in sys.modules
'''])

    def test_legacy_first_canonical_second(self):
        self.check_import_order("trustpcb.rq2_confidence_distribution", "trustpcb.rq2.confidence_distribution")

    def test_canonical_first_legacy_second(self):
        self.check_import_order("trustpcb.rq2.confidence_distribution", "trustpcb.rq2_confidence_distribution")

    def check_help(self, module):
        output = self.run_python(["-m", module, "--help"])
        self.assertIn("{plan,extract}", output)
        self.assertIn("--dicc", output)

    def test_legacy_cli_help(self):
        self.check_help("trustpcb.rq2_confidence_distribution")

    def test_canonical_cli_help(self):
        self.check_help("trustpcb.rq2.confidence_distribution")

    def test_legacy_cli_dispatches_canonical_main(self):
        self.run_python(["-c", '''
import runpy
import sys
from unittest.mock import patch
from trustpcb.rq2 import confidence_distribution as canonical
original_main = sys.modules["__main__"]
with patch.object(canonical, "main") as main:
    runpy.run_path("src/trustpcb/rq2_confidence_distribution.py", run_name="__main__")
    main.assert_called_once_with()
assert sys.modules["__main__"] is original_main
assert "torch" not in sys.modules
assert "ultralytics" not in sys.modules
'''])


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dataset = self.root / "empty dataset"
        self.dataset.mkdir()
        self.images = ["images/train/a.jpg", "images/val/b.jpg", "images/train/c.jpg"]
        manifest = self.root / confidence.DEVELOPMENT
        manifest.parent.mkdir(parents=True)
        raw = ("\n".join(self.images) + "\n").encode()
        manifest.write_bytes(raw)
        checkpoint = self.root / confidence.CHECKPOINT
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint")
        for target, value in (("IMAGE_COUNT", 3), ("MANIFEST_SHA", hashlib.sha256(raw).hexdigest()),
                              ("CHECKPOINT_SHA", hashlib.sha256(checkpoint.read_bytes()).hexdigest())):
            mock = patch.object(confidence, target, value)
            mock.start()
            self.addCleanup(mock.stop)
        for mock in (
            patch.object(confidence.rq1, "git_provenance", return_value={"git_commit": "a" * 40, "git_dirty": False}),
            patch.object(confidence.rq1, "_environment", return_value={"synthetic": True}),
            patch.object(confidence.importlib.metadata, "version", return_value=confidence.VERSION),
        ):
            mock.start()
            self.addCleanup(mock.stop)

    def test_bin_boundaries(self):
        scores = list(confidence.EDGES)
        summary = confidence.summarize(self.images, [row(self.images[0], c) for c in scores])
        self.assertEqual([b["count"] for b in summary["confidence_intervals"]], [1, 1, 1, 1, 1, 1, 2])
        self.assertAlmostEqual(sum(b["percentage"] for b in summary["confidence_intervals"]), 100)
        for edge in confidence.EDGES[1:-1]:
            index = confidence.EDGES.index(edge)
            summary = confidence.summarize(self.images, [row(self.images[0], math.nextafter(edge, 0))])
            self.assertEqual(summary["confidence_intervals"][index - 1]["count"], 1)

    def test_cumulative_threshold_counts(self):
        summary = confidence.summarize(self.images, [row(self.images[0], c) for c in confidence.EDGES])
        self.assertEqual([r["count"] for r in summary["cumulative_retained"]], [8, 7, 6, 5, 4])

    def test_deterministic_summary_includes_zero_images(self):
        rows = [row(self.images[0], 0.1), row(self.images[0], 0.5)]
        summary = confidence.summarize(self.images, rows)
        self.assertEqual(summary, confidence.summarize(self.images, list(reversed(rows))))
        self.assertEqual(summary["predictions_per_image"], {"min": 0, "median": 0, "mean": 2 / 3, "max": 2})
        self.assertEqual(summary["images_with_predictions"], 1)
        self.assertEqual(summary["per_class"][0]["count"], 2)
        self.assertIsNone(summary["final_calibration_threshold"])

    def test_empty_and_invalid_predictions(self):
        empty = confidence.summarize(self.images, [])
        self.assertEqual(empty["total_predictions"], 0)
        self.assertIsNone(empty["confidence"]["min"])
        self.assertEqual(empty["confidence_intervals"][0]["percentage"], 0)
        for score in (0, 0.0009, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                confidence.summarize(self.images, [row(self.images[0], score)])
        with self.assertRaises(ValueError):
            confidence.summarize(self.images, [row("images/val/test-only.jpg", 0.2)])

    def test_frozen_real_development_text_and_partition_substitution(self):
        # Restore constants for this read-only check of committed manifest TEXT.
        from trustpcb.rq2_detector_training import MANIFESTS
        with patch.object(confidence, "IMAGE_COUNT", 821), patch.object(confidence, "MANIFEST_SHA", MANIFESTS["val"][2]):
            self.assertEqual(len(confidence.development_images(REPO)), 821)
        manifest = self.root / confidence.DEVELOPMENT
        manifest.write_text("images/val/test-only.jpg\n")
        with self.assertRaises(ValueError):
            confidence.development_images(self.root)

    def test_checkpoint_hash_and_missing_checkpoint(self):
        identity = confidence.checkpoint_identity(self.root)
        self.assertEqual(identity["epoch"], 72)
        path = self.root / confidence.CHECKPOINT
        path.write_bytes(b"different checkpoint")
        with self.assertRaises(ValueError):
            confidence.checkpoint_identity(self.root)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            confidence.checkpoint_identity(self.root)

    @staticmethod
    def fake_plot(rows, target):
        target.write_bytes(b"synthetic plot adapter")

    def fake_predict(self, root, dataset, images, metadata):
        self.assertEqual(images, self.images)
        for index, image in enumerate(images):
            yield image, [row(image, 0.1)] if index == 0 else []

    def test_extraction_outputs_and_no_overwrite(self):
        out = confidence.extract(self.root, self.dataset, self.fake_predict, self.fake_plot)
        summary = json.loads((out / "summary.json").read_text())
        provenance = json.loads((out / "provenance.json").read_text())
        self.assertEqual(summary["total_predictions"], 1)
        self.assertEqual(provenance["status"], "complete")
        self.assertEqual(provenance["prediction_settings"]["conf"], 0.001)
        self.assertEqual(set(provenance), set(confidence.plan(self.root)) | {
            "checkpoint_identity", "environment", "matplotlib_version", "status", "started_utc",
            "coordinates", "confidence", "completed_utc", "output_sha256"})
        self.assertEqual(provenance["experiment"], "rq2_confidence_distribution")
        self.assertEqual(provenance["checkpoint_sha256"], confidence.CHECKPOINT_SHA)
        self.assertEqual(provenance["manifest_sha256_lf"], confidence.MANIFEST_SHA)
        with (out / "predictions.csv").open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            self.assertEqual(reader.fieldnames, ["image", "class_id", "class_name", "confidence",
                                               "x1", "y1", "x2", "y2"])
            self.assertEqual(len(list(reader)), 1)
        self.assertEqual({p.name for p in out.iterdir()}, {
            "predictions.csv", "images.csv", "summary.json", "confidence_histogram.png", "provenance.json"})
        self.assertNotIn(str(self.root), (out / "predictions.csv").read_text())
        self.assertEqual(len((out / "images.csv").read_text().splitlines()), 4)
        with self.assertRaises(FileExistsError):
            confidence.extract(self.root, self.dataset, self.fake_predict, self.fake_plot)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("ultralytics", sys.modules)

    def test_missing_result_is_incomplete_and_blocks_retry(self):
        def partial(*_):
            yield self.images[0], []
        with self.assertRaises(RuntimeError):
            confidence.extract(self.root, self.dataset, partial, self.fake_plot)
        provenance = json.loads((self.root / confidence.OUTPUT / "provenance.json").read_text())
        self.assertEqual(provenance["status"], "incomplete")
        with self.assertRaises(FileExistsError):
            confidence.extract(self.root, self.dataset, self.fake_predict, self.fake_plot)

    def test_foreign_prediction_result_is_rejected(self):
        def foreign(*_):
            yield "images/val/test-only.jpg", []
        with self.assertRaises(RuntimeError):
            confidence.extract(self.root, self.dataset, foreign, self.fake_plot)

    def test_windows_gate_precedes_model_access(self):
        with patch.object(confidence, "find_project_root", return_value=self.root), patch.object(os, "name", "nt"):
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                confidence.main(["extract", "--dicc"])

    def test_plan_parser_preserves_schema_without_extraction(self):
        output = io.StringIO()
        with patch.object(confidence, "find_project_root", return_value=self.root), \
                patch.object(confidence, "extract") as extract, \
                patch.object(confidence, "checkpoint_identity") as checkpoint, \
                contextlib.redirect_stdout(output):
            confidence.main(["plan"])
        self.assertEqual(json.loads(output.getvalue()), confidence.plan(self.root))
        extract.assert_not_called()
        checkpoint.assert_not_called()
        self.assertEqual(set(confidence.plan(self.root)), {
            "experiment", "development_manifest", "manifest_sha256_lf", "development_images",
            "checkpoint", "checkpoint_epoch", "checkpoint_sha256", "prediction_settings",
            "ultralytics_version", "output", "purpose", "final_calibration_threshold",
            "git_commit", "git_dirty"})

    def test_extract_parser_dispatches_mock_only(self):
        with patch.object(confidence, "find_project_root", return_value=self.root), \
                patch.object(confidence, "os", SimpleNamespace(name="posix")), \
                patch.object(confidence, "load_paths", return_value=SimpleNamespace(
                    project_root=self.root, dataset_root=self.dataset)), \
                patch.object(confidence, "extract", return_value="synthetic") as extract, \
                patch.object(confidence, "_predict", side_effect=AssertionError("no inference")) as predict, \
                contextlib.redirect_stdout(io.StringIO()):
            confidence.main(["extract", "--dicc"])
            extract.assert_called_once_with(self.root, self.dataset, predict)
            predict.assert_not_called()

    def test_extract_requires_dicc_flag_before_paths_or_prediction(self):
        with patch.object(confidence, "find_project_root", return_value=self.root), \
                patch.object(confidence, "os", SimpleNamespace(name="posix")), \
                patch.object(confidence, "load_paths") as paths, \
                patch.object(confidence, "extract") as extract:
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                confidence.main(["extract"])
            paths.assert_not_called()
            extract.assert_not_called()

    def test_row_class_and_geometry_validation(self):
        original = row(self.images[0], 0.1)
        for changes in ({"class_id": True}, {"class_id": 9}, {"class_name": "wrong"},
                        {"x1": -1}, {"x2": 0}, {"y2": 1}, {"x1": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                confidence.validate_row({**original, **changes}, set(self.images))

    def fake_model(self, settings):
        image = self.images[0]
        source = self.dataset / image
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"inert synthetic image fixture; never decoded")
        def tensor(values):
            return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values))
        result = SimpleNamespace(path=str(source), boxes=SimpleNamespace(
            xyxy=tensor([[1., 2., 3., 4.]]), conf=tensor([0.1]), cls=tensor([0.])))
        model = SimpleNamespace(names=confidence.rq2_detector_training.NAMES.copy(),
                                predictor=SimpleNamespace(
                                    args=SimpleNamespace(**{**settings, "device": "0"}),
                                    device="synthetic-device"), predict=Mock(return_value=[result]))
        stub = ModuleType("ultralytics")
        stub.YOLO = Mock(return_value=model)
        return stub, model, source

    def test_prediction_adapter_with_inert_module_stub(self):
        self.assertNotIn("ultralytics", sys.modules)
        for override in (None, {**confidence.SETTINGS, "conf": 0.01}):
            with self.subTest(override=override is not None):
                settings = confidence.SETTINGS if override is None else override
                stub, model, source = self.fake_model(settings)
                metadata = {}
                with patch.dict(sys.modules, {"ultralytics": stub}):
                    results = list(confidence._predict(
                        self.root, self.dataset, [self.images[0]], metadata,
                        settings=override, record_image_hashes=override is not None))
                self.assertEqual(results, [(self.images[0], [row(self.images[0], 0.1)])])
                stub.YOLO.assert_called_once_with(str(self.root / confidence.CHECKPOINT))
                model.predict.assert_called_once_with(source=str(source), **settings)
                self.assertEqual(metadata["effective_prediction_settings"], {**settings, "device": "0"})
                self.assertEqual(metadata["resolved_device"], "synthetic-device")
                if override is not None:
                    self.assertEqual(metadata["source_image_sha256"], {
                        self.images[0]: hashlib.sha256(source.read_bytes()).hexdigest()})
                else:
                    self.assertNotIn("source_image_sha256", metadata)
        self.assertNotIn("ultralytics", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_prediction_adapter_rejects_class_source_and_setting_drift(self):
        for changed in ("class", "source", "settings"):
            with self.subTest(changed=changed):
                stub, model, _ = self.fake_model(confidence.SETTINGS)
                if changed == "class":
                    model.names = {0: "wrong"}
                elif changed == "source":
                    model.predict.return_value[0].path = str(self.dataset / "wrong.jpg")
                else:
                    model.predictor.args.conf = 0.5
                with patch.dict(sys.modules, {"ultralytics": stub}), \
                        self.assertRaises((ValueError, RuntimeError)):
                    list(confidence._predict(self.root, self.dataset, [self.images[0]], {}))


if __name__ == "__main__":
    unittest.main()
