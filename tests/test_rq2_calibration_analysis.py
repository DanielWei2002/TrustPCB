"""Synthetic CPU-only correctness/calibration tests; no detector or real labels."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trustpcb import rq2_calibration_analysis as analysis


IMAGE = "images/train/synthetic.jpg"


def prediction(conf=0.9, box=(0, 0, 4, 1), cls=0, image=IMAGE):
    return dict(zip(analysis.raw.FIELDS, (image, cls, analysis.rq2_detector_training.NAMES[cls], conf, *box)))


def gt(box=(0, 0, 4, 1), cls=0, identifier="labels/train/synthetic.txt:1"):
    return {"id": identifier, "class_id": cls, "box": box}


class MatchingTests(unittest.TestCase):
    def label(self, predictions, truth):
        return analysis.label_predictions([IMAGE], predictions, {IMAGE: truth})

    def test_one_to_one_descending_confidence_duplicate(self):
        rows = self.label([prediction(0.2), prediction(0.9)], [gt()])
        self.assertEqual([r["confidence"] for r in rows], [0.9, 0.2])
        self.assertEqual([r["correct_at_iou50"] for r in rows], [1, 0])
        self.assertEqual(rows[1]["status_iou50"], "duplicate")
        self.assertEqual(rows[1]["matched_gt_id_iou50"], "")
        self.assertEqual(rows[1]["best_same_class_iou_iou50"], 1)

    def test_same_class_and_unmatched(self):
        rows = self.label([prediction(cls=1)], [gt(cls=0)])
        self.assertEqual(rows[0]["correct_at_iou50"], 0)
        self.assertEqual(rows[0]["status_iou50"], "unmatched")
        self.assertEqual(self.label([prediction(box=(10, 10, 12, 12))], [gt()])[0]["correct_at_iou75"], 0)
        self.assertEqual(self.label([prediction()], [])[0]["status_iou50"], "unmatched")

    def test_exact_iou50_boundary(self):
        row = self.label([prediction()], [gt(box=(0, 0, 2, 1))])[0]
        self.assertEqual(row["matched_iou_iou50"], 0.5)
        self.assertEqual(row["correct_at_iou50"], 1)
        self.assertEqual(row["correct_at_iou75"], 0)

    def test_exact_iou75_boundary(self):
        row = self.label([prediction()], [gt(box=(0, 0, 3, 1))])[0]
        self.assertEqual(row["matched_iou_iou75"], 0.75)
        self.assertEqual(row["correct_at_iou75"], 1)

    def test_sensitivity_uses_independent_unused_state(self):
        rows = self.label([prediction(0.9, box=(0, 0, 8, 1)), prediction(0.8)], [gt()])
        self.assertEqual([r["correct_at_iou50"] for r in rows], [1, 0])
        self.assertEqual([r["correct_at_iou75"] for r in rows], [0, 1])

    def test_highest_unused_iou_and_deterministic_ties(self):
        truth = [gt(box=(0, 0, 2, 1), identifier="first"), gt(identifier="second")]
        inputs = [prediction(0.9), prediction(0.9)]
        rows = self.label(inputs, truth)
        self.assertEqual([r["matched_gt_id_iou50"] for r in rows], ["second", "first"])
        self.assertEqual([r["source_row"] for r in rows], [1, 2])
        self.assertEqual(rows, self.label(inputs, truth))
        tied = self.label([prediction()], [gt(identifier="first"), gt(identifier="second")])
        self.assertEqual(tied[0]["matched_gt_id_iou50"], "first")
        self.assertEqual(analysis.summarize(rows), analysis.summarize(self.label(inputs, truth)))

    def test_fixed_inclusion_boundary_and_foreign_row(self):
        rows = self.label([prediction(0.009999), prediction(0.01)], [gt()])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["confidence"], 0.01)
        self.assertEqual(rows[0]["correct_at_iou50"], 1)
        with self.assertRaises(ValueError):
            self.label([prediction(0.001, image="images/val/test.jpg")], [])

    def test_known_metrics_and_bin_boundaries(self):
        rows = [{"confidence": 0.2, "y": 0}, {"confidence": 0.8, "y": 1}]
        result = analysis.metrics(rows, "y")
        self.assertAlmostEqual(result["brier"], 0.04)
        self.assertAlmostEqual(result["negative_log_likelihood"], -math.log(0.8))
        self.assertAlmostEqual(result["ece"], 0.2)
        self.assertEqual(result["reliability_bins"][2]["count"], 1)
        self.assertEqual(result["reliability_bins"][8]["count"], 1)
        result = analysis.metrics([{"confidence": 1., "y": 0}], "y")
        self.assertEqual(result["reliability_bins"][-1]["count"], 1)
        self.assertTrue(math.isfinite(result["negative_log_likelihood"]))
        self.assertIsNone(analysis.metrics([], "y")["ece"])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.directory = self.root / analysis.raw.OUTPUT
        self.directory.mkdir(parents=True)
        manifest = self.root / analysis.raw.DEVELOPMENT
        manifest.parent.mkdir(parents=True)
        content = (IMAGE + "\n").encode()
        manifest.write_bytes(content)
        for mock in (patch.object(analysis.raw, "IMAGE_COUNT", 1),
                     patch.object(analysis.raw, "MANIFEST_SHA", hashlib.sha256(content).hexdigest()),
                     patch.object(analysis.rq1, "git_provenance", return_value={"git_commit": "a" * 40, "git_dirty": False}),
                     patch.object(analysis.importlib.metadata, "version", return_value="synthetic")):
            mock.start()
            self.addCleanup(mock.stop)
        self.write_input()

    def write_input(self, image=IMAGE):
        with (self.directory / "predictions.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=analysis.raw.FIELDS)
            writer.writeheader()
            writer.writerow(prediction(image=image))
        (self.directory / "images.csv").write_text(f"image,prediction_count\n{IMAGE},1\n")
        self.provenance = {"status": "complete", "development_manifest": analysis.raw.DEVELOPMENT,
                           "manifest_sha256_lf": analysis.raw.MANIFEST_SHA, "development_images": 1,
                           "checkpoint": analysis.raw.CHECKPOINT, "checkpoint_epoch": 72,
                           "checkpoint_sha256": analysis.raw.CHECKPOINT_SHA,
                           "checkpoint_identity": {"sha256": analysis.raw.CHECKPOINT_SHA},
                           "output_sha256": {p: analysis.rq1._sha(self.directory / p) for p in ("predictions.csv", "images.csv")}}
        analysis.rq1._write_json(self.directory / "provenance.json", self.provenance)

    def test_development_only_and_changed_manifest(self):
        analysis.read_predictions(self.root)
        self.write_input("images/val/reserved-test.jpg")
        with self.assertRaises(ValueError):
            analysis.read_predictions(self.root)
        (self.root / analysis.raw.DEVELOPMENT).write_text("images/val/reserved-test.jpg\n")
        with self.assertRaises(ValueError):
            analysis.read_predictions(self.root)

    def test_provenance_and_prediction_tampering(self):
        self.provenance["checkpoint_epoch"] = 71
        analysis.rq1._write_json(self.directory / "provenance.json", self.provenance)
        with self.assertRaises(ValueError):
            analysis.read_predictions(self.root)
        self.write_input()
        with (self.directory / "predictions.csv").open("a") as file:
            file.write("tampered\n")
        with self.assertRaises(ValueError):
            analysis.read_predictions(self.root)

    def test_complete_outputs_protection_and_input_immutability(self):
        before = {p: p.read_bytes() for p in self.directory.iterdir()}
        def truth_reader(dataset, images):
            self.assertEqual(images, [IMAGE])
            return {IMAGE: [gt()]}, {IMAGE: {"synthetic": True}}
        def plotter(rows, report, output):
            (output / "synthetic-plot.txt").write_text("fixture")
        output = analysis.analyze(self.root, self.root, truth_reader, plotter)
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(report["primary_iou50"]["correct"], 1)
        self.assertEqual(report["total_retained_predictions"], 1)
        self.assertEqual(before, {p: p.read_bytes() for p in self.directory.iterdir()})
        with self.assertRaises(FileExistsError):
            analysis.analyze(self.root, self.root, truth_reader, plotter)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("ultralytics", sys.modules)

    def test_ground_truth_conversion_development_paths_only(self):
        labels = self.root / "labels/train/synthetic.txt"
        labels.parent.mkdir(parents=True)
        labels.write_text("0 0.5 0.5 0.2 0.5\n")
        opened = MagicMock()
        opened.__enter__.return_value = SimpleNamespace(size=(100, 80), getexif=lambda: {})
        image_api = SimpleNamespace(open=MagicMock(return_value=opened))
        with patch.dict(sys.modules, {"PIL": SimpleNamespace(Image=image_api)}):
            truth, evidence = analysis.read_ground_truth(self.root, [IMAGE])
        image_api.open.assert_called_once_with((self.root / IMAGE).resolve())
        self.assertEqual(truth[IMAGE][0]["box"], (40., 20., 60., 60.))
        self.assertEqual(truth[IMAGE][0]["id"], "labels/train/synthetic.txt:1")
        self.assertEqual(evidence[IMAGE]["objects"], 1)

    def test_windows_analysis_gate(self):
        with patch.object(os, "name", "nt"):
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                analysis.main(["analyze", "--dicc"])

    def test_completed_historical_predictions_read_without_metadata_rewrite(self):
        from trustpcb.rq2_artifact_paths import historical_name
        self.provenance["development_manifest"] = historical_name(analysis.raw.DEVELOPMENT)
        self.provenance["checkpoint"] = historical_name(analysis.raw.CHECKPOINT)
        analysis.rq1._write_json(self.directory / "provenance.json", self.provenance)
        historical = self.root / historical_name(analysis.raw.OUTPUT)
        historical.parent.mkdir(parents=True, exist_ok=True)
        self.directory.rename(historical)
        before = {p.name: p.read_bytes() for p in historical.iterdir()}
        images, rows, _ = analysis.read_predictions(self.root)
        self.assertEqual(images, [IMAGE])
        self.assertEqual(len(rows), 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in historical.iterdir()})
        # The same immutable metadata is accepted after moving the directory too.
        historical.rename(self.directory)
        self.assertEqual(analysis.read_predictions(self.root)[1], rows)


if __name__ == "__main__":
    unittest.main()
