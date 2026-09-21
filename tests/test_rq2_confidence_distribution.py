"""Confidence distribution tests: text and tiny synthetic predictions only, no ML imports."""

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trustpcb import rq2_confidence_distribution as confidence


def row(image, score):
    return dict(zip(confidence.FIELDS, (image, 0, "SH", score, 1., 2., 3., 4.)))


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


if __name__ == "__main__":
    unittest.main()
