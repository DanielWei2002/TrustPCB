"""Synthetic final-fit fixtures only; no scientific data, inference or GPU."""

import csv
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustpcb import rq2_final_calibrator as final

c = final.comparison


class FinalCalibratorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = [f"images/train/synthetic_{i}.jpg" for i in range(821)]
        manifest = self.root / c.raw.DEVELOPMENT
        manifest.parent.mkdir(parents=True)
        content = "\n".join(self.images) + "\n"
        manifest.write_text(content, encoding="utf-8")
        for mocked in (patch.object(c.raw, "MANIFEST_SHA", hashlib.sha256(content.encode()).hexdigest()),
                       patch.object(c, "EXPECTED_PREDICTIONS", 40),
                       patch.object(final.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"})):
            mocked.start()
            self.addCleanup(mocked.stop)
        pairs = self.root / c.partition.PAIRS
        pairs.parent.mkdir(parents=True)
        pairs.write_text("train_file,val_file\n", encoding="utf-8")
        self.analysis = self.root / c.analysis.OUTPUT
        self.analysis.mkdir(parents=True)
        self.comparison = self.root / c.OUTPUT
        self.comparison.mkdir(parents=True)
        self.write_population()
        self.bind_comparison()

    def write_population(self, size=40, sensitivity="opaque-sensitivity"):
        table = self.analysis / "labelled_predictions.csv"
        with table.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("image", "source_row", "class_id", "x1", "confidence", "correct_at_iou50", "correct_at_iou75"))
            writer.writeheader()
            for i in range(size):
                writer.writerow({"image": self.images[i % 821], "source_row": i + 1,
                                 "class_id": i % 3, "x1": i, "confidence": (.2, .2, .8, .8)[i % 4],
                                 "correct_at_iou50": i % 2, "correct_at_iou75": sensitivity})
        final.rq1._write_json(self.analysis / "summary.json", {"synthetic": True})
        self.bind_analysis()

    def bind_analysis(self):
        metadata = {"status": "complete", "inclusion_threshold": .01, "primary_iou": .5,
                    "checkpoint_epoch": 72, "checkpoint_sha256": c.raw.CHECKPOINT_SHA,
                    "manifest_sha256_lf": c.raw.MANIFEST_SHA,
                    "output_sha256": {name: final.rq1._sha(self.analysis / name)
                                      for name in ("labelled_predictions.csv", "summary.json")}}
        final.rq1._write_json(self.analysis / "provenance.json", metadata)

    def bind_comparison(self, method="Beta", status="comparison_complete_pending_review", complete="complete", fitted=False):
        final.rq1._write_json(self.comparison / "selection_summary.json", {
            "selected_method": method, "status": status, "final_fit_performed": fitted})
        (self.comparison / "method_metrics.csv").write_text("method,nll\nBeta,0.7\n", encoding="utf-8")
        hashes = c.load_inputs(self.root)[3]
        final.rq1._write_json(self.comparison / "provenance.json", {
            "status": complete, "final_fit_performed": fitted, "seed": c.SEED,
            "folds": 5, "resamples": 10000, "inclusion": .01, "primary_iou": .5,
            "log_epsilon": 1e-15, "git_commit": "synthetic-comparison", "input_sha256": hashes,
            "output_sha256": {name: final.rq1._sha(self.comparison / name)
                              for name in ("selection_summary.json", "method_metrics.csv")}})

    def load(self):
        return final.load_inputs(self.root, approve_beta_selection=True)

    def execute(self):
        return final.run(self.root, approve_beta_selection=True)

    def test_mapping_reuse_deterministic_monotonic_probabilities(self):
        p = [.2, .2, .8, .8]
        y = [0, 1, 0, 1]
        one = c.fit_calibrator("Beta", p, y)
        self.assertEqual(one, c.fit_calibrator("Beta", p, y))
        grid = np.linspace(0, 1, 101)
        values = c.apply_calibrator(one, grid)
        a, b, intercept = one["parameters"]
        safe = np.clip(grid, 1e-15, 1 - 1e-15)
        np.testing.assert_allclose(values, c.expit(a * np.log(safe) - b * np.log1p(-safe) + intercept))
        self.assertTrue(np.all(np.diff(values) >= 0))
        self.assertTrue(np.all((values >= 0) & (values <= 1)))

    def test_full_sized_synthetic_inventory_exactly_once_no_fit(self):
        with patch.object(c, "EXPECTED_PREDICTIONS", 7131):
            self.write_population(size=7131)
            self.bind_comparison()
            images, rows, originals, *_ = self.load()
        self.assertEqual(len(images), 821)
        self.assertEqual(len(rows), 7131)
        self.assertEqual(len({row["prediction_id"] for row in rows}), 7131)
        self.assertEqual([int(row["source_row"]) for row in originals], list(range(1, 7132)))

    def test_one_fit_all_rows_identity_metrics_and_provenance(self):
        _, rows, originals, before, *_ = self.load()
        with patch.object(c, "fit_calibrator", wraps=c.fit_calibrator) as fit:
            output = self.execute()
        fit.assert_called_once_with("Beta", [r["raw_confidence"] for r in rows], [r["correct_iou50"] for r in rows])
        artifact = final.rq1._read_json(output / "final_calibrator.json")
        self.assertEqual(artifact["mathematical_form"], final.FORM)
        self.assertEqual(artifact["input_sha256"], before)
        self.assertEqual(artifact["git_commit"], "synthetic")
        self.assertTrue(artifact["optimizer"]["converged"])
        self.assertEqual(artifact["epsilon"], 1e-15)
        self.assertEqual(artifact["source_comparison"]["selection"]["selected_method"], "Beta")
        with (output / "calibrated_development_predictions.csv").open() as file:
            exported = list(csv.DictReader(file))
        for original, row in zip(originals, exported):
            for key, value in original.items():
                self.assertEqual(row[key], value)
        self.assertEqual(len(exported), len(rows))
        report = final.rq1._read_json(output / "fit_summary.json")
        self.assertIn("in-sample", report["diagnostic_scope"])
        self.assertEqual(report["raw"], c.metric_summary([r["raw_confidence"] for r in rows], [r["correct_iou50"] for r in rows]))
        self.assertAlmostEqual(report["fitted_beta"]["nll"], np.log(2), places=6)
        self.assertAlmostEqual(report["fitted_beta"]["brier"], .25, places=6)
        provenance = final.rq1._read_json(output / "provenance.json")
        self.assertEqual(provenance["status"], "complete")
        for name, digest in provenance["output_sha256"].items():
            self.assertEqual(final.rq1._sha(output / name), digest)
        self.assertEqual(self.load()[3], before)

    def test_sensitivity_cannot_change_fit(self):
        rows = self.load()[1]
        self.write_population(sensitivity="entirely-different-IoU75")
        self.bind_comparison()
        self.assertEqual(self.load()[1], rows)

    def test_incorrect_population_rejected(self):
        with patch.object(c, "EXPECTED_PREDICTIONS", 7131):
            with self.assertRaisesRegex(ValueError, "7131"):
                self.load()

    def test_below_threshold_and_foreign_image_and_missing_label_rejected(self):
        table = self.analysis / "labelled_predictions.csv"
        original = table.read_text()
        for text in (original.replace("0.2", "0.009", 1),
                     original.replace(self.images[0], "images/val/final-test.jpg", 1),
                     original.replace("correct_at_iou50", "missing-primary-label")):
            table.write_text(text)
            self.bind_analysis()
            with self.assertRaises((ValueError, KeyError)):
                self.load()
        table.write_text(original)

    def test_non_beta_incomplete_or_already_fitted_rejected(self):
        for arguments in ({"method": "Platt"}, {"complete": "incomplete"},
                          {"status": "scientific_review_required"}, {"fitted": True}):
            self.bind_comparison(**arguments)
            with self.assertRaises(ValueError):
                self.load()

    def test_unreviewed_selection_rejected_before_reading(self):
        with patch.object(c, "load_inputs") as read:
            with self.assertRaisesRegex(ValueError, "Reviewed"):
                final.load_inputs(self.root)
        read.assert_not_called()

    def test_comparison_hash_tampering_rejected(self):
        (self.comparison / "method_metrics.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load()

    def test_comparison_population_mismatch_rejected(self):
        self.write_population(sensitivity="changed-input-after-comparison")
        with self.assertRaisesRegex(ValueError, "input hashes"):
            self.load()

    def test_complete_output_never_overwritten(self):
        self.execute()
        with patch.object(c, "fit_calibrator") as fit:
            with self.assertRaises(FileExistsError):
                self.execute()
        fit.assert_not_called()

    def test_failed_fit_leaves_incomplete_output_and_blocks_retry(self):
        with patch.object(c, "fit_calibrator", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                self.execute()
        provenance = final.rq1._read_json(self.root / final.OUTPUT / "provenance.json")
        self.assertEqual(provenance["status"], "incomplete")
        with self.assertRaises(FileExistsError):
            self.execute()

    def test_windows_gate_before_data_access(self):
        with patch.object(final.os, "name", "nt"), patch.object(final, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                final.main(["--dicc", "--approve-beta-selection"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
