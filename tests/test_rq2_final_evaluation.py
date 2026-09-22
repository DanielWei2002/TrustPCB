"""Synthetic final-test orchestration; no real test data or model libraries."""

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustpcb import rq2_final_evaluation as evaluation


def prediction(image, confidence=.8, box=(20, 20, 40, 40)):
    return dict(zip(evaluation.raw.FIELDS, (image, 0, "SH", confidence, *box)))


class FrozenInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.images = [f"images/val/synthetic_{i}.jpg" for i in range(2052)]
        manifest = self.root / evaluation.MANIFEST
        manifest.parent.mkdir(parents=True)
        manifest.write_text("\n".join(self.images) + "\n")
        self.report_path = self.root / evaluation.PARTITION_REPORT
        self.report_path.parent.mkdir(parents=True)
        self.report = {"input_sha256": {evaluation.MANIFEST: evaluation.lf_hash(manifest)}}
        evaluation.rq1._write_json(self.report_path, self.report)
        train = self.root / "data/splits/synthetic_train.txt"
        dev = self.root / "data/splits/synthetic_dev.txt"
        train.write_text("images/train/train.jpg\n")
        dev.write_text("images/train/dev.jpg\n")
        specs = {"train": (train.relative_to(self.root).as_posix(), 1, evaluation.lf_hash(train)),
                 "val": (dev.relative_to(self.root).as_posix(), 1, evaluation.lf_hash(dev))}
        for mocked in (patch.object(evaluation, "MANIFEST_SHA", evaluation.lf_hash(manifest)),
                       patch.object(evaluation, "REPORT_SHA", evaluation.lf_hash(self.report_path)),
                       patch.object(evaluation.raw.rq2_detector_training, "MANIFESTS", specs)):
            mocked.start()
            self.addCleanup(mocked.stop)
        self.checkpoint = self.root / evaluation.raw.CHECKPOINT
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"synthetic checkpoint")
        mocked = patch.object(evaluation.raw, "CHECKPOINT_SHA", evaluation.rq1._sha(self.checkpoint))
        mocked.start()
        self.addCleanup(mocked.stop)
        self.folder = self.root / evaluation.final.OUTPUT
        self.folder.mkdir(parents=True)
        self.artifact = {"method": "Beta", "epsilon": 1e-15, "mathematical_form": evaluation.final.FORM,
            **dict(zip(("a", "b", "c"), evaluation.BETA)), "inclusion_threshold": .01,
            "correctness_target": evaluation.final.TARGET, "development_images": 821, "fitted_predictions": 7131,
            "input_sha256": {}}
        evaluation.rq1._write_json(self.folder / "fit_summary.json", {"synthetic": True})
        (self.folder / "calibrated_development_predictions.csv").write_text("synthetic\n")
        self.bind_calibrator()

    def bind_calibrator(self):
        evaluation.rq1._write_json(self.folder / "final_calibrator.json", self.artifact)
        evaluation.rq1._write_json(self.folder / "provenance.json", {"status": "complete", "final_fit_performed": True,
            "beta_selection_reviewed": True, "input_sha256": {}, "output_sha256": {
                name: evaluation.rq1._sha(self.folder / name) for name in ("final_calibrator.json", "fit_summary.json", "calibrated_development_predictions.csv")}})

    def test_exact_synthetic_2052_and_provenance_hash(self):
        self.assertEqual(evaluation.frozen_test_images(self.root), self.images)
        images, hashes, identity = evaluation.frozen_inputs(self.root)
        self.assertEqual(len(images), 2052)
        self.assertIn(evaluation.MANIFEST, hashes)
        self.assertEqual(identity["epoch"], 72)

    def test_count_guard_even_when_hashes_consistent(self):
        manifest = self.root / evaluation.MANIFEST
        manifest.write_text("\n".join(self.images[:-1]) + "\n")
        digest = evaluation.lf_hash(manifest)
        self.report["input_sha256"][evaluation.MANIFEST] = digest
        evaluation.rq1._write_json(self.report_path, self.report)
        with patch.object(evaluation, "MANIFEST_SHA", digest), patch.object(evaluation, "REPORT_SHA", evaluation.lf_hash(self.report_path)):
            with self.assertRaisesRegex(ValueError, "2052"):
                evaluation.frozen_test_images(self.root)

    def test_train_dev_overlap_rejected(self):
        relative, _, _ = evaluation.raw.rq2_detector_training.MANIFESTS["val"]
        path = self.root / relative
        path.write_text(self.images[0] + "\n")
        specs = {**evaluation.raw.rq2_detector_training.MANIFESTS, "val": (relative, 1, evaluation.lf_hash(path))}
        with patch.object(evaluation.raw.rq2_detector_training, "MANIFESTS", specs):
            with self.assertRaisesRegex(ValueError, "overlaps"):
                evaluation.frozen_test_images(self.root)

    def test_manifest_tampering_and_report_tampering(self):
        self.report_path.write_text(self.report_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            evaluation.frozen_test_images(self.root)

    def test_checkpoint_changed_and_historical_alias(self):
        historical = self.root / "runs/rq2/stage2/seed_24209199/weights/selected.pt"
        historical.parent.mkdir(parents=True)
        self.checkpoint.rename(historical)
        evaluation.frozen_inputs(self.root)
        historical.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Checkpoint differs"):
            evaluation.frozen_inputs(self.root)

    def test_changed_beta_even_if_provenance_rehashed(self):
        self.artifact["a"] += .01
        self.bind_calibrator()
        with self.assertRaisesRegex(ValueError, "Beta calibrator differs"):
            evaluation.frozen_inputs(self.root)

    def test_calibrator_artifact_hash_changed(self):
        path = self.folder / "final_calibrator.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "output hash mismatch"):
            evaluation.frozen_inputs(self.root)


class ReuseTests(unittest.TestCase):
    def test_original_adapter_forwards_settings_without_changing_development_defaults(self):
        with patch.object(evaluation.os, "name", "posix"), patch.object(evaluation.raw, "_predict", return_value=iter(())) as predict:
            list(evaluation._original("synthetic-root", "synthetic-dataset", ["synthetic"], {}))
        self.assertEqual(predict.call_args.kwargs["settings"]["conf"], .01)
        self.assertTrue(predict.call_args.kwargs["record_image_hashes"])
        self.assertEqual(evaluation.raw.SETTINGS["conf"], .001)

    def test_changed_transform_strength_rejected(self):
        changed = copy.deepcopy(evaluation.stability.SPECS)
        changed[0]["factor"] = .8
        with patch.object(evaluation.stability, "SPECS", changed):
            with self.assertRaisesRegex(ValueError, "transformation definitions"):
                evaluation.validate_protocol()

    def test_frozen_settings_and_eleven_transforms(self):
        self.assertEqual(evaluation.ORIGINAL_SETTINGS["conf"], .01)
        self.assertEqual(evaluation.raw.SETTINGS["conf"], .001)
        self.assertEqual(evaluation.raw.SETTINGS["max_det"], 300)
        self.assertEqual(evaluation.ORIGINAL_SETTINGS["imgsz"], 640)
        self.assertEqual(len(evaluation.stability.SPECS), 11)

    def test_correctness_independent_matching_and_frozen_beta(self):
        image = "images/val/synthetic.jpg"
        predictions = [prediction(image, .9, (20, 20, 40, 40)), prediction(image, .8, (20, 20, 40, 32)), prediction(image, .009)]
        truth = {image: [{"id": "syntheticGT", "class_id": 0, "box": (20, 20, 40, 32)}]}
        with patch.object(evaluation.calibration, "fit_calibrator", side_effect=AssertionError("refit forbidden")):
            rows = evaluation.calibrate_and_label([image], predictions, truth)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["correct_iou50"] for r in rows], [1, 0])
        self.assertEqual([r["correct_iou75"] for r in rows], [0, 1])
        expected = evaluation.calibration.apply_calibrator({"method": "Beta", "parameters": evaluation.BETA}, [.9, .8])
        np.testing.assert_allclose([r["calibrated_confidence"] for r in rows], expected)
        self.assertEqual(rows[1]["status_iou50"], "duplicate")

    def test_risks_and_primary_secondary_draws_reused(self):
        rows = [{"image": "a", "prediction_id": "a1", "raw_confidence": .3, "calibrated_confidence": .4,
                 "class_consistency": .7, "localisation_stability": .9, "correct_iou50": 0, "correct_at_iou75": 0},
                {"image": "b", "prediction_id": "b1", "raw_confidence": .8, "calibrated_confidence": .9,
                 "class_consistency": .8, "localisation_stability": .9, "correct_iou50": 1, "correct_at_iou75": 0}]
        scores = evaluation.risk.risk_scores(rows)
        np.testing.assert_allclose([scores[m][0] for m in evaluation.risk.METHODS], [.7, .6, .3, .1, .45, .35, .2, 1/3])
        with tempfile.TemporaryDirectory() as tmp, patch.object(evaluation, "REPLICATES", 32):
            output = Path(tmp)
            first = evaluation.ranking_outputs(output, ["a", "b", "zero"], rows, scores)
            second = evaluation.ranking_outputs(output, ["a", "b", "zero"], rows, scores, secondary=True)
            self.assertEqual(first, second)
            summary = evaluation.rq1._read_json(output / "sensitivity_iou75/summary.json")
            self.assertEqual(summary["incorrect"], 2)
            self.assertEqual(summary["analysis_label"], evaluation.sensitivity.LABEL)
            metrics = evaluation.risk.read_csv(output / "primary_iou50/method_metrics.csv")
            self.assertTrue(all(float(r["auroc"]) == 1 for r in metrics if r["method"] != "localisation_risk"))
            boot = evaluation.rq1._read_json(output / "sensitivity_iou75/bootstrap_summary.json")
            self.assertTrue(all(r["auroc_invalid_replicates"] == 32 for r in boot["validity_by_method"]))


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = ["images/val/synthetic_a.jpg", "images/val/synthetic_b.jpg"]
        self.evidence = {}
        for image in self.images:
            path = self.root / image
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic pixels are never decoded")
            label = image.replace("images/", "labels/").replace(".jpg", ".txt")
            label_path = self.root / label
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("synthetic labels never parsed")
            self.evidence[image] = {"label": label, "label_sha256": evaluation.rq1._sha(label_path), "width": 100, "height": 100}
        self.truth = {self.images[0]: [{"id": "syntheticGT", "class_id": 0, "box": (20, 20, 40, 32)}], self.images[1]: []}
        for mocked in (patch.object(evaluation, "frozen_inputs", return_value=(self.images, {f"{evaluation.final.OUTPUT}/final_calibrator.json": "synthetic"}, {"epoch": 72, "sha256": "synthetic"})),
                       patch.object(evaluation.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"}),
                       patch.object(evaluation.importlib.metadata, "version", side_effect=lambda name: evaluation.raw.VERSION if name == "ultralytics" else "synthetic"),
                       patch.object(evaluation, "REPLICATES", 16)):
            mocked.start()
            self.addCleanup(mocked.stop)

    def original(self, root, dataset, images, metadata):
        metadata["source_image_sha256"] = {image: evaluation.rq1._sha(dataset / image) for image in images}
        for image in images:
            yield image, [prediction(image, .9), prediction(image, .8, (20, 20, 40, 32))] if image == images[0] else []

    def transformed(self, root, dataset, images, metadata):
        for image in images:
            for spec in evaluation.stability.SPECS:
                # Exercise the exact development rasterisation on tiny synthetic arrays.
                _, matrix, median = evaluation.stability.transform(np.zeros((100, 100, 3), dtype=np.uint8), spec)
                detections = [{"class_id": 0, "confidence": .005, "box": [20, 20, 40, 40]}]
                yield image, 100, 100, spec["id"], detections, {"source_image_sha256": evaluation.rq1._sha(dataset / image),
                    "forward_affine": matrix.tolist(), "border_median_bgr": median}

    def execute(self, original=None, transformed=None):
        return evaluation.run(self.root, self.root, original or self.original, transformed or self.transformed,
                              lambda dataset, images: (copy.deepcopy(self.truth), copy.deepcopy(self.evidence)))

    def test_complete_pipeline_reuses_functions_no_fitting_development_isolation(self):
        sentinel = self.root / evaluation.final.OUTPUT / "synthetic_immutable.txt"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_bytes(b"development unchanged")
        before = evaluation.rq1._sha(sentinel)
        with patch.object(evaluation.calibration, "fit_calibrator", side_effect=AssertionError("no fitting")), \
             patch.object(evaluation.stability, "match_transform", wraps=evaluation.stability.match_transform) as matching, \
             patch.object(evaluation.stability, "aggregate", wraps=evaluation.stability.aggregate) as aggregate:
            output = self.execute()
        self.assertEqual(matching.call_count, 22)
        self.assertEqual(aggregate.call_count, 2)
        self.assertEqual(evaluation.rq1._sha(sentinel), before)
        provenance = evaluation.rq1._read_json(output / "provenance.json")
        self.assertEqual(provenance["test_role"], evaluation.ROLE)
        self.assertEqual(provenance["status"], "complete")
        self.assertEqual(provenance["beta_parameters"], list(evaluation.BETA))
        self.assertEqual(provenance["risk_formulas"], evaluation.risk.FORMULAS)
        for name, digest in provenance["output_sha256"].items():
            self.assertEqual(evaluation.rq1._sha(output / name), digest)
        labelled = evaluation.risk.read_csv(output / "labelled_predictions.csv")
        metrics = evaluation.rq1._read_json(output / "calibration_metrics.json")["metrics"]
        expected = evaluation.calibration.metric_summary([float(r["calibrated_confidence"]) for r in labelled], [int(r["correct_iou50"]) for r in labelled])
        self.assertEqual(metrics["Beta"], expected)
        self.assertEqual(len(evaluation.risk.read_csv(output / "calibration_bins.csv")), 20)
        audit = evaluation.rq1._read_json(output / "inference_audit.json")
        self.assertEqual(audit["expected_transformed_passes"], 22)
        summary = evaluation.rq1._read_json(output / "primary_iou50/summary.json")
        self.assertFalse(summary["threshold_selected"])
        self.assertFalse(summary["weights_tuned"])
        with self.assertRaises(FileExistsError):
            self.execute()

    def test_reference_floor_rejected_and_partial_never_reused(self):
        def below(root, dataset, images, metadata):
            yield images[0], [prediction(images[0], .009)]
        with self.assertRaisesRegex(ValueError, "reference floor"):
            self.execute(original=below)
        self.assertEqual(evaluation.rq1._read_json(self.root / evaluation.OUTPUT / "provenance.json")["status"], "incomplete")
        with self.assertRaises(FileExistsError):
            self.execute()

    def test_foreign_original_population_rejected(self):
        def foreign(*args):
            yield "images/train/development.jpg", []
        with self.assertRaisesRegex(RuntimeError, "Foreign"):
            self.execute(original=foreign)

    def test_missing_transformed_views_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete transformed"):
            self.execute(transformed=lambda *args: iter(()))

    def test_windows_gate_before_data_or_inference(self):
        with patch.object(evaluation.os, "name", "nt"), patch.object(evaluation, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "DICC-only"):
                evaluation.main(["--dicc"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
