"""Synthetic callbacks, fake clocks and tiny pixel arrays; no ML/GPU imports."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from trustpcb import rq3_computational_efficiency as benchmark


class TimingTests(unittest.TestCase):
    def test_warmup_order_synchronization_and_five_repetitions(self):
        images = [f"synthetic_{i}" for i in range(23)]
        events = []
        def pipeline(name):
            return lambda image: (events.append((name, image)) or benchmark.PASSES[name])
        tick = iter(range(20))
        def clock():
            events.append(("clock",))
            return next(tick)
        records = benchmark.measure(images, {name: pipeline(name) for name in benchmark.PASSES},
                                    lambda: events.append(("sync",)), clock=clock)
        expected = []
        for name in benchmark.PASSES:
            expected.extend((name, image) for image in images[:20])
            expected.append(("sync",))
        for _ in range(5):
            for name in benchmark.PASSES:
                expected.extend([("sync",), ("clock",)])
                expected.extend((name, image) for image in images)
                expected.extend([("sync",), ("clock",)])
        self.assertEqual(events, expected)
        self.assertEqual(len(records), 10)
        self.assertTrue(all(r["seconds"] == 1 for r in records))
        self.assertEqual(sum(r["detector_passes"] for r in records if r["pipeline"] == "trustpcb"), 23*12*5)

    def test_summary_units_standard_deviation_and_overhead(self):
        records = [{"pipeline": name, "repetition": i+1, "crops": 821,
                    "detector_passes": 821*benchmark.PASSES[name], "seconds": factor*(i+1)}
                   for name, factor in (("baseline", 1), ("trustpcb", 3)) for i in range(5)]
        summary = benchmark.summarize(records)
        self.assertEqual(summary["relative_overhead"], 3)
        self.assertEqual(summary["percentage_increase"], 200)
        self.assertAlmostEqual(summary["pipelines"]["baseline"]["standard_deviation_seconds"], np.std([1, 2, 3, 4, 5], ddof=1))
        self.assertAlmostEqual(summary["pipelines"]["trustpcb"]["mean_ms_per_crop"], 9000/821)
        self.assertAlmostEqual(summary["pipelines"]["trustpcb"]["throughput_crops_per_second"], 821/9)
        self.assertEqual(summary["pipelines"]["trustpcb"]["detector_passes_per_repetition"], 9852)
        records[0]["seconds"] = 0
        with self.assertRaises(ValueError):
            benchmark.summarize(records)

    def test_invalid_inventory_and_pass_counts(self):
        with self.assertRaises(ValueError):
            benchmark.measure(["a"], {}, lambda: None)
        images = [str(i) for i in range(20)]
        with self.assertRaisesRegex(RuntimeError, "warmup"):
            benchmark.measure(images, {"baseline": lambda _: 0, "trustpcb": lambda _: 12}, lambda: None)
        with self.assertRaises(ValueError):
            benchmark.summarize([])


class PipelineTests(unittest.TestCase):
    def test_frozen_transformation_specs(self):
        benchmark.validate_specs()
        self.assertEqual(len(benchmark.stability.SPECS), 11)
        with patch.object(benchmark.stability, "SPECS", benchmark.stability.SPECS[:-1]):
            with self.assertRaises(ValueError):
                benchmark.validate_specs()
        changed = [dict(s) for s in benchmark.stability.SPECS]
        changed[0]["factor"] = .8
        with patch.object(benchmark.stability, "SPECS", changed):
            with self.assertRaises(ValueError):
                benchmark.validate_specs()

    def test_full_crop_reuses_frozen_math_and_twelve_passes(self):
        detections = [{"box": [3, 3, 7, 7], "confidence": .5, "class_id": 0},
                      {"box": [1, 1, 2, 2], "confidence": .005, "class_id": 0}]
        beta = {"method": "Beta", "parameters": [1., 1., 0.]}
        with patch.object(benchmark, "predict", return_value=(detections, (10, 10))) as predict, \
             patch.object(benchmark.stability, "transform", wraps=benchmark.stability.transform) as transform, \
             patch.object(benchmark.stability, "aggregate", wraps=benchmark.stability.aggregate) as aggregate, \
             patch.object(benchmark.fusion, "weighted_risk", wraps=benchmark.fusion.weighted_risk) as weighted:
            result = benchmark.full_crop(object(), Path("synthetic.png"), "synthetic", beta,
                                         pixel_loader=lambda _: np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(result, 12)
        self.assertEqual(predict.call_count, 12)
        self.assertEqual([c.args[1] for c in transform.call_args_list], list(benchmark.stability.SPECS))
        self.assertEqual(aggregate.call_count, 1)  # Below-inclusion prediction is not a reference.
        self.assertEqual(weighted.call_args.args[1], (89, 1, 10))

    def test_empty_reference_crop_still_runs_all_transforms(self):
        with patch.object(benchmark, "predict", return_value=([], (4, 4))) as predict, \
             patch.object(benchmark.fusion, "weighted_risk", side_effect=AssertionError("no scores without references")):
            self.assertEqual(benchmark.full_crop(object(), Path("synthetic"), "synthetic", {},
                pixel_loader=lambda _: np.zeros((4, 4, 3), dtype=np.uint8)), 12)
        self.assertEqual(predict.call_count, 12)

    def test_predict_preserves_settings_and_postprocessing(self):
        effective = SimpleNamespace(**{**benchmark.raw.SETTINGS, "device": "0"})
        model = SimpleNamespace(predictor=SimpleNamespace(args=effective))
        seen = []
        model.predict = lambda **kwargs: (seen.append(kwargs) or [SimpleNamespace(boxes=None, orig_shape=(4, 5))])
        self.assertEqual(benchmark.predict(model, "synthetic"), ([], (4, 5)))
        self.assertEqual(seen, [{"source": "synthetic", **benchmark.raw.SETTINGS}])
        effective.batch = 2
        with self.assertRaises(RuntimeError):
            benchmark.predict(model, "synthetic")

    def test_windows_guard_precedes_ml_imports(self):
        with patch.object(benchmark.os, "name", "nt"):
            with self.assertRaises(RuntimeError):
                benchmark.adapter(Path("synthetic"), {}, {})
            with patch.object(benchmark, "run") as run:
                with self.assertRaises(RuntimeError):
                    benchmark.main(["--dicc"])
                run.assert_not_called()


class WorkflowTests(unittest.TestCase):
    def test_synthetic_outputs_isolation_and_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root/"synthetic_dataset"
            dataset.mkdir()
            images = [f"synthetic_{i}.png" for i in range(20)]
            for image in images:
                (dataset/image).write_bytes(b"synthetic fixture, not an image")
            sentinel = root/"runs/rq2/immutable.txt"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b"immutable")
            versions = lambda name: benchmark.raw.VERSION if name == "ultralytics" else "synthetic"
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda _: "synthetic GPU"),
                version=SimpleNamespace(cuda="synthetic"), backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 0)))
            model = SimpleNamespace(predictor=SimpleNamespace(device=SimpleNamespace(type="cuda", index=0),
                args=SimpleNamespace(**benchmark.raw.SETTINGS)))
            inputs = (images, {"synthetic": "hash"}, {}, {}, {"method": "Beta", "parameters": [1, 1, 0]})
            def factory(*args):
                return {name: (lambda image, n=n: n) for name, n in benchmark.PASSES.items()}, lambda: None, model
            with patch.object(benchmark, "IMAGE_COUNT", 20), \
                 patch.object(benchmark, "load_inputs", return_value=inputs), \
                 patch.object(benchmark.rq1, "git_provenance", return_value={"git_dirty": False, "git_commit": "synthetic"}), \
                 patch.object(benchmark.importlib.metadata, "version", side_effect=versions), \
                 patch.dict(sys.modules, {"torch": fake_torch}):
                output = benchmark.run(root, dataset, adapter_factory=factory)
                with self.assertRaises(FileExistsError):
                    benchmark.run(root, dataset, adapter_factory=factory)
            self.assertEqual(sentinel.read_bytes(), b"immutable")
            provenance = benchmark.rq1._read_json(output/"provenance.json")
            self.assertEqual(provenance["status"], "complete")
            self.assertEqual(provenance["warmup_images"], images)
            self.assertFalse(provenance["test_population_accessed"])
            self.assertEqual(len(benchmark.risk.read_csv(output/"repetition_timings.csv")), 10)
            for name, digest in provenance["output_sha256"].items():
                self.assertEqual(benchmark.rq1._sha(output/name), digest)

    def test_dirty_inputs_block_before_loading_population(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(benchmark.rq1, "git_provenance", return_value={"git_dirty": True}), \
             patch.object(benchmark, "load_inputs") as load:
            with self.assertRaises(RuntimeError):
                benchmark.run(tmp, tmp)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
