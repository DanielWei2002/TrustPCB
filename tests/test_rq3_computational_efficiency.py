"""Synthetic callbacks, fake clocks and tiny pixel arrays; no ML/GPU imports."""

import importlib
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from trustpcb.rq3 import computational_efficiency as benchmark


class PackageCompatibilityTests(unittest.TestCase):
    def check_order(self, names):
        script = f'''
import importlib, sys
from unittest.mock import patch
first, second = [importlib.import_module(n) for n in {names!r}]
assert first is second
assert first.__name__ == 'trustpcb.rq3.computational_efficiency'
for owner, observer in ((first, second), (second, first)):
    with patch.object(owner, 'predict', return_value='synthetic') as predict:
        assert observer.predict(None, None) == 'synthetic'
        predict.assert_called_once_with(None, None)
    with patch.dict(owner.PASSES, synthetic=0):
        assert observer.PASSES['synthetic'] == 0
assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules
'''
        subprocess.run([sys.executable, "-B", "-c", script], check=True, capture_output=True)

    def test_legacy_first_and_lazy_imports(self):
        self.check_order(("trustpcb.rq3_computational_efficiency", "trustpcb.rq3.computational_efficiency"))

    def test_canonical_first_and_lazy_imports(self):
        self.check_order(("trustpcb.rq3.computational_efficiency", "trustpcb.rq3_computational_efficiency"))

    def test_canonical_dependencies_and_direct_binding(self):
        from trustpcb.rq2 import confidence_distribution, transformation_stability, risk_evaluation
        from trustpcb.rq2 import weighted_fusion, weighted_final_evaluation, artifact_paths
        for actual, expected in ((benchmark.raw, confidence_distribution),
                                 (benchmark.stability, transformation_stability),
                                 (benchmark.risk, risk_evaluation), (benchmark.fusion, weighted_fusion),
                                 (benchmark.resolve_input, artifact_paths.resolve_input)):
            self.assertIs(actual, expected)
        original = benchmark.selected_identity
        self.assertIs(original, weighted_final_evaluation.selected_identity)
        with patch.object(weighted_final_evaluation, "selected_identity") as replacement:
            self.assertIs(benchmark.selected_identity, original)
            self.assertIsNot(benchmark.selected_identity, replacement)
        root = Path("synthetic")
        with patch.object(benchmark, "selected_identity", return_value={"synthetic": True}) as selected, \
             patch.object(benchmark.risk, "load_inputs", return_value=(["a"], [], {})), \
             patch.object(benchmark.stability, "load_inputs", return_value=(["a"], [], {}, {})), \
             patch.object(benchmark, "resolve_input", return_value=root), \
             patch.object(benchmark.rq1, "_read_json", return_value={"a": 1, "b": 1, "c": 0}):
            self.assertEqual(benchmark.load_inputs(root),
                             (["a"], {}, {"synthetic": True}, {}, {"method": "Beta", "parameters": [1., 1., 0.]}))
        selected.assert_called_once_with(root)

    def test_default_bindings_and_alias_patches(self):
        legacy = importlib.import_module("trustpcb.rq3_computational_efficiency")
        clock = benchmark.perf_counter
        adapter = benchmark.adapter
        self.assertEqual(benchmark.measure.__kwdefaults__, {"clock": clock})
        self.assertEqual(benchmark.run.__kwdefaults__, {"adapter_factory": adapter})
        self.assertEqual(benchmark.summarize.__defaults__, (821,))
        for owner, observer in ((legacy, benchmark), (benchmark, legacy)):
            with patch.object(owner, "perf_counter"), patch.object(owner, "adapter"), \
                 patch.object(owner, "IMAGE_COUNT", 20):
                self.assertIs(observer.measure.__kwdefaults__["clock"], clock)
                self.assertIs(observer.run.__kwdefaults__["adapter_factory"], adapter)
                self.assertEqual(observer.summarize.__defaults__, (821,))
                self.assertIs(owner.adapter, observer.adapter)

    def test_both_cli_help_forms(self):
        for module in ("trustpcb.rq3_computational_efficiency", "trustpcb.rq3.computational_efficiency"):
            result = subprocess.run([sys.executable, "-B", "-m", module, "--help"],
                                    check=True, capture_output=True, text=True)
            self.assertIn("--dicc", result.stdout)

    def test_legacy_cli_dispatch(self):
        with patch.object(benchmark, "main") as main:
            runpy.run_path(str(Path(__file__).resolve().parents[1]/"src/trustpcb/rq3_computational_efficiency.py"),
                           run_name="__main__")
        main.assert_called_once_with()

    def test_parser_and_local_path_guards(self):
        root, dataset = Path("synthetic"), Path("synthetic_dataset")
        with patch.object(benchmark.os, "name", "posix"), \
             patch.object(benchmark, "find_project_root", return_value=root) as find, \
             patch.object(benchmark, "load_paths", return_value=SimpleNamespace(project_root=root, dataset_root=dataset)) as paths, \
             patch.object(benchmark, "run") as run, patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "--dicc required"):
                benchmark.main([])
            find.assert_not_called()
            benchmark.main(["--dicc"])
            paths.assert_called_once_with(root)
            run.assert_called_once_with(root, dataset)
            paths.return_value.project_root = dataset
            with self.assertRaisesRegex(ValueError, "project_root"):
                benchmark.main(["--dicc"])
            self.assertEqual(run.call_count, 1)

    def test_frozen_contract(self):
        self.assertEqual((benchmark.IMAGE_COUNT, benchmark.WARMUP_CROPS, benchmark.REPETITIONS), (821, 20, 5))
        self.assertEqual(list(benchmark.PASSES.items()), [("baseline", 1), ("trustpcb", 12)])
        self.assertEqual([821*n for n in benchmark.PASSES.values()], [821, 9852])
        self.assertEqual(benchmark.raw.VERSION, "8.4.117")
        self.assertEqual(benchmark.raw.EPOCH, 72)
        self.assertEqual(benchmark.raw.CHECKPOINT_SHA, "793992d25ef671c45c820dc1b5a61bca5837af0d3726cd3830b2f2c656a6a85f")
        self.assertEqual((benchmark.raw.SETTINGS["batch"], benchmark.raw.SETTINGS["device"]), (1, 0))
        self.assertEqual(benchmark.PROTOCOL["weights"], [.89, .01, .10])
        self.assertEqual(benchmark.OUTPUT, "runs/rq3/computational_efficiency")

    def test_missing_cuda_rejected_without_model_creation(self):
        root = Path("synthetic")
        from unittest.mock import Mock
        yolo = Mock(side_effect=AssertionError("no model creation"))
        fake_modules = {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                        "ultralytics": SimpleNamespace(YOLO=yolo),
                        "PIL": SimpleNamespace(Image=object(), ImageOps=object())}
        with patch.object(benchmark.os, "name", "posix"), patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback prohibited"):
                benchmark.adapter(root, {}, {})
        yolo.assert_not_called()


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
        self.assertEqual(list(records[0]), ["pipeline", "repetition", "crops", "seconds", "detector_passes",
                                           "milliseconds_per_crop", "throughput_crops_per_second"])
        self.assertTrue(all(r["seconds"] == 1 for r in records))
        self.assertEqual(sum(r["detector_passes"] for r in records if r["pipeline"] == "trustpcb"), 23*12*5)

    def test_summary_units_standard_deviation_and_overhead(self):
        records = [{"pipeline": name, "repetition": i+1, "crops": 821,
                    "detector_passes": 821*benchmark.PASSES[name], "seconds": factor*(i+1)}
                   for name, factor in (("baseline", 1), ("trustpcb", 3)) for i in range(5)]
        summary = benchmark.summarize(records)
        self.assertEqual(list(summary), ["pipelines", "relative_overhead", "percentage_increase"])
        self.assertEqual(list(summary["pipelines"]), ["baseline", "trustpcb"])
        self.assertEqual(list(summary["pipelines"]["baseline"]), ["total_seconds_per_repetition", "mean_seconds",
            "standard_deviation_seconds", "mean_ms_per_crop", "throughput_crops_per_second",
            "detector_passes_per_crop", "detector_passes_per_repetition"])
        self.assertEqual(summary["pipelines"]["baseline"]["mean_seconds"], 3)
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
             patch.object(benchmark.stability, "match_transform", wraps=benchmark.stability.match_transform) as matching, \
             patch.object(benchmark.stability.final.comparison, "apply_calibrator",
                          wraps=benchmark.stability.final.comparison.apply_calibrator) as calibrate, \
             patch.object(benchmark.fusion, "weighted_risk", wraps=benchmark.fusion.weighted_risk) as weighted:
            result = benchmark.full_crop(object(), Path("synthetic.png"), "synthetic", beta,
                                         pixel_loader=lambda _: np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(result, 12)
        self.assertEqual(predict.call_count, 12)
        self.assertEqual([c.args[1] for c in transform.call_args_list], list(benchmark.stability.SPECS))
        self.assertEqual(aggregate.call_count, 1)  # Below-inclusion prediction is not a reference.
        self.assertEqual(matching.call_count, 11)
        calibrate.assert_called_once_with(beta, [.5])
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
                with patch.object(benchmark.importlib.metadata, "version", return_value="wrong"), \
                     patch.object(benchmark, "adapter") as unused:
                    with self.assertRaisesRegex(RuntimeError, "Expected ultralytics"):
                        benchmark.run(root, dataset, adapter_factory=unused)
                    unused.assert_not_called()
                measure = benchmark.measure
                ticks = iter(range(20))
                with patch.object(benchmark, "measure", side_effect=lambda *args: measure(*args, clock=lambda: next(ticks))):
                    output = benchmark.run(root, dataset, adapter_factory=factory)
                with self.assertRaises(FileExistsError):
                    benchmark.run(root, dataset, adapter_factory=factory)
            self.assertEqual(sentinel.read_bytes(), b"immutable")
            provenance = benchmark.rq1._read_json(output/"provenance.json")
            self.assertEqual(provenance["status"], "complete")
            self.assertEqual(provenance["experiment"], "rq3_computational_efficiency")
            self.assertEqual(provenance["prediction_settings"], benchmark.raw.SETTINGS)
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

    def test_partial_directory_blocks_before_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(benchmark, "load_inputs") as load:
            (Path(tmp)/benchmark.OUTPUT).mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "partial/completed"):
                benchmark.run(tmp, tmp)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
