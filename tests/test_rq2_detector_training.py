"""Text-only Detector training tests. Checkpoints are tiny inert byte strings; no ML imports."""

import csv
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trustpcb import rq1, rq2_detector_training as detector


class DetectorTrainingCompatibilityTests(unittest.TestCase):
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
first = importlib.import_module({first!r})
second = importlib.import_module({second!r})
from trustpcb import rq2_detector_training as legacy
from trustpcb.rq2 import detector_training as canonical
from trustpcb import rq2_confidence_distribution as raw
from trustpcb import rq2_calibration_analysis as matching
from trustpcb import rq2_final_evaluation as final
from trustpcb import rq3_computational_efficiency as efficiency
assert first is second is legacy is canonical
assert sys.modules["trustpcb.rq2_detector_training"] is canonical
assert sys.modules["trustpcb.rq2.detector_training"] is canonical
assert Path(canonical.__file__).resolve() == Path("src/trustpcb/rq2/detector_training.py").resolve()
for consumer in (raw.rq2_detector_training, matching.rq2_detector_training,
                 final.raw.rq2_detector_training, efficiency.raw.rq2_detector_training):
    assert consumer is canonical
for name in ("MANIFESTS", "NAMES", "_argument_matches", "file_identity", "CheckpointSelector",
             "METRIC", "TEMPLATE", "build_plan", "execute", "frozen_inputs", "guard_before_fit",
             "prepare_runtime", "selected_epoch", "find_project_root", "main"):
    assert getattr(legacy, name) is getattr(canonical, name)
assert canonical.main.__globals__ is canonical.__dict__
assert canonical.build_plan.__globals__ is canonical.__dict__
for owner, observer in ((legacy, canonical), (canonical, legacy)):
    with patch.object(owner, "NAMES", {{0: "synthetic"}}):
        owner.NAMES[1] = "mutated"
        assert observer.NAMES is raw.rq2_detector_training.NAMES
        assert matching.rq2_detector_training.NAMES[1] == "mutated"
    with patch.object(owner, "MANIFESTS", {{"val": "synthetic"}}):
        assert final.raw.rq2_detector_training.MANIFESTS is observer.MANIFESTS
    with patch.object(owner, "_argument_matches", return_value="synthetic") as mocked:
        assert efficiency.raw.rq2_detector_training._argument_matches("device", "0", 0) == "synthetic"
        mocked.assert_called_once_with("device", "0", 0)
        assert observer._argument_matches is mocked
assert "torch" not in sys.modules
assert "ultralytics" not in sys.modules
'''])

    def test_legacy_first_canonical_second(self):
        self.check_import_order("trustpcb.rq2_detector_training", "trustpcb.rq2.detector_training")

    def test_canonical_first_legacy_second(self):
        self.check_import_order("trustpcb.rq2.detector_training", "trustpcb.rq2_detector_training")

    def check_help(self, module):
        output = self.run_python(["-m", module, "--help"])
        self.assertIn("{plan,train,_worker}", output)
        self.assertIn("--dicc", output)

    def test_legacy_cli_help(self):
        self.check_help("trustpcb.rq2_detector_training")

    def test_canonical_cli_help(self):
        self.check_help("trustpcb.rq2.detector_training")

    def test_legacy_main_dispatch(self):
        self.run_python(["-c", '''
import runpy
import sys
from unittest.mock import patch
from trustpcb.rq2 import detector_training as canonical
original_main = sys.modules["__main__"]
with patch.object(canonical, "main") as main:
    runpy.run_path("src/trustpcb/rq2_detector_training.py", run_name="__main__")
    main.assert_called_once_with()
assert sys.modules["__main__"] is original_main
assert "torch" not in sys.modules
assert "ultralytics" not in sys.modules
'''])


class DetectorTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for relative in (detector.TEMPLATE, rq1.BASELINE_FILE, *detector.SOURCE_FILES,
                         *rq1.SOURCE_FILES, "src/trustpcb/dataset_config.py",
                         *(spec[0] for spec in detector.MANIFESTS.values())):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, target)
        self.info = {"git_commit": "a" * 40, "git_dirty": False, "git_input_status": ""}
        self.git = patch.object(rq1, "git_provenance", return_value=self.info)
        self.git_mock = self.git.start()
        self.addCleanup(self.git.stop)
        self.plan = detector.build_plan(self.root)
        dataset = self.root / "empty synthetic dataset"
        dataset.mkdir()
        local = self.root / "configs/local/paths.yaml"
        local.parent.mkdir(parents=True)
        local.write_text(yaml.safe_dump({"project_root": str(self.root), "dataset_root": str(dataset)}))
        self.plan["dataset_root"] = str(dataset)
        self.plan["runtime_hashes"] = detector.prepare_runtime(self.plan, dataset)
        model = self.root / "notebooks/yolov8n.pt"
        model.parent.mkdir()
        model.write_bytes(b"synthetic original weights")
        self.plan["pretrained_model"] = rq1.bind_pretrained_model(self.root, "yolov8n.pt")

    def write_csv(self, output, scores):
        with (output / "results.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("epoch", detector.METRIC))
            writer.writerows(enumerate(scores, 1))

    def fake_training(self, plan, selector):
        output = Path(plan["output_dir"])
        (output / "weights").mkdir(parents=True)
        (output / "args.yaml").write_text(yaml.safe_dump(
            {"model": plan["pretrained_model"]["path"], **plan["train_kwargs"]}))
        scores = []
        for epoch in range(1, 101):
            scores.append("0.8" if epoch in (37, 38) else "0.2")
            self.write_csv(output, scores)
            last = output / "weights/last.pt"
            last.write_bytes(f"synthetic epoch {epoch}".encode())
            selector.on_model_save(SimpleNamespace(save_dir=output, last=last, epoch=epoch - 1))

    def test_frozen_config_and_manifest_wiring(self):
        self.assertEqual(self.plan["baseline_config"], rq1.APPROVED_BASELINE)
        self.assertNotIn("src/trustpcb/rq1.py", self.plan["input_hashes"])
        for relative in rq1.SOURCE_FILES:
            self.assertEqual(self.plan["input_hashes"][relative], rq1._sha(self.root / relative))
        lists = detector.frozen_inputs(self.root)
        self.assertEqual([len(lists[k]) for k in ("train", "val")], [7386, 821])
        self.assertFalse(set(lists["train"]) & set(lists["val"]))
        config = yaml.safe_load(Path(self.plan["runtime_dataset_yaml"]).read_text())
        self.assertEqual(set(config), {"train", "val", "names"})
        for key in ("train", "val"):
            actual = Path(config[key]).read_text().splitlines()
            relative = [Path(p).relative_to(self.plan["dataset_root"]).as_posix() for p in actual]
            self.assertEqual(relative, lists[key])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("ultralytics", sys.modules)

    def test_test_set_template_redirect_rejected(self):
        path = self.root / detector.TEMPLATE
        config = yaml.safe_load(path.read_text())
        config["val"] = "configs/datasets/similarity_aware_val_images.txt"
        path.write_text(yaml.safe_dump(config))
        with self.assertRaises(ValueError):
            detector.build_plan(self.root)

    def test_future_source_inventory(self):
        expected_sources = (
            "src/trustpcb/rq2/detector_training.py",
            "src/trustpcb/rq2_detector_training.py",
            "src/trustpcb/rq2/__init__.py",
        )
        self.assertEqual(detector.SOURCE_FILES, expected_sources)
        expected_inputs = {detector.TEMPLATE, rq1.BASELINE_FILE, *expected_sources,
                           *rq1.SOURCE_FILES, "src/trustpcb/dataset_config.py"}
        self.assertEqual(set(self.plan["input_hashes"]), expected_inputs)
        for relative in expected_inputs:
            self.assertEqual(self.plan["input_hashes"][relative], rq1._sha(self.root / relative))
        self.assertNotEqual(self.plan["input_hashes"][expected_sources[0]],
                            self.plan["input_hashes"][expected_sources[1]])

    def test_changed_migration_sources_block_execution(self):
        for relative in detector.SOURCE_FILES:
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"\n# synthetic source change\n")
                    with self.assertRaisesRegex(RuntimeError, "configuration/Git state changed"):
                        detector.execute(self.plan, lambda *_: self.fail("must block"))
                finally:
                    path.write_bytes(original)

    def test_pre_migration_source_inventory_blocks_reuse(self):
        historical_plan = json.loads(json.dumps(self.plan))
        historical_plan["input_hashes"].pop("src/trustpcb/rq2/detector_training.py")
        historical_plan["input_hashes"].pop("src/trustpcb/rq2/__init__.py")
        historical_plan["input_hashes"]["src/trustpcb/rq2_detector_training.py"] = "historical-source-fixture"
        with self.assertRaisesRegex(RuntimeError, "configuration/Git state changed"):
            detector.execute(historical_plan, lambda *_: self.fail("must block"))
        output = Path(self.plan["output_dir"])
        sidecar = output.with_name(output.name + ".provenance.json")
        rq1._write_json(sidecar, {"status": "complete", "plan": historical_plan})
        before = sidecar.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "incomplete or different"):
            detector.execute(self.plan, lambda *_: self.fail("must not overwrite historical run"))
        self.assertEqual(sidecar.read_bytes(), before)
        self.assertFalse(output.exists())

    def test_plan_cli_uses_synthetic_root_without_execution(self):
        output = io.StringIO()
        with patch.object(detector, "find_project_root", return_value=self.root), \
                patch.object(detector, "execute") as execute, \
                patch.object(detector.subprocess, "run") as worker, \
                contextlib.redirect_stdout(output):
            detector.main(["plan"])
        self.assertEqual(json.loads(output.getvalue()), detector.build_plan(self.root))
        execute.assert_not_called()
        worker.assert_not_called()

    def test_worker_parsing_and_dispatch_without_training(self):
        environment = {"TRUSTPCB_RQ2_WORKER": "1", "PYTHONHASHSEED": "24209199",
                       "TRUSTPCB_RQ2_PLAN": json.dumps(self.plan)}
        with patch.object(detector, "find_project_root", return_value=self.root), \
                patch.object(detector, "os", SimpleNamespace(name="posix", environ=environment)), \
                patch.object(detector, "execute") as execute, \
                patch.object(detector, "_train", side_effect=AssertionError("no training")) as train:
            detector.main(["_worker", "--dicc"])
            execute.assert_called_once_with(self.plan, train)
            train.assert_not_called()

    def test_worker_rejects_missing_authorization_or_wrong_seed(self):
        for args, environment, message in (
                (["_worker"], {}, "DICC-only"),
                (["_worker", "--dicc"], {}, "guarded train launcher"),
                (["_worker", "--dicc"], {"TRUSTPCB_RQ2_WORKER": "1", "PYTHONHASHSEED": "1"},
                 "guarded train launcher")):
            with self.subTest(args=args, environment=environment), \
                    patch.object(detector, "find_project_root", return_value=self.root), \
                    patch.object(detector, "os", SimpleNamespace(name="posix", environ=environment)), \
                    patch.object(detector, "execute") as execute:
                with self.assertRaisesRegex(RuntimeError, message):
                    detector.main(args)
                execute.assert_not_called()

    def test_manifest_mutation_and_configuration_drift_rejected(self):
        path = self.root / detector.MANIFESTS["train"][0]
        original = path.read_bytes()
        path.write_bytes(original + b"images/train/extra.jpg\n")
        with self.assertRaises(ValueError):
            detector.build_plan(self.root)
        path.write_bytes(original)
        baseline = self.root / rq1.BASELINE_FILE
        config = yaml.safe_load(baseline.read_text())
        config["seed"] = 1
        baseline.write_text(yaml.safe_dump(config))
        with self.assertRaises(ValueError):
            detector.build_plan(self.root)

    def test_lf_crlf_manifest_identity(self):
        before = detector.frozen_inputs(self.root)
        for relative, _, _ in detector.MANIFESTS.values():
            path = self.root / relative
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.assertEqual(detector.frozen_inputs(self.root), before)

    def test_earliest_exact_tie_and_decimal_precision(self):
        self.write_csv(self.root, ["0.2", "0.800", "0.8"])
        self.assertEqual(detector.selected_epoch(self.root / "results.csv", 3)["epoch"], 2)
        self.write_csv(self.root, ["0.80000000000000000001", "0.80000000000000000002"])
        self.assertEqual(detector.selected_epoch(self.root / "results.csv", 2)["epoch"], 2)

    def test_invalid_or_incomplete_csv_rejected(self):
        for scores in (["NaN"], ["Infinity"], ["1.01"], ["-0.1"], ["bad"]):
            self.write_csv(self.root, scores)
            with self.assertRaises(ValueError):
                detector.selected_epoch(self.root / "results.csv", 1)
        self.write_csv(self.root, ["0.5"])
        with self.assertRaises(ValueError):
            detector.selected_epoch(self.root / "results.csv", 100)
        (self.root / "results.csv").write_text(f"epoch,{detector.METRIC}\n2,0.5\n")
        with self.assertRaises(ValueError):
            detector.selected_epoch(self.root / "results.csv", 1)

    def test_exact_epoch_checkpoint_freeze_and_completed_reuse(self):
        receipt = detector.execute(self.plan, self.fake_training)
        self.assertEqual(receipt["selected"]["epoch"], 37)
        self.assertEqual(Path(receipt["selected"]["checkpoint"]["path"]).read_bytes(), b"synthetic epoch 37")
        self.assertEqual(receipt["plan"]["pretrained_model"]["size_bytes"], 26)
        self.assertIn("environment", receipt)
        self.assertEqual(detector.execute(self.plan, lambda *_: self.fail("must reuse")), receipt)
        Path(receipt["selected"]["checkpoint"]["path"]).write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            detector.execute(self.plan, lambda *_: self.fail("must block"))

    def test_missing_or_changed_original_weights_block(self):
        binding = self.root / "runs/rq1/pretrained_model.json"
        original = binding.read_bytes()
        binding.unlink()
        with self.assertRaises(FileNotFoundError):
            detector.execute(self.plan, lambda *_: self.fail("must block"))
        binding.write_bytes(original)
        Path(self.plan["pretrained_model"]["path"]).write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            detector.execute(self.plan, lambda *_: self.fail("must block"))

    def test_weights_changed_after_completion_block_reuse(self):
        detector.execute(self.plan, self.fake_training)
        Path(self.plan["pretrained_model"]["path"]).write_bytes(b"changed original")
        with self.assertRaises(RuntimeError):
            detector.execute(self.plan, lambda *_: self.fail("must block"))

    def test_candidate_tampering_blocks_completion(self):
        def tamper(plan, selector):
            self.fake_training(plan, selector)
            selector.candidate.write_bytes(b"changed candidate")
        with self.assertRaises(RuntimeError):
            detector.execute(self.plan, tamper)

    def test_dicc_launcher_startup_seed_and_parent_verification(self):
        def fake_worker(command, *, cwd, env, check):
            self.assertEqual(env["PYTHONHASHSEED"], "24209199")
            self.assertEqual(env["TRUSTPCB_RQ2_WORKER"], "1")
            self.assertEqual(command[-2:], ["_worker", "--dicc"])
            self.assertEqual(command, [sys.executable, "-B", "-m",
                                      "trustpcb.rq2.detector_training", "_worker", "--dicc"])
            self.assertEqual(cwd, self.root)
            self.assertTrue(check)
            self.assertEqual(env["PYTHONPATH"], str(self.root / "src"))
            detector.execute(json.loads(env["TRUSTPCB_RQ2_PLAN"]), self.fake_training)
        with patch.object(detector, "find_project_root", return_value=self.root), \
                patch.object(detector, "os", SimpleNamespace(
                    name="posix", environ=os.environ, getpid=os.getpid, fstat=os.fstat)), \
                patch.object(detector.subprocess, "run", side_effect=fake_worker), \
                contextlib.redirect_stdout(io.StringIO()):
            detector.main(["train", "--dicc"])
        self.assertFalse((Path(self.plan["output_dir"]).parent / ".runner.lock").exists())

    def test_runtime_redirect_and_dirty_inputs_block(self):
        self.git_mock.return_value = {**self.info, "git_dirty": True}
        with self.assertRaises(RuntimeError):
            detector.execute(self.plan, lambda *_: self.fail("must block"))
        self.git_mock.return_value = self.info
        Path(self.plan["runtime_dataset_yaml"]).write_text("val: forbidden-test.txt")
        with self.assertRaises(RuntimeError):
            detector.execute(self.plan, lambda *_: self.fail("must block"))
        with self.assertRaises(RuntimeError):
            detector.prepare_runtime(self.plan, self.plan["dataset_root"])

    def test_incomplete_run_cannot_restart(self):
        def interrupted(*_):
            raise RuntimeError("synthetic interruption")
        with self.assertRaises(RuntimeError):
            detector.execute(self.plan, interrupted)
        with self.assertRaises(RuntimeError):
            detector.execute(self.plan, lambda *_: self.fail("must block"))

    def test_existing_output_cannot_overwrite(self):
        Path(self.plan["output_dir"]).mkdir(parents=True)
        with self.assertRaises(RuntimeError):
            detector.execute(self.plan, lambda *_: self.fail("must block"))

    def test_args_only_prefit_failure_requires_manual_archive(self):
        output = Path(self.plan["output_dir"])
        output.mkdir(parents=True)
        args = output / "args.yaml"
        args.write_bytes(b"model: synthetic-prefit-fixture\n")
        sidecar = output.with_name(output.name + ".provenance.json")
        for has_sidecar in (False, True):
            with self.subTest(has_sidecar=has_sidecar):
                if has_sidecar:
                    rq1._write_json(sidecar, {"status": "incomplete", "plan": self.plan})
                before = {p: p.read_bytes() for p in output.parent.rglob("*") if p.is_file()}
                with self.assertRaisesRegex(RuntimeError, "args.yaml-only"):
                    detector.execute(self.plan, lambda *_: self.fail("must not launch training"))
                after = {p: p.read_bytes() for p in output.parent.rglob("*") if p.is_file()}
                self.assertEqual(after, before)
                self.assertEqual(list(output.iterdir()), [args])

    def test_missing_callbacks_fail_closed(self):
        selector = detector.CheckpointSelector(self.root)
        self.write_csv(self.root, ["0.5", "0.6"])
        with self.assertRaises(RuntimeError):
            selector.on_model_save(SimpleNamespace(save_dir=self.root, epoch=1))
        with self.assertRaises(RuntimeError):
            selector.freeze(2)

    def test_local_training_prohibited(self):
        with patch.object(detector, "find_project_root", return_value=self.root), patch.object(os, "name", "nt"):
            with self.assertRaises(RuntimeError):
                detector.main(["train", "--dicc"])

    def test_ultralytics_84117_string_device_passes_prefit_guard(self):
        expected = self.plan["train_kwargs"]["device"]
        self.assertIs(type(expected), int)
        self.assertEqual(expected, 0)
        args = {"model": self.plan["pretrained_model"]["path"], **self.plan["train_kwargs"]}
        # BaseTrainer stores parse_device(0) == '0' before this callback.
        args["device"] = "0"
        self.assertIs(type(args["device"]), str)
        self.assertNotEqual(args["device"], expected)  # reproduces the old failure
        trainer = SimpleNamespace(save_dir=self.plan["output_dir"], args=SimpleNamespace(**args))
        detector.guard_before_fit(self.plan, trainer)
        self.assertEqual(self.plan["train_kwargs"]["device"], 0)

    def test_device_guard_rejects_other_devices_and_types(self):
        args = {"model": self.plan["pretrained_model"]["path"], **self.plan["train_kwargs"]}
        for device in (1, "1", "cpu", "mps", "", None, -1, "-1", "0,1", [0], False, 0.0, "cuda:0"):
            with self.subTest(device=device):
                trainer = SimpleNamespace(save_dir=self.plan["output_dir"],
                                          args=SimpleNamespace(**{**args, "device": device}))
                with self.assertRaisesRegex(RuntimeError, "before fitting: device"):
                    detector.guard_before_fit(self.plan, trainer)
        self.assertFalse(detector._argument_matches("device", "0", 1))
        self.assertFalse(detector._argument_matches("device", "0", "0"))

    def test_device_normalization_does_not_relax_other_arguments(self):
        args = {"model": self.plan["pretrained_model"]["path"], **self.plan["train_kwargs"], "device": "0"}
        for key, changed in (("seed", 1), ("batch", "32"), ("epochs", 99),
                             ("data", "forbidden-test.yaml"), ("resume", True)):
            with self.subTest(key=key):
                trainer = SimpleNamespace(save_dir=self.plan["output_dir"],
                                          args=SimpleNamespace(**{**args, key: changed}))
                with self.assertRaisesRegex(RuntimeError, f"before fitting: {key}"):
                    detector.guard_before_fit(self.plan, trainer)

    def test_serialized_string_device_passes_completion_and_reuse(self):
        def serialized_training(plan, selector):
            self.fake_training(plan, selector)
            path = Path(plan["output_dir"]) / "args.yaml"
            args = yaml.safe_load(path.read_text())
            args["device"] = "0"  # BaseTrainer serializes the canonical string.
            path.write_text(yaml.safe_dump(args))
        receipt = detector.execute(self.plan, serialized_training)
        self.assertEqual(receipt["actual_training_config"]["device"], "0")
        self.assertIs(type(receipt["plan"]["train_kwargs"]["device"]), int)
        self.assertEqual(detector.execute(self.plan, lambda *_: self.fail("must reuse")), receipt)


if __name__ == "__main__":
    unittest.main()
