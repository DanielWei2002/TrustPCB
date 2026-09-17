"""RQ1 verification using configuration text and fabricated test artifacts only.

No YOLO object, PyTorch import, model.train call, or real dataset access.
"""

import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trustpcb import rq1
from trustpcb.dataset_config import ProjectPaths, generate_runtime_configs


class RQ1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for directory in ("configs/datasets", "configs/experiments", "data/splits", "src/trustpcb"):
            shutil.copytree(REPO / directory, self.root / directory,
                            ignore=shutil.ignore_patterns("__pycache__"))
        self.dataset = self.root / "empty dataset"
        self.dataset.mkdir()
        generate_runtime_configs(ProjectPaths(self.root.resolve(), self.dataset.resolve()))
        self.plans = rq1.build_plan(self.root, {"git_commit": "a" * 40, "git_dirty": False})
        self.model_file = self.root / "yolov8n.pt"
        self.model_file.write_bytes(b"synthetic pretrained input; not a real model")
        pretrained = rq1.bind_pretrained_model(self.root, "yolov8n.pt")
        for plan in self.plans:
            plan["pretrained_model"] = pretrained
        self.git_mock = patch.object(rq1, "git_provenance", return_value={
            "git_commit": "a" * 40, "git_dirty": False, "git_input_status": ""})
        self.git_mock.start()
        self.addCleanup(self.git_mock.stop)
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def fake_training(self, plan):
        """Write 100 tiny CSV rows; the checkpoint is an inert test text file."""
        output = Path(plan["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        (output / "weights").mkdir()
        (output / "weights/best.pt").write_text("synthetic fixture; not a model", encoding="utf-8")
        (output / "args.yaml").write_text(
            yaml.safe_dump({"model": plan["pretrained_model"]["path"], **plan["train_kwargs"]}), encoding="utf-8")
        with (output / "results.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", *rq1.METRICS.values()])
            writer.writeheader()
            offset = list(dict(rq1.SEEDS)).index(plan["seed_label"]) / 100
            for epoch in range(1, 101):
                # Tied maximum at epochs 37 and 38; first row must win.
                score = 0.8 + offset if epoch in (37, 38) else 0.2
                writer.writerow({"epoch": epoch, "metrics/precision(B)": 0.5,
                                 "metrics/recall(B)": 0.6, "metrics/mAP50(B)": 0.7,
                                 "metrics/mAP50-95(B)": score})

    def test_frozen_matrix_pair_equality_and_paths(self):
        expected = [("24209199", 24209199), ("20260902", 20260902),
                    ("02092026", 2092026), ("12345678", 12345678), ("20202027", 20202027)]
        self.assertEqual(list(rq1.SEEDS), expected)
        self.assertEqual(len(self.plans), 10)
        self.assertEqual([(p["seed_label"], p["split"]) for p in self.plans],
                         [(label, split) for label, _ in expected for split in rq1.SPLITS])
        self.assertEqual(len({p["output_dir"] for p in self.plans}), 10)
        for index, (label, value) in enumerate(expected):
            first, second = self.plans[2 * index:2 * index + 2]
            self.assertEqual(first["seed_value"], value)
            self.assertEqual(second["seed_value"], value)
            for plan in (first, second):
                self.assertEqual(Path(plan["output_dir"]).name, "seed_" + label)
                self.assertEqual(Path(plan["output_dir"]).relative_to(self.root).parts[:2], ("runs", "rq1"))
                self.assertEqual(plan["train_kwargs"]["seed"], value)
                self.assertFalse(plan["train_kwargs"]["exist_ok"])
                self.assertEqual(Path(plan["runtime_dataset_yaml"]).parent, self.root / "configs/local/datasets")
            strip = lambda p: {k: v for k, v in p["train_kwargs"].items() if k not in ("data", "project")}
            self.assertEqual(strip(first), strip(second))
            self.assertEqual(first["model"], second["model"])
        self.assertEqual(self.plans[4]["seed_label"], "02092026")
        self.assertIn("seed_02092026", self.plans[4]["experiment_id"])

    def test_selection_and_seed_label_validation(self):
        self.assertEqual(len(rq1.select_runs(self.plans, "24209199")), 2)
        self.assertEqual(len(rq1.select_runs(self.plans, "02092026", "supplied")), 1)
        self.assertEqual(rq1.select_runs(self.plans, remaining=True), self.plans[2:])
        for seed in ("2092026", "42"):
            with self.assertRaises(ValueError):
                rq1.select_runs(self.plans, seed)
        with self.assertRaises(ValueError):
            rq1.select_runs(self.plans, split="supplied")

    def test_configuration_drift_is_rejected(self):
        config = self.root / rq1.BASELINE_FILE
        value = yaml.safe_load(config.read_text())
        value["batch"] = 16
        config.write_text(yaml.safe_dump(value))
        with self.assertRaisesRegex(ValueError, "approved"):
            rq1.build_plan(self.root, {})

    def test_provenance_and_verified_skip(self):
        plan = self.plans[4]
        self.assertEqual(rq1.execute_one(plan, self.fake_training), "completed")
        def forbidden(_):
            self.fail("A completed run must never be invoked again")
        self.assertEqual(rq1.execute_one(plan, forbidden), "skipped")
        status, record, _ = rq1.inspect_run(plan)
        self.assertEqual(status, "completed")
        self.assertEqual(record["plan"]["git_commit"], "a" * 40)
        self.assertEqual(record["plan"]["seed_label"], "02092026")
        self.assertEqual(record["artifacts"]["metrics"]["best_epoch"], 37)
        self.assertIn("packages", record["environment"])
        self.assertEqual(len(record["runtime_hashes"]), 3)

    def test_existing_ambiguous_output_blocks_without_writes(self):
        plan = self.plans[0]
        output = Path(plan["output_dir"])
        output.mkdir(parents=True)
        sentinel = output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(RuntimeError, "STOP"):
            rq1.execute_one(plan, lambda _: self.fail("must not run"))
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertFalse(rq1._sidecar(plan).exists())

    def test_failure_is_incomplete_and_not_retried(self):
        plan = self.plans[0]
        def failing(_):
            raise RuntimeError("synthetic failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            rq1.execute_one(plan, failing)
        with self.assertRaisesRegex(RuntimeError, "STOP"):
            rq1.execute_one(plan, lambda _: self.fail("must not retry"))
        rows, aggregates = rq1.extract_results(self.root, self.plans)
        self.assertEqual(rows[0]["status"], "incomplete")
        self.assertEqual(rows[1]["status"], "not_started")
        for row in rows:
            self.assertEqual(row["best_epoch"], "")
            self.assertTrue(all(row[m] == "" for m in rq1.METRICS))
        self.assertTrue(all(a["mean"] == a["std"] == "" for a in aggregates))

    def test_completion_rejects_tampering(self):
        plan = self.plans[0]
        rq1.execute_one(plan, self.fake_training)
        results = Path(plan["output_dir"]) / "results.csv"
        results.write_text(results.read_text().replace("0.5", "0.51"))
        self.assertEqual(rq1.inspect_run(plan)[0], "incomplete")
        with self.assertRaises(RuntimeError):
            rq1.execute_one(plan, lambda _: self.fail("must not overwrite"))

    def test_changed_runtime_and_commit_block_reuse(self):
        plan = self.plans[0]
        rq1.execute_one(plan, self.fake_training)
        changed = {**plan, "git_commit": "b" * 40}
        self.assertEqual(rq1.inspect_run(changed)[0], "incomplete")
        data = Path(plan["runtime_dataset_yaml"])
        data.write_text(data.read_text() + "\n# changed\n")
        self.assertEqual(rq1.inspect_run(plan)[0], "incomplete")

    def test_partial_results_do_not_create_final_aggregates(self):
        rq1.execute_one(self.plans[0], self.fake_training)
        rows, aggregates = rq1.extract_results(self.root, self.plans)
        self.assertEqual(sum(r["status"] == "completed" for r in rows), 1)
        self.assertTrue(all(a["status"] == "incomplete" and a["mean"] == "" for a in aggregates))

    def test_full_aggregation_and_csv_seed_labels(self):
        for plan in self.plans:
            rq1.execute_one(plan, self.fake_training)
        rows, aggregates = rq1.extract_results(self.root, self.plans)
        self.assertTrue(all(r["status"] == "completed" for r in rows))
        self.assertTrue(all(a["n_completed"] == 5 and a["status"] == "complete" for a in aggregates))
        score = next(a for a in aggregates if a["split"] == "supplied" and a["metric"] == "mAP@0.50:0.95")
        self.assertAlmostEqual(score["mean"], 0.82)
        self.assertAlmostEqual(score["std"], 0.0158113883008419)
        with (self.root / "results/rq1/per_seed_metrics.csv").open(newline="") as file:
            csv_rows = list(csv.DictReader(file))
        self.assertEqual(csv_rows[4]["seed_label"], "02092026")
        self.assertEqual(csv_rows[4]["seed_value"], "2092026")
        self.assertEqual(list(self.dataset.iterdir()), [])

    def test_sequential_launcher_sets_startup_seed_and_blocks(self):
        calls = []
        def launch(command, *, env, cwd, check):
            plan = self.plans[len(calls)]
            self.assertEqual(env["PYTHONHASHSEED"], str(plan["seed_value"]))
            self.assertEqual(command[command.index("--seed") + 1], plan["seed_label"])
            self.assertEqual(command[command.index("--split") + 1], plan["split"])
            self.assertEqual(cwd, str(self.root))
            self.assertTrue(check)
            calls.append(plan["experiment_id"])
            rq1.execute_one(plan, self.fake_training)
        rq1.run_sequential(self.root, self.plans, launch=launch)
        self.assertEqual(calls, [p["experiment_id"] for p in self.plans])
        def fail_launch(*args, **kwargs):
            self.fail("Verified completed runs should not launch a worker")
        rq1.run_sequential(self.root, self.plans, launch=fail_launch)

    def test_launcher_stops_after_worker_failure(self):
        calls = []
        def launch(command, **kwargs):
            calls.append(command)
            raise subprocess.CalledProcessError(1, command)
        with self.assertRaises(subprocess.CalledProcessError):
            rq1.run_sequential(self.root, self.plans, launch=launch)
        self.assertEqual(len(calls), 1)

    def test_truncated_or_nonfinite_metrics_rejected(self):
        plan = self.plans[0]
        self.fake_training(plan)
        results = Path(plan["output_dir"]) / "results.csv"
        original = results.read_text()
        for content in ("\n".join(original.splitlines()[:3]), original.replace("0.5", "nan")):
            results.write_text(content)
            with self.assertRaises(ValueError):
                rq1.read_metrics(results, 100)

    def test_cli_train_gate_precedes_local_settings_or_import(self):
        with patch.object(rq1, "load_paths", side_effect=AssertionError("must not load")):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(rq1.main(["--project-dir", str(self.root), "train", "--all"]), 1)
        self.assertNotIn("ultralytics", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_old_git_checks_ignore_generated_artifacts_and_detect_all_change_types(self):
        self.git_mock.stop()
        changes = {
            "runs/rq1/.runner.lock": "??", "runs/rq1/supplied/seed_24209199/results.csv": " M",
            "runs/rq1/pretrained_model.json": "??", "results/rq1/per_seed_metrics.csv": "??",
            "configs/local/paths.yaml": "??", "configs/local/datasets/dspcbsd_supplied.yaml": "??",
        }
        def git(command, **kwargs):
            self.assertEqual(kwargs["cwd"], str(self.root))
            self.assertNotIn("-C", command)
            self.assertNotIn("status", command)
            if command[1] in ("diff", "ls-files"):
                self.assertIn("-z", command)
                scopes = command[command.index("--") + 1:]
                self.assertEqual(tuple(scopes), rq1.EXECUTION_INPUTS)
                if command[1] == "ls-files":
                    self.assertEqual(command[1:command.index("--")],
                                     ["ls-files", "--others", "--exclude-standard", "-z"])
                    selected_status = "??"
                else:
                    self.assertIn("--name-only", command)
                    selected_status = "M " if "--cached" in command else " M"
                return "".join(name + "\0" for name, status in changes.items()
                               if status == selected_status and
                               any(name == scope or name.startswith(scope.rstrip('/') + '/')
                                   for scope in scopes))
            self.assertEqual(command, ["git", "rev-parse", "HEAD"])
            return "a" * 40 + "\n"
        with patch.object(rq1.subprocess, "check_output", side_effect=git):
            self.assertFalse(rq1.require_clean_inputs(self.root)["git_dirty"])
            for name in ("src/trustpcb/rq1.py", "src/trustpcb/dataset_config.py",
                         rq1.BASELINE_FILE, "configs/datasets/new.txt", "data/splits/new.txt"):
                for status in (" M", "M ", "??"):
                    with self.subTest(name=name, status=status):
                        changes[name] = status
                        with self.assertRaisesRegex(RuntimeError, "committed and clean"):
                            rq1.require_clean_inputs(self.root)
                        del changes[name]

    def test_git_command_failures_never_mean_clean(self):
        self.git_mock.stop()
        for failing_command in ("unstaged", "staged", "untracked"):
            def git(command, **kwargs):
                kind = ("untracked" if "ls-files" in command else
                        "staged" if "--cached" in command else "unstaged")
                if kind == failing_command:
                    raise subprocess.CalledProcessError(129, command)
                return ""
            with self.subTest(command=failing_command), patch.object(
                rq1.subprocess, "check_output", side_effect=git
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    rq1.require_clean_inputs(self.root)

    def test_notebooks_pretrained_file_resolves_with_hash_and_size(self):
        notebook_model = self.root / "notebooks/yolov8n.pt"
        notebook_model.parent.mkdir()
        self.model_file.replace(notebook_model)
        identity = rq1.model_provenance(self.root, "yolov8n.pt")
        self.assertEqual(identity, {
            "path": str(notebook_model.resolve()),
            "sha256": hashlib.sha256(notebook_model.read_bytes()).hexdigest(),
            "size_bytes": notebook_model.stat().st_size,
        })
        self.assertEqual(self.plans[0]["model"], "yolov8n.pt")

    def test_multiple_approved_weight_locations_block_even_with_identical_bytes(self):
        notebook_model = self.root / "notebooks/yolov8n.pt"
        notebook_model.parent.mkdir()
        notebook_model.write_bytes(self.model_file.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "Ambiguous pretrained model"):
            rq1.run_sequential(self.root, self.plans, launch=lambda *a, **k: self.fail("must not launch"))

    def test_unapproved_weight_location_is_not_searched(self):
        other = self.root / "other/yolov8n.pt"
        other.parent.mkdir()
        self.model_file.replace(other)
        with self.assertRaisesRegex(FileNotFoundError, "automatic download is disabled"):
            rq1.model_provenance(self.root, "yolov8n.pt")

    def test_dirty_inputs_block_launch_and_direct_execution(self):
        dirty = {"git_commit": "a" * 40, "git_dirty": True,
                 "git_input_status": "?? src/trustpcb/rq1.py"}
        with patch.object(rq1, "git_provenance", return_value=dirty):
            with self.assertRaisesRegex(RuntimeError, "committed and clean"):
                rq1.run_sequential(self.root, self.plans, launch=lambda *a, **k: self.fail("must not launch"))
            with self.assertRaisesRegex(RuntimeError, "committed and clean"):
                rq1.execute_one(self.plans[0], lambda _: self.fail("must not train"))
        self.assertFalse(rq1._sidecar(self.plans[0]).exists())

    def test_missing_pretrained_model_blocks_first_worker(self):
        self.model_file.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "automatic download is disabled"):
            rq1.run_sequential(self.root, self.plans, launch=lambda *a, **k: self.fail("must not launch"))
        self.assertFalse(rq1._sidecar(self.plans[0]).exists())

    def test_pretrained_identity_recorded_in_runtime_provenance(self):
        rq1.execute_one(self.plans[0], self.fake_training)
        record = json.loads(rq1._sidecar(self.plans[0]).read_text())
        identity = record["plan"]["pretrained_model"]
        self.assertEqual(identity, {
            "path": str(self.model_file.resolve()),
            "sha256": hashlib.sha256(self.model_file.read_bytes()).hexdigest(),
            "size_bytes": self.model_file.stat().st_size,
        })

    def test_changed_model_blocks_reuse_and_remaining_pairs(self):
        rq1.execute_one(self.plans[0], self.fake_training)
        original = self.model_file.read_bytes()
        self.model_file.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        self.assertEqual(self.model_file.stat().st_size, len(original))
        with self.assertRaisesRegex(RuntimeError, "model file/hash changed"):
            rq1.execute_one(self.plans[0], lambda _: self.fail("must not reuse"))
        # A later invocation selecting only remaining seeds cannot adopt new weights.
        new_plans = rq1.build_plan(self.root, {"git_commit": "a" * 40, "git_dirty": False})
        with self.assertRaisesRegex(RuntimeError, "frozen RQ1 model"):
            rq1.run_sequential(self.root, rq1.select_runs(new_plans, remaining=True),
                               launch=lambda *a, **k: self.fail("must not launch"))

    def test_model_changed_during_run_is_incomplete(self):
        def changed_model(plan):
            self.fake_training(plan)
            self.model_file.write_bytes(b"changed during synthetic execution")
        with self.assertRaisesRegex(RuntimeError, "model file/hash changed"):
            rq1.execute_one(self.plans[0], changed_model)
        record = json.loads(rq1._sidecar(self.plans[0]).read_text())
        self.assertEqual(record["status"], "incomplete")
        self.assertFalse((Path(self.plans[0]["output_dir"]) / "rq1_complete.json").exists())

    def test_parent_checks_model_again_after_worker(self):
        def launch(command, **kwargs):
            rq1.execute_one(self.plans[0], self.fake_training)
            self.model_file.write_bytes(b"changed after the worker returned")
        with self.assertRaisesRegex(RuntimeError, "model file/hash changed"):
            rq1.run_sequential(self.root, self.plans, launch=launch)
        self.assertFalse(rq1._sidecar(self.plans[1]).exists())

    def test_missing_binding_cannot_adopt_new_weights_for_old_artifacts(self):
        rq1.execute_one(self.plans[0], self.fake_training)
        (self.root / "runs/rq1/pretrained_model.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "binding is missing"):
            rq1.run_sequential(self.root, self.plans, launch=lambda *a, **k: self.fail("must not launch"))


if __name__ == "__main__":
    unittest.main()
