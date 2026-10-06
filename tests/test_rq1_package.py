"""Package compatibility and source identity; synthetic fixtures only."""

import importlib
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import trustpcb.rq1
from trustpcb import rq1
from trustpcb.rq1 import SEEDS, _sha, build_plan
from trustpcb.rq1 import workflow
from trustpcb.common import provenance


class RQ1PackageTests(unittest.TestCase):
    def test_explicit_exports_and_direct_imports(self):
        self.assertIs(trustpcb.rq1, rq1)
        self.assertTrue(hasattr(rq1, "__path__"))
        self.assertFalse((REPO / "src/trustpcb/rq1.py").exists())
        self.assertIs(SEEDS, workflow.SEEDS)
        self.assertIs(build_plan, workflow.build_plan)
        self.assertIs(_sha, provenance.sha256_file)
        for name in rq1.__all__:
            with self.subTest(name=name):
                self.assertIs(getattr(rq1, name), getattr(workflow, name))

    def test_package_patches_reach_workflow_and_restore(self):
        original = workflow.git_provenance
        with patch.object(rq1, "git_provenance", return_value={"git_dirty": False}) as collect:
            self.assertIs(workflow.git_provenance, collect)
            self.assertEqual(rq1.require_clean_inputs("synthetic"), {"git_dirty": False})
            collect.assert_called_once_with("synthetic")
        self.assertIs(rq1.git_provenance, original)
        self.assertIs(workflow.git_provenance, original)
        with patch.object(workflow, "git_provenance", return_value={"git_dirty": True, "git_input_status": "synthetic"}):
            with self.assertRaisesRegex(RuntimeError, "committed and clean"):
                rq1.require_clean_inputs("synthetic")
        self.assertIs(rq1.git_provenance, original)

    def test_entry_point_delegates_and_propagates_exit_code(self):
        with patch.object(rq1, "main", return_value=7) as main:
            with self.assertRaises(SystemExit) as result:
                runpy.run_module("trustpcb.rq1", run_name="__main__")
        self.assertEqual(result.exception.code, 7)
        main.assert_called_once_with()

    def test_real_cli_help_without_ml_imports_or_execution(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        result = subprocess.run([sys.executable, "-B", "-m", "trustpcb.rq1", "--help"],
                                cwd=REPO, env=environment, check=True, capture_output=True, text=True)
        self.assertIn("{plan,train,_worker,extract}", result.stdout)
        self.assertIn("--project-dir", result.stdout)
        self.assertEqual(result.stderr, "")
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("ultralytics", sys.modules)

    def test_downstream_modules_keep_same_facade(self):
        for path in sorted((REPO / "src/trustpcb").glob("rq[23]_*.py")):
            module = importlib.import_module("trustpcb." + path.stem)
            if hasattr(module, "rq1"):
                with self.subTest(module=module.__name__):
                    self.assertIs(module.rq1, rq1)
                    self.assertIs(module.rq1._sha, provenance.sha256_file)

    def test_future_plan_hashes_real_package_files_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (*rq1.SOURCE_FILES, rq1.BASELINE_FILE, "src/trustpcb/dataset_config.py"):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / relative, target)
            info = {"git_commit": "synthetic", "git_dirty": False}
            hashes = rq1.build_plan(root, info)[0]["input_hashes"]
            self.assertNotIn("src/trustpcb/rq1.py", hashes)
            for relative in rq1.SOURCE_FILES:
                with self.subTest(relative=relative):
                    self.assertEqual(hashes[relative], rq1._sha(root / relative))
                    target = root / relative
                    original = target.read_bytes()
                    target.write_bytes(original + b"\n# synthetic change\n")
                    self.assertNotEqual(rq1.build_plan(root, info)[0]["input_hashes"][relative], hashes[relative])
                    target.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
