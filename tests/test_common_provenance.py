"""Synthetic infrastructure checks; no research artifacts or ML imports."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustpcb import rq1
from trustpcb.common import provenance


class CommonProvenanceTests(unittest.TestCase):
    def test_legacy_serialization_interfaces_are_canonical_aliases(self):
        for old, new in (("_json", "serialize_json"), ("_sha", "sha256_file"),
                         ("_write_json", "write_json"), ("_read_json", "read_json")):
            self.assertIs(getattr(rq1, old), getattr(provenance, new))

    def test_exact_serialization_and_nonfinite_rejection(self):
        self.assertEqual(provenance.serialize_json({"z": "é", "a": [1, True]}),
                         '{\n  "a": [\n    1,\n    true\n  ],\n  "z": "\\u00e9"\n}\n')
        with self.assertRaises(ValueError):
            provenance.serialize_json({"value": float("nan")})

    def test_write_modes_hash_bytes_and_object_requirement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "record.json"
            provenance.write_json(path, {"value": 1}, exclusive=True)
            original = path.read_bytes()
            self.assertEqual(original, provenance.serialize_json({"value": 1}).encode())
            with self.assertRaises(FileExistsError):
                provenance.write_json(path, {"value": 2}, exclusive=True)
            self.assertEqual(path.read_bytes(), original)
            provenance.write_json(path, {"value": 2})
            self.assertEqual(provenance.read_json(path), {"value": 2})
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            self.assertEqual(provenance.sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Expected an object"):
                provenance.read_json(path)
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                provenance.read_json(path)

    def test_git_scope_commands_and_change_order(self):
        scope = ("synthetic inputs/", "protocol.yaml")
        responses = ["synthetic inputs/a\nb.py\0", "protocol.yaml\0", "synthetic inputs/new.py\0", "f" * 40 + "\n"]
        with patch.object(provenance.subprocess, "check_output", side_effect=responses) as git:
            info = provenance.git_provenance(Path("synthetic-root"), scope)
        self.assertEqual(info, {"git_commit": "f" * 40, "git_dirty": True,
                               "git_input_status": "unstaged: 'synthetic inputs/a\\nb.py'\nstaged: 'protocol.yaml'\nuntracked: 'synthetic inputs/new.py'"})
        for call, command in zip(git.call_args_list[:3],
                                 (("diff", "--name-only", "-z"),
                                  ("diff", "--cached", "--name-only", "-z"),
                                  ("ls-files", "--others", "--exclude-standard", "-z"))):
            self.assertEqual(call.args[0], ["git", *command, "--", *scope])
            self.assertEqual(call.kwargs, {"cwd": "synthetic-root", "text": True, "stderr": subprocess.PIPE})
        self.assertEqual(git.call_args_list[-1].args[0], ["git", "rev-parse", "HEAD"])

    def test_git_failure_propagates_and_legacy_scope_remains_dynamic(self):
        with patch.object(provenance.subprocess, "check_output", side_effect=subprocess.CalledProcessError(129, "git")):
            with self.assertRaises(subprocess.CalledProcessError):
                provenance.git_provenance("root", ("inputs/",))
        with patch.object(rq1, "EXECUTION_INPUTS", ("synthetic/",)), \
             patch.object(provenance, "git_provenance", return_value={"git_dirty": False}) as collect:
            self.assertEqual(rq1.git_provenance("root"), {"git_dirty": False})
            collect.assert_called_once_with("root", ("synthetic/",))

    def test_environment_schema_missing_version_and_legacy_inventory(self):
        def version(name):
            if name == "missing":
                raise importlib.metadata.PackageNotFoundError(name)
            return "synthetic-version"

        with patch.object(provenance.importlib.metadata, "version", side_effect=version), \
             patch.object(provenance.platform, "platform", return_value="synthetic-platform"), \
             patch.dict(os.environ, {"PYTHONHASHSEED": "7", "CUDA_VISIBLE_DEVICES": "0"}):
            info = provenance.environment(("present", "missing"))
        self.assertEqual(info, {"python": sys.version, "platform": "synthetic-platform",
                               "packages": {"present": "synthetic-version", "missing": "not installed"},
                               "PYTHONHASHSEED": "7", "CUDA_VISIBLE_DEVICES": "0"})
        with patch.object(provenance, "environment", return_value=info) as collect:
            self.assertEqual(rq1._environment(), info)
            collect.assert_called_once_with(("ultralytics", "torch", "numpy", "PyYAML"))


if __name__ == "__main__":
    unittest.main()
