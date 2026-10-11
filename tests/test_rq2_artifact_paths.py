"""Synthetic artifact-location compatibility; no scientific execution or model loads."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trustpcb import rq2_artifact_paths as paths
from trustpcb import rq2_confidence_distribution as confidence
from trustpcb import rq2_detector_training as detector


class ArtifactPathImportTests(unittest.TestCase):
    def check_import_order(self, first, second):
        script = f'''
import importlib
import sys
from unittest.mock import patch

first = importlib.import_module({first!r})
second = importlib.import_module({second!r})
from trustpcb import rq2_artifact_paths as legacy
from trustpcb.rq2 import artifact_paths as canonical
assert first is second is legacy is canonical
assert sys.modules["trustpcb.rq2_artifact_paths"] is canonical
assert sys.modules["trustpcb.rq2.artifact_paths"] is canonical
assert canonical.__name__ == "trustpcb.rq2.artifact_paths"
for name in ("canonical_identifier", "reject_historical_output", "resolve_input",
             "HISTORICAL_PATHS", "historical_name"):
    assert getattr(legacy, name) is getattr(canonical, name)
assert canonical.historical_name.__globals__ is canonical.__dict__

original = canonical.HISTORICAL_PATHS
for owner, observer in ((legacy, canonical), (canonical, legacy)):
    with patch.object(owner, "HISTORICAL_PATHS", {{"synthetic/current": "synthetic/historical"}}):
        assert observer.historical_name("synthetic/current/file") == "synthetic/historical/file"
        assert observer.canonical_identifier("synthetic/historical/file") == "synthetic/current/file"
        owner.HISTORICAL_PATHS["synthetic/extra"] = "synthetic/previous"
        assert observer.historical_name("synthetic/extra") == "synthetic/previous"
    assert legacy.HISTORICAL_PATHS is canonical.HISTORICAL_PATHS is original
    replacement = lambda value: "patched:" + value
    with patch.object(owner, "historical_name", replacement):
        assert observer.historical_name("fixture") == "patched:fixture"
assert "torch" not in sys.modules
assert "ultralytics" not in sys.modules
'''
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(REPO / "src"), environment.get("PYTHONPATH", "")])
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=REPO,
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_legacy_first_canonical_second(self):
        self.check_import_order("trustpcb.rq2_artifact_paths", "trustpcb.rq2.artifact_paths")

    def test_canonical_first_legacy_second(self):
        self.check_import_order("trustpcb.rq2.artifact_paths", "trustpcb.rq2_artifact_paths")


class ArtifactPathTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_historical_checkpoint_resolves_with_same_bytes(self):
        checkpoint = self.root / paths.historical_name(confidence.CHECKPOINT)
        checkpoint.parent.mkdir(parents=True)
        payload = b"synthetic checkpoint bytes"
        checkpoint.write_bytes(payload)
        with patch.object(confidence, "CHECKPOINT_SHA", hashlib.sha256(payload).hexdigest()):
            identity = confidence.checkpoint_identity(self.root)
        self.assertEqual(identity["relative_path"], confidence.CHECKPOINT)
        self.assertEqual(checkpoint.read_bytes(), payload)
        with self.assertRaises(ValueError):
            confidence.checkpoint_identity(self.root)

    def test_ambiguous_locations_never_choose_silently(self):
        for name in (confidence.CHECKPOINT, paths.historical_name(confidence.CHECKPOINT)):
            checkpoint = self.root / name
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"same bytes")
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            paths.resolve_input(self.root, confidence.CHECKPOINT)

    def test_all_historical_outputs_block_new_writes(self):
        for current, historical in paths.HISTORICAL_PATHS.items():
            with self.subTest(current=current):
                (self.root / historical).mkdir(parents=True, exist_ok=True)
                with self.assertRaises(FileExistsError):
                    paths.reject_historical_output(self.root, current)
                self.assertFalse((self.root / current).exists())

    def test_historical_sidecar_blocks_training_before_verification(self):
        current = "runs/rq2/detector_training/seed_24209199"
        historical = self.root / paths.historical_name(current)
        historical.parent.mkdir(parents=True)
        sidecar = historical.with_name(historical.name + ".provenance.json")
        sidecar.write_bytes(b'{"status":"complete"}')
        with self.assertRaises(FileExistsError):
            detector.execute({"project_root": str(self.root)}, lambda *_: self.fail("no training"))
        self.assertEqual(sidecar.read_bytes(), b'{"status":"complete"}')

    def test_aliases_are_exact_and_provenance_not_modified(self):
        current = confidence.DEVELOPMENT
        original = {"development_manifest": paths.historical_name(current)}
        self.assertEqual(paths.canonical_identifier(original["development_manifest"]), current)
        self.assertEqual(original["development_manifest"], paths.historical_name(current))
        self.assertEqual(paths.canonical_identifier("unrelated/" + paths.historical_name(current)),
                         "unrelated/" + paths.historical_name(current))


if __name__ == "__main__":
    unittest.main()
