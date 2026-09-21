"""Synthetic artifact-location compatibility; no scientific execution or model loads."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trustpcb import rq2_artifact_paths as paths
from trustpcb import rq2_confidence_distribution as confidence
from trustpcb import rq2_detector_training as detector


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
