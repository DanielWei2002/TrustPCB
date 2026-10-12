"""Reporting provenance validation with temporary synthetic bytes only."""

import json
from pathlib import Path
import tempfile
import unittest

from trustpcb.rq1 import reporting


class ReportingManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "outputs/rq1"
        self.output.mkdir(parents=True)
        (self.root / "source.csv").write_bytes(b"synthetic\n")
        (self.output / "copy.csv").write_bytes(b"synthetic\n")
        record = {"path": "copy.csv", "sha256": reporting.digest(b"synthetic\n"),
                  "sources": [{"path": "source.csv", "sha256": reporting.digest(b"synthetic\n")}]}
        (self.output / "manifest.json").write_text(json.dumps({"artifacts": [record]}), encoding="utf-8")

    def test_valid_copy(self):
        self.assertEqual(reporting.verify(self.root), 1)

    def test_source_change_rejected(self):
        (self.root / "source.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            reporting.verify(self.root)

    def test_output_change_rejected(self):
        (self.output / "copy.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            reporting.verify(self.root)

    def test_unlisted_file_blocks_rebuild(self):
        (self.output / "user.csv").write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "Unlisted"):
            reporting.build(self.root)
        self.assertEqual((self.output / "user.csv").read_bytes(), b"keep")
