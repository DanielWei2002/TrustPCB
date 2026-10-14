"""Reporting provenance validation with temporary synthetic bytes only."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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

    def test_archived_source_is_verified_without_a_working_file(self):
        from trustpcb.rq1 import archived_sources
        manifest_path = self.output / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        source = manifest['artifacts'][0]['sources'][0]
        source.update(kind='git_blob', commit='a' * 40,
                      blob_sha256=reporting.digest(b'synthetic\n'))
        manifest_path.write_text(json.dumps(manifest))
        (self.root / 'source.csv').unlink()
        with patch.object(archived_sources.subprocess, 'check_output', return_value=b'synthetic\n'):
            self.assertEqual(reporting.verify(self.root), 1)
        with patch.object(archived_sources.subprocess, 'check_output', return_value=b'tampered\n'):
            with self.assertRaisesRegex(ValueError, 'Archived blob changed'):
                reporting.verify(self.root)

    def test_validator_dependency_hash_is_not_skipped(self):
        manifest_path = self.output / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['validation_sources'] = [{'path': 'source.csv', 'sha256': '0' * 64}]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'source changed'):
            reporting.verify(self.root)
