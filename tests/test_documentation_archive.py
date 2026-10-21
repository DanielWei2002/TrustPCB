"""Tiny documentation-retirement fixtures; no research or figure regeneration."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trustpcb import documentation_archive as archive


class DocumentationArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "docs").mkdir()
        (self.root / "outputs/rq2").mkdir(parents=True)
        self.old = b"Historical documentation\n"
        self.current = b"Current editable document fixture"
        (self.root / "docs/guide.docx").write_bytes(self.current)
        self.source = {
            "path": "outputs/rq2/README.md", "sha256": archive.digest(self.old),
            "kind": "git_blob", "commit": "a" * 40, "blob_oid": "b" * 40,
            "blob_sha256": archive.digest(self.old),
            "checkout_transform": {"mode": "identity"},
        }
        self.registry = {
            "archived_sources": [self.source],
            "retired_documents": [{"path": self.source["path"], "sha256": self.source["sha256"],
                                   "destination": "docs/guide.docx"}],
            "current_files": [{"path": "docs/guide.docx", "sha256": archive.digest(self.current)}],
        }
        self.manifest = {"artifacts": [{"path": "README.md", "sha256": archive.digest(self.old),
                                       "sources": []}]}
        self.save()
        def git_bytes(command, **kwargs):
            return "b" * 40 + "\n" if "rev-parse" in command else self.old
        mocked = patch("trustpcb.documentation_archive._archive_sources.subprocess.check_output", side_effect=git_bytes)
        self.git = mocked.start()
        self.addCleanup(mocked.stop)

    def save(self):
        (self.root / archive.INDEX).write_text(json.dumps(self.registry), encoding="utf-8")
        (self.root / "outputs/rq2/manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def verify(self):
        return archive.verify_reporting_manifest(self.root, "outputs/rq2", self.manifest)

    def test_archived_document_counts_without_local_readme(self):
        self.assertEqual(self.verify(), 1)
        self.assertTrue(any("show" in call.args[0] for call in self.git.call_args_list))
        self.assertTrue(any("rev-parse" in call.args[0] for call in self.git.call_args_list))

    def test_builder_reuses_exact_record_without_recreating_readme(self):
        record = archive.retired_summary(self.root, "outputs/rq2", "README.md")
        self.assertEqual(record, self.manifest["artifacts"][0])
        self.assertFalse((self.root / "outputs/rq2/README.md").exists())
        self.assertIsNone(archive.retired_summary(self.root, "outputs/rq2", "table.csv"))

    def test_current_document_edit_is_rejected(self):
        (self.root / "docs/guide.docx").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.verify()

    def test_archived_blob_hash_mismatch_is_rejected(self):
        self.source["blob_sha256"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "Archived blob changed"):
            self.verify()

    def test_missing_git_history_fails(self):
        self.git.side_effect = FileNotFoundError("missing Git history")
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_retired_readme_reappearance_is_rejected(self):
        (self.root / "outputs/rq2/README.md").write_bytes(self.old)
        with self.assertRaisesRegex(ValueError, "unexpectedly exists"):
            self.verify()

    def test_scientific_archive_exception_is_rejected(self):
        self.source["path"] = "runs/rq2/summary.json"
        self.save()
        with self.assertRaisesRegex(ValueError, "historical documentation"):
            self.verify()

    def test_scientific_output_changes_still_fail(self):
        self.manifest["artifacts"].append({"path": "table.csv", "sha256": archive.digest(b"original"), "sources": []})
        (self.root / "outputs/rq2/table.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.verify()

    def test_missing_archive_mapping_does_not_skip_original_hash(self):
        self.manifest["artifacts"][0]["sha256"] = "0" * 64
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_unsafe_paths_are_rejected(self):
        for path in ("../outside", "C:/outside", "a\\outside", "/outside"):
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                archive.checked_source(self.root, {"path": path, "sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()
