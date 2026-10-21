"""Explicit Git-backed documentation retirement without changing frozen manifests.

Only documentation and the three presentation generators may be archived here.
Scientific inputs, copies, figures and original references always validate locally.
The sidecar preserves historical hashes and separately binds current guides/code.
"""

import importlib.util
import json
from pathlib import Path, PurePosixPath

# Reuse the existing pure Git byte verifier without executing rq1.__init__,
# which eagerly imports its scientific workflow and YAML dependency.
_spec = importlib.util.spec_from_file_location(
    "trustpcb_documentation_git_sources", Path(__file__).parent / "rq1/archived_sources.py")
_archive_sources = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_archive_sources)
digest = _archive_sources.digest
read_source = _archive_sources.read_source

INDEX = "docs/documentation_archive.json"
REPORTERS = {f"src/trustpcb/rq{i}/reporting.py" for i in (1, 2, 3)}


def checked_source(root, record):
    path = record["path"]
    part = PurePosixPath(path)
    if part.is_absolute() or ".." in part.parts or "\\" in path or ":" in path:
        raise ValueError(f"Unsafe reporting path: {path}")
    resolved = (Path(root) / path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Reporting path escapes root: {path}")
    return read_source(root, record)


def documentation_path(path):
    """Never permit an archive exception for a scientific artifact."""
    return (path == "README.md" or path in REPORTERS
            or (path.startswith("docs/") and path.endswith(".md"))
            or (path.startswith(("outputs/rq1/", "outputs/rq2/", "outputs/rq3/", "results/rq1/"))
                and PurePosixPath(path).name == "README.md"))


class DocumentationArchive:
    def __init__(self, root):
        self.root = Path(root)
        index = self.root / INDEX
        self.records = {}
        self.retired = set()
        if not index.exists():
            return  # Legacy manifests and synthetic fixtures have no archive.
        registry = json.loads(index.read_text(encoding="utf-8"))
        for record in registry["archived_sources"]:
            if not documentation_path(record["path"]) or record.get("kind") != "git_blob":
                raise ValueError("Archive exception must identify historical documentation")
            key = (record["path"], record["sha256"])
            if key in self.records:
                raise ValueError("Duplicate documentation archive identity")
            self.records[key] = record
        for item in registry["retired_documents"]:
            path = item["path"]
            if PurePosixPath(path).name != "README.md" or path == "README.md":
                raise ValueError("Only nonroot READMEs may be retired")
            if (path, item["sha256"]) not in self.records:
                raise ValueError("Retired documentation is missing its archive identity")
            self.retired.add(path)
            if (self.root / path).exists():
                raise ValueError(f"Retired README unexpectedly exists: {path}")
        # Current document and validation code hashes supplement, never replace,
        # the historical generator/source hashes in the reporting manifests.
        for record in registry["current_files"]:
            if "kind" in record:
                raise ValueError("Current documentation must validate from local bytes")
            checked_source(self.root, record)

    def read(self, record):
        if record.get("kind") == "git_blob":
            return checked_source(self.root, record)
        archived = self.records.get((record["path"], record["sha256"]))
        return checked_source(self.root, archived or record)


def verify_reporting_manifest(root, output_relative, manifest):
    root = Path(root)
    output = root / output_relative
    archive = DocumentationArchive(root)
    for record in manifest.get("validation_sources", []):
        archive.read(record)
    expected = {"manifest.json"}
    seen = set()
    for record in manifest["artifacts"]:
        path = record["path"]
        if path in seen or path == "manifest.json":
            raise ValueError("Duplicate reporting artifact path")
        seen.add(path)
        relative = f"{output_relative}/{path}"
        source = {"path": relative, "sha256": record["sha256"]}
        if relative not in archive.retired:
            expected.add(path)
            # Do not allow archived source rules to conceal a modified output.
            try:
                checked_source(root, source)
            except ValueError as exc:
                raise ValueError(f"Reporting artifact changed: {path}") from exc
        else:
            archive.read(source)
        for source in record["sources"]:
            archive.read(source)
    for record in manifest.get("references", []):
        try:
            checked_source(root, record)
        except ValueError as exc:
            raise ValueError(f"Referenced evidence changed: {record['path']}") from exc
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError(f"Unlisted/missing reporting files: {actual ^ expected}")
    return len(manifest["artifacts"])


def retired_summary(root, output_relative, path):
    """Reuse a verified historical record without recreating a retired README.

    Called by presentation builders before writing authored documentation. The
    original manifest must remain available; missing history is a hard failure.
    """
    if PurePosixPath(path).name != "README.md":
        return None
    archive = DocumentationArchive(root)
    relative = f"{output_relative}/{path}"
    if relative not in archive.retired:
        return None
    manifest = json.loads((Path(root) / output_relative / "manifest.json").read_text(encoding="utf-8"))
    record = next(item for item in manifest["artifacts"] if item["path"] == path)
    archive.read({"path": relative, "sha256": record["sha256"]})
    for source in record["sources"]:
        archive.read(source)
    return record
