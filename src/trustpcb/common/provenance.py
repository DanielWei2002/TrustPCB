"""Serialization, byte hashing and provenance collection without research policy.

Callers supply the Git input scope and package list. No model or dataset is loaded.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def serialize_json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def sha256_file(path):
    # Called only on small source/configuration/CSV files, never checkpoints.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(serialize_json(value))
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(serialize_json(value), encoding="utf-8")
        temporary.replace(path)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def git_provenance(root, execution_inputs):
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=str(root), text=True, stderr=subprocess.PIPE
        )
    # Separate plumbing avoids older Git's limited `status` pathspec support.
    # NUL-delimited names preserve whitespace/newlines; command failures propagate.
    changes = []
    for kind, command in (
        ("unstaged", ("diff", "--name-only", "-z")),
        ("staged", ("diff", "--cached", "--name-only", "-z")),
        ("untracked", ("ls-files", "--others", "--exclude-standard", "-z")),
    ):
        names = git(*command, "--", *execution_inputs).split("\0")
        changes.extend(f"{kind}: {name!r}" for name in names if name)
    status = "\n".join(changes)
    return {"git_commit": git("rev-parse", "HEAD").strip(),
            "git_dirty": bool(status), "git_input_status": status}


def environment(packages):
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
