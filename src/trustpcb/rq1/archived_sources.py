"""Verify historical source bytes in Git without restoring or executing notebooks.

Blob SHA-256 is the authoritative Git identity. A recorded newline transform
also verifies the original reporting manifest's checkout-byte SHA-256 unchanged.
Missing history or either mismatching hash is an error, never a skipped check.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_source(root, record):
    root = Path(root)
    path = record['path']
    if PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts or '\\' in path:
        raise ValueError('Invalid source path')
    if record.get('kind') != 'git_blob':
        data = (root / path).read_bytes()
    else:
        commit = record['commit']
        if not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError('Archive requires an immutable full Git commit')
        spec = f'{commit}:{path}'
        blob = subprocess.check_output(['git', '-C', str(root), 'show', spec])
        if digest(blob) != record['blob_sha256']:
            raise ValueError(f'Archived blob changed: {path}')
        if 'blob_oid' in record:
            oid = subprocess.check_output(['git', '-C', str(root), 'rev-parse', spec], text=True).strip()
            if oid != record['blob_oid']:
                raise ValueError(f'Archived object identity changed: {path}')
        transform = record.get('checkout_transform', {'mode': 'identity'})
        if transform['mode'] == 'identity':
            data = blob
        elif transform['mode'] == 'lf_to_crlf_except':
            if b'\r\n' in blob:
                raise ValueError('Expected LF Git source')
            keep = set(transform['lf_lines'])
            # Split on LF only: Unicode separators inside JSON strings are data.
            lines = blob.split(b'\n')
            if not keep <= set(range(len(lines) - 1)):
                raise ValueError('Invalid archived newline indexes')
            data = b''.join(line + (b'\n' if i in keep else b'\r\n')
                            for i, line in enumerate(lines[:-1])) + lines[-1]
        else:
            raise ValueError('Unknown archived checkout transform')
    if digest(data) != record['sha256']:
        raise ValueError(f'Reporting source changed: {path}')
    return data


def notebook_record(root, name):
    records = json.loads((Path(root) / 'docs/rq1/notebook_archive.json').read_text(encoding='utf-8'))['sources']
    return next(r for r in records if r['path'] == f'notebooks/{name}')


def read_notebook(root, name):
    """Read archived JSON as data; never execute a cell."""
    return json.loads(read_source(root, notebook_record(root, name)))
