"""Execution boundaries shared by reconstructed historical RQ1 workflows."""

import inspect
import json
import os
from pathlib import Path


def require_dicc(confirmed):
    if os.name == 'nt' or not confirmed:
        raise RuntimeError('Research execution belongs on DICC; use --dicc there. No Windows execution.')


def fresh_output(path, project_root):
    path, root = Path(path).resolve(), Path(project_root).resolve()
    for name in ('configs', 'data', 'runs', 'results', 'outputs', 'src', 'tests', 'docs', '.git'):
        if path == root / name or root / name in path.parents:
            raise ValueError('Use a fresh reconstruction destination outside protected repository areas')
    if path == root or path in root.parents:
        raise ValueError('Destination cannot contain the checkout')
    path.mkdir(parents=True, exist_ok=False)
    (path / 'RECONSTRUCTION.json').write_text(json.dumps({
        'classification': 'new_reconstruction_execution',
        'historical_execution': False,
        'notice': 'New Python reconstruction outputs; never replace frozen historical evidence.'
    }, indent=2) + '\n', encoding='utf-8')
    return path


def inventory(dataset_root):
    """Historical sorted, nonrecursive image/label enumeration (not preprocessing)."""
    root = Path(dataset_root)
    result = {}
    for partition in ('train', 'val'):
        images = sorted(p for p in (root / 'images' / partition).iterdir()
                        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
        labels = sorted(p for p in (root / 'labels' / partition).iterdir()
                        if p.is_file() and p.suffix.lower() == '.txt')
        result[partition + '_images'] = images
        result[partition + '_labels'] = labels
        result[partition.upper() + '_LABELS'] = root / 'labels' / partition
    result['all_images'] = [(s, p) for s in ('train', 'val') for p in result[s + '_images']]
    result['all_pairs'] = [(s, p, result[s.upper() + '_LABELS'] / (p.stem + '.txt'))
                           for s, p in result['all_images']]
    return result


def apply_stage(state, function):
    """Connect explicitly named stage inputs/outputs; no notebook interpreter."""
    state.update(function(**{name: state[name] for name in inspect.signature(function).parameters}))
