"""Read-only compatibility for immutable artifacts created before semantic naming.

Only the explicit historical aliases below are accepted. New artifacts always
use the keys; stored metadata and scientific files are never rewritten here.
"""

from pathlib import Path


HISTORICAL_PATHS = {
    "data/splits/rq2_train_development_split": "data/splits/rq2_stage1_v2",
    "data/splits/rq2_rejected_train_development_split": "data/splits/rq2_stage1",
    "runs/rq2/detector_training": "runs/rq2/stage2",
    "runs/rq2/confidence_distribution": "runs/rq2/stage3b1/raw_confidence",
}


def historical_name(relative):
    for current, historical in HISTORICAL_PATHS.items():
        if relative == current or relative.startswith(current + "/"):
            return historical + relative[len(current):]
    return None


def canonical_identifier(value):
    """Compare known historical relative paths without changing recorded provenance."""
    if isinstance(value, str):
        for current, historical in HISTORICAL_PATHS.items():
            if value == historical or value.startswith(historical + "/"):
                return current + value[len(historical):]
    return value


def resolve_input(root, relative, required=True):
    root = Path(root)
    names = [relative]
    historical = historical_name(relative)
    if historical is not None:
        names.append(historical)
    found = [root / name for name in names if (root / name).exists()]
    if len(found) > 1:
        raise RuntimeError(f"Ambiguous artifact locations; inspect before choosing: {found}")
    if not found and required:
        raise FileNotFoundError(f"Missing artifact: {relative}")
    return found[0] if found else root / relative


def reject_historical_output(root, relative):
    """Do not rerun an experiment merely because its destination was renamed."""
    historical = historical_name(relative)
    if historical is None:
        return
    path = Path(root) / historical
    sidecar = path.with_name(path.name + ".provenance.json")
    if path.exists() or sidecar.exists():
        raise FileExistsError(
            f"Historical artifact already exists: {path} (or its sidecar). "
            "Keep its provenance unchanged; consume the existing result instead of rerunning."
        )
