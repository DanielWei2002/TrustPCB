"""Materialize portable dataset templates without reading dataset images.

Run from the checkout with PYTHONPATH=src:
    python -m trustpcb.dataset_config
Requires PyYAML, also used by the research notebooks.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    dataset_root: Path


def find_project_root(start=None):
    """Find the checkout from the current directory or a directory below it."""
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "configs" / "paths.example.yaml").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate TrustPCB; run from its checkout.")


def _read_yaml(path):
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_paths(project_dir=None):
    """Load local settings; relative roots are relative to the checkout."""
    checkout = find_project_root() if project_dir is None else Path(project_dir).resolve()
    config_path = checkout / "configs" / "local" / "paths.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing {config_path}; copy configs/paths.example.yaml there "
            "and configure project_root and dataset_root."
        )
    config = _read_yaml(config_path)
    roots = {}
    for key in ("project_root", "dataset_root"):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{config_path}: {key} must be a nonempty path string")
        path = Path(value).expanduser()
        roots[key] = (path if path.is_absolute() else checkout / path).resolve()
        if not roots[key].is_dir():
            raise FileNotFoundError(f"{key} is not an existing directory: {roots[key]}")
    return ProjectPaths(**roots)


def generate_runtime_configs(paths):
    """Write six local files, preserving every manifest entry's position.

    Only the dataset root directory is checked. No images, labels, caches,
    models, or Ultralytics APIs are accessed. Physical train/val folders in
    each source entry are retained regardless of scientific split assignment.
    """
    if not paths.dataset_root.is_absolute() or not paths.project_root.is_absolute():
        raise ValueError("Use absolute roots from load_paths()")
    if not paths.dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root is not an existing directory: {paths.dataset_root}")
    source_dir = paths.project_root / "configs" / "datasets"
    runtime_dir = paths.project_root / "configs" / "local" / "datasets"
    payloads = {}
    yaml_files = {}
    for split in ("supplied", "similarity_aware"):
        yaml_name = f"dspcbsd_{split}.yaml"
        config = _read_yaml(source_dir / yaml_name)
        if "names" not in config:
            raise ValueError(f"{yaml_name}: missing class names")
        for partition in ("train", "val"):
            list_name = f"{split}_{partition}_images.txt"
            if config.get(partition) != list_name:
                raise ValueError(f"{yaml_name}: {partition} must reference {list_name}")
            entries = (source_dir / list_name).read_text(encoding="utf-8").splitlines()
            if not entries:
                raise ValueError(f"Empty manifest: {list_name}")
            absolute_entries = []
            for index, entry in enumerate(entries, 1):
                parts = PurePosixPath(entry).parts
                if (
                    len(parts) != 3
                    or parts[0] != "images"
                    or parts[1] not in ("train", "val")
                    or parts[2] in (".", "..")
                    or "\\" in entry
                    or ":" in entry
                    or entry != "/".join(parts)
                    or entry != entry.strip()
                ):
                    raise ValueError(f"{list_name}:{index}: invalid dataset-relative path {entry!r}")
                absolute_entries.append((paths.dataset_root / Path(*parts)).as_posix())
            payloads[list_name] = "\n".join(absolute_entries) + "\n"
            config[partition] = (runtime_dir / list_name).as_posix()
        payloads[yaml_name] = yaml.safe_dump(config, sort_keys=False)
        yaml_files[split] = runtime_dir / yaml_name
    # Validate all templates before writing any runtime file.
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        (runtime_dir / name).write_text(content, encoding="utf-8", newline="\n")
    return yaml_files


if __name__ == "__main__":
    for name, path in generate_runtime_configs(load_paths()).items():
        print(f"{name}: {path}")
