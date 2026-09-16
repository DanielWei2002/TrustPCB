"""Frozen, sequential RQ1 workflow. Only the explicit DICC worker imports YOLO."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import yaml

from trustpcb.dataset_config import find_project_root, generate_runtime_configs, load_paths


SEEDS = (
    ("24209199", 24209199),
    ("20260902", 20260902),
    ("02092026", 2092026),
    ("12345678", 12345678),
    ("20202027", 20202027),
)
SPLITS = ("supplied", "similarity_aware")
BASELINE_FILE = "configs/experiments/yolov8n_preliminary_baseline_v1.yaml"
EXECUTION_INPUTS = ("src/trustpcb/", BASELINE_FILE, "configs/datasets/", "data/splits/")
# A drift guard for the approved design, not a second configuration source.
APPROVED_BASELINE = {
    "model": "yolov8n.pt", "imgsz": 640, "epochs": 100, "batch": 32,
    "seed": 24209199, "deterministic": True, "device": 0, "workers": 2,
    "patience": 100, "pretrained": True, "amp": True, "plots": True,
    "save": True, "val": True,
}
METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP@0.50": "metrics/mAP50(B)",
    "mAP@0.50:0.95": "metrics/mAP50-95(B)",
}
SELECTION = "highest logged mAP50-95; first epoch on a CSV tie"


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _sha(path):
    # Called only on small source/configuration/CSV files, never checkpoints.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(_json(value))
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(_json(value), encoding="utf-8")
        temporary.replace(path)


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def git_provenance(root):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE
        ).strip()
    status = git("status", "--porcelain", "--untracked-files=all", "--", *EXECUTION_INPUTS)
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(status), "git_input_status": status}


def require_clean_inputs(root):
    info = git_provenance(root)
    if info["git_dirty"]:
        raise RuntimeError(
            "RQ1 execution/scientific inputs must be committed and clean before training/reuse. "
            "Commit or resolve these changes explicitly:\n" + info["git_input_status"]
        )
    return info


def model_provenance(root, model):
    """Hash the local pretrained input as bytes, without loading a model."""
    path = (Path(root) / model).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required local pretrained model is missing: {path}; automatic download is disabled")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        before = os.fstat(file.fileno())
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(file.fileno())
    signature = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise RuntimeError("Pretrained model changed while being hashed")
    if after.st_size == 0:
        raise ValueError(f"Pretrained model is empty: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": after.st_size}


def bind_pretrained_model(root, model):
    """Persist one immutable input identity across separate DICC invocations."""
    identity = model_provenance(root, model)
    binding = Path(root) / "runs/rq1/pretrained_model.json"
    if binding.exists():
        if _read_json(binding) != identity:
            raise RuntimeError("Pretrained model differs from the frozen RQ1 model file; stopping")
    else:
        if binding.parent.exists() and any(p.name != ".runner.lock" for p in binding.parent.iterdir()):
            raise RuntimeError("RQ1 artifacts exist but their pretrained model binding is missing; inspect manually")
        _write_json(binding, identity, exclusive=True)
    return identity


def verify_pretrained_model(plan):
    root = Path(plan["output_dir"]).parents[3]
    expected = plan["pretrained_model"]
    if _read_json(root / "runs/rq1/pretrained_model.json") != expected:
        raise RuntimeError("RQ1 pretrained model binding changed")
    if model_provenance(root, plan["model"]) != expected:
        raise RuntimeError("Pretrained model file/hash changed; stopping training/reuse")
    return expected


def build_plan(root, git_info=None):
    """Read configuration text and construct all ten runs without dataset access."""
    root = Path(root).resolve()
    baseline = yaml.safe_load((root / BASELINE_FILE).read_text(encoding="utf-8"))
    if _json(baseline) != _json(APPROVED_BASELINE):
        raise ValueError(f"{BASELINE_FILE} differs from the approved RQ1 configuration")
    git_info = git_provenance(root) if git_info is None else git_info
    inputs = [root / BASELINE_FILE, root / "src/trustpcb/rq1.py",
              root / "src/trustpcb/dataset_config.py"]
    inputs += sorted((root / "configs/datasets").glob("*"))
    inputs += sorted((root / "data/splits").glob("*"))
    hashes = {p.relative_to(root).as_posix(): _sha(p) for p in inputs if p.is_file()}
    plans = []
    for label, seed in SEEDS:
        for split in SPLITS:
            output = root / "runs/rq1" / split / f"seed_{label}"
            data = root / "configs/local/datasets" / f"dspcbsd_{split}.yaml"
            kwargs = {k: v for k, v in baseline.items() if k != "model"}
            kwargs.update(seed=seed, data=str(data), project=str(output.parent),
                          name=output.name, exist_ok=False)
            plans.append({
                "schema_version": 1, "experiment_id": f"rq1_{split}_seed_{label}",
                "split": split, "seed_label": label, "seed_value": seed,
                **git_info, "input_hashes": hashes, "baseline_file": BASELINE_FILE,
                "baseline_config": baseline, "model": baseline["model"],
                "train_kwargs": kwargs, "runtime_dataset_yaml": str(data),
                "output_dir": str(output), "best_checkpoint": str(output / "weights/best.pt"),
                "metric_selection": SELECTION,
            })
    return plans


def select_runs(plans, seed=None, split=None, remaining=False):
    if seed is not None and seed not in dict(SEEDS):
        raise ValueError(f"Unknown seed label: {seed!r}; preserve leading zeroes")
    if split is not None and (split not in SPLITS or seed is None):
        raise ValueError("A split requires one frozen seed label")
    if seed is not None and remaining:
        raise ValueError("Select a seed or remaining pairs, not both")
    return [p for p in plans
            if (seed is None or p["seed_label"] == seed)
            and (split is None or p["split"] == split)
            and (not remaining or p["seed_label"] != SEEDS[0][0])]


def _sidecar(plan):
    output = Path(plan["output_dir"])
    return output.with_name(output.name + ".provenance.json")


def _identity(plan):
    return {k: v for k, v in plan.items() if k != "git_dirty"}


def _runtime_hashes(plan):
    data = Path(plan["runtime_dataset_yaml"])
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    files = [data, Path(config["train"]), Path(config["val"])]
    if any(p.parent != data.parent for p in files):
        raise ValueError("Runtime YAML must reference manifests in configs/local/datasets")
    return {p.name: _sha(p) for p in files}


def read_metrics(path, epochs):
    """Extract a single common epoch's metrics; never combine per-metric maxima."""
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        records = []
        for raw in reader:
            row = {key.strip(): value.strip() for key, value in raw.items()
                   if key is not None and value is not None}
            epoch_value = float(row["epoch"])
            if not epoch_value.is_integer():
                raise ValueError("Non-integer epoch in results.csv")
            values = {name: float(row[column]) for name, column in METRICS.items()}
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
                raise ValueError("Invalid or missing detection metrics in results.csv")
            records.append({"best_epoch": int(epoch_value), **values})
    if [r["best_epoch"] for r in records] != list(range(1, epochs + 1)):
        raise ValueError(f"Expected exactly contiguous epochs 1..{epochs}")
    return max(records, key=lambda row: row["mAP@0.50:0.95"])


def _artifacts(plan):
    output = Path(plan["output_dir"])
    args_path = output / "args.yaml"
    actual = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    expected = {"model": plan.get("pretrained_model", {}).get("path", plan["model"]),
                **plan["train_kwargs"]}
    for key, value in expected.items():
        observed = actual.get(key)
        if key == "device":
            matches = str(observed) == str(value)
        elif key in ("data", "project") or (key == "model" and "pretrained_model" in plan):
            matches = isinstance(observed, str) and Path(observed).resolve() == Path(value).resolve()
        else:
            matches = observed == value
        if not matches:
            raise ValueError(f"args.yaml mismatch for {key}: {observed!r} != {value!r}")
    checkpoint = Path(plan["best_checkpoint"])
    size = checkpoint.stat().st_size
    if not checkpoint.is_file() or size <= 0:
        raise ValueError("Missing or empty best.pt")
    metrics = read_metrics(output / "results.csv", expected["epochs"])
    return {"metrics": metrics, "results_sha256": _sha(output / "results.csv"),
            "args_sha256": _sha(args_path), "best_checkpoint_bytes": size,
            "best_checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns}


def inspect_run(plan, for_execution=True):
    """No checkpoint loads: success receipt + matching provenance + artifact checks."""
    output = Path(plan["output_dir"])
    sidecar = _sidecar(plan)
    if not output.exists() and not sidecar.exists():
        return "not_started", None, "No run artifacts"
    try:
        record = _read_json(sidecar)
        stored = record["plan"]
        if for_execution:
            verify_pretrained_model(plan)
            if _identity(stored) != _identity(plan):
                raise ValueError("Existing provenance differs from the current plan/commit/source")
            if record["runtime_hashes"] != _runtime_hashes(plan):
                raise ValueError("Runtime dataset configuration changed")
        else:
            # Reporting preserves the execution commit even if reporting code has changed.
            for key in ("experiment_id", "seed_label", "seed_value", "split", "model",
                        "train_kwargs", "baseline_config", "output_dir", "best_checkpoint",
                        "runtime_dataset_yaml", "metric_selection"):
                if stored[key] != plan[key]:
                    raise ValueError(f"Stored plan mismatch: {key}")
            for name, digest in plan["input_hashes"].items():
                if name.startswith(("configs/", "data/splits/")) and stored["input_hashes"].get(name) != digest:
                    raise ValueError(f"Scientific/configuration input changed: {name}")
        if record["status"] != "completed":
            raise ValueError(f"Run status is {record['status']!r}")
        receipt = _read_json(output / "rq1_complete.json")
        if receipt != record:
            raise ValueError("Completion receipt and provenance disagree")
        if record["artifacts"] != _artifacts(stored):
            raise ValueError("Completed artifacts changed")
        return "completed", record, "Verified completion"
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, yaml.YAMLError) as exc:
        return "incomplete", None, str(exc)


def _environment():
    versions = {}
    for name in ("ultralytics", "torch", "numpy", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


def _yolo_train(plan):
    # DICC-only. No import or invocation of this adapter during local verification.
    pretrained = verify_pretrained_model(plan)
    from ultralytics import YOLO
    model = YOLO(pretrained["path"])

    def check_directory(trainer):
        if Path(trainer.save_dir).resolve() != Path(plan["output_dir"]).resolve():
            raise RuntimeError("Ultralytics selected an unexpected output directory; stopping before training")

    model.add_callback("on_pretrain_routine_start", check_directory)
    model.train(**plan["train_kwargs"])
    if Path(model.trainer.save_dir).resolve() != Path(plan["output_dir"]).resolve():
        raise RuntimeError("Ultralytics used an unexpected output directory")


def execute_one(plan, train_call):
    """Execute an injected adapter; tests supply a fake artifact writer only."""
    require_clean_inputs(Path(plan["output_dir"]).parents[3])
    verify_pretrained_model(plan)
    status, _, detail = inspect_run(plan)
    if status == "completed":
        print(f"SKIP {plan['experiment_id']}: {detail}", flush=True)
        return "skipped"
    if status != "not_started":
        raise RuntimeError(f"STOP {plan['output_dir']}: {detail}. Inspect manually; no automatic overwrite/resume.")
    record = {"plan": plan, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "environment": _environment(), "runtime_hashes": _runtime_hashes(plan)}
    # Reserve an adjacent sidecar, not the YOLO directory: exist_ok=False stays effective.
    _write_json(_sidecar(plan), record, exclusive=True)
    print(f"START {plan['experiment_id']} seed={plan['seed_value']} data={plan['runtime_dataset_yaml']}", flush=True)
    try:
        train_call(plan)
        verify_pretrained_model(plan)
        artifacts = _artifacts(plan)
        if record["runtime_hashes"] != _runtime_hashes(plan):
            raise RuntimeError("Runtime dataset files changed during training")
        record.update(status="completed", artifacts=artifacts,
                      completed_at=datetime.now(timezone.utc).isoformat())
        _write_json(Path(plan["output_dir"]) / "rq1_complete.json", record, exclusive=True)
        _write_json(_sidecar(plan), record)
    except BaseException as exc:
        record.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        _write_json(_sidecar(plan), record)
        raise
    print(f"DONE {plan['experiment_id']}", flush=True)
    return "completed"


def _worker_command(root, plan):
    return [sys.executable, "-B", "-m", "trustpcb.rq1", "--project-dir", str(root),
            "_worker", "--seed", plan["seed_label"], "--split", plan["split"], "--dicc"]


def run_sequential(root, plans, launch=None):
    """Blocking child processes: one interpreter per run, never a parallel queue."""
    launch = subprocess.run if launch is None else launch
    require_clean_inputs(root)
    pretrained = bind_pretrained_model(root, plans[0]["model"])
    for plan in plans:
        plan["pretrained_model"] = pretrained
    for plan in plans:
        require_clean_inputs(root)
        verify_pretrained_model(plan)
        status, _, detail = inspect_run(plan)
        if status == "completed":
            print(f"SKIP {plan['experiment_id']}: {detail}", flush=True)
            continue
        if status != "not_started":
            raise RuntimeError(f"STOP {plan['output_dir']}: {detail}; explicit manual action required")
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(plan["seed_value"])
        env["PYTHONPATH"] = str(Path(root) / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["TRUSTPCB_RQ1_WORKER"] = "1"
        env["TRUSTPCB_RQ1_MODEL"] = _json(pretrained)
        print(f"NEXT {plan['seed_label']} {plan['split']} -> {plan['output_dir']}", flush=True)
        launch(_worker_command(root, plan), env=env, cwd=str(root), check=True)
        verify_pretrained_model(plan)
        status, _, detail = inspect_run(plan)
        if status != "completed":
            raise RuntimeError(f"Worker did not produce verified completion: {detail}")


def extract_results(root, plans):
    """Read logs/receipts only; blank metrics for absent or ambiguous runs."""
    rows = []
    for plan in plans:
        status, record, detail = inspect_run(plan, for_execution=False)
        row = {key: plan[key] for key in ("experiment_id", "seed_label", "seed_value", "split",
                                        "output_dir", "best_checkpoint", "metric_selection")}
        row.update(status=status, detail=detail, git_commit="", best_epoch="",
                   **{metric: "" for metric in METRICS})
        if record is not None:
            row.update(git_commit=record["plan"]["git_commit"], **record["artifacts"]["metrics"])
        rows.append(row)
        print(f"{plan['experiment_id']}: {status} ({detail})")
    aggregates = []
    for split in SPLITS:
        complete = [row for row in rows if row["split"] == split and row["status"] == "completed"]
        full = len(complete) == 5 and {row["seed_label"] for row in complete} == set(dict(SEEDS))
        for metric in METRICS:
            aggregates.append({"split": split, "metric": metric, "n_completed": len(complete),
                               "n_expected": 5, "status": "complete" if full else "incomplete",
                               "mean": statistics.mean(r[metric] for r in complete) if full else "",
                               "std": statistics.stdev(r[metric] for r in complete) if full else "",
                               "std_ddof": 1})
    destination = Path(root) / "results/rq1"
    destination.mkdir(parents=True, exist_ok=True)
    for name, records in (("per_seed_metrics.csv", rows), ("aggregate_metrics.csv", aggregates)):
        output = destination / name
        temporary = output.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(output)
    return rows, aggregates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "train", "_worker"):
        action = sub.add_parser(command)
        selection = action.add_mutually_exclusive_group(required=command != "plan")
        selection.add_argument("--seed", choices=list(dict(SEEDS)))
        selection.add_argument("--all", action="store_true")
        selection.add_argument("--remaining", action="store_true", help="The last four seeds in frozen order")
        action.add_argument("--split", choices=SPLITS)
        if command != "plan":
            action.add_argument("--dicc", action="store_true", help="Explicitly authorize training on DICC")
    sub.add_parser("extract")
    args = parser.parse_args(argv)
    try:
        root = args.project_dir.resolve() if args.project_dir else find_project_root()
        if args.command in ("train", "_worker"):
            if os.name == "nt" or not args.dicc:
                raise ValueError("Training is DICC-only; run on DICC with --dicc. Local Windows training is disabled.")
            require_clean_inputs(root)
            paths = load_paths(root)
            if paths.project_root != root:
                raise ValueError("--project-dir must match project_root in configs/local/paths.yaml")
            if args.command == "_worker":
                if os.environ.get("TRUSTPCB_RQ1_WORKER") != "1" or args.seed is None or args.split is None:
                    raise ValueError("Internal worker must be launched by the sequential runner")
                if os.environ.get("PYTHONHASHSEED") != str(dict(SEEDS)[args.seed]):
                    raise ValueError("Worker PYTHONHASHSEED does not match the integer seed")
                plan = select_runs(build_plan(root), args.seed, args.split)[0]
                plan["pretrained_model"] = json.loads(os.environ["TRUSTPCB_RQ1_MODEL"])
                verify_pretrained_model(plan)
                execute_one(plan, _yolo_train)
                return 0
            lock = root / "runs/rq1/.runner.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            # An interrupted parent leaves this lock for explicit human inspection.
            with lock.open("x", encoding="utf-8") as file:
                file.write(f"pid={os.getpid()}\n")
            try:
                generate_runtime_configs(paths)
                plans = select_runs(build_plan(root), args.seed, args.split, args.remaining)
                run_sequential(root, plans)
            finally:
                lock.unlink()
        else:
            plans = build_plan(root)
            if args.command == "extract":
                extract_results(root, plans)
            else:
                plans = select_runs(plans, args.seed, args.split, args.remaining)
                print(_json({"status": "planned", "runs": plans}), end="")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError, yaml.YAMLError) as exc:
        print(f"RQ1 STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
