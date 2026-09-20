"""RQ2 Stage 2 preparation and guarded DICC-only execution. No ML imports at import time."""

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

import yaml

from trustpcb import rq1
from trustpcb.dataset_config import find_project_root, load_paths

STAGE1_COMMIT = "aaed8091323cac90ae5ae565719ce4b48a926335"
TEMPLATE = "configs/datasets/dspcbsd_rq2_train_dev.yaml"
MANIFESTS = {
    "train": ("data/splits/rq2_stage1_v2/detector_train_images.txt", 7386,
              "ec3f484e81364a1cd8619852578020aed9b9ce928eccf39cebc2613477155c06"),
    "val": ("data/splits/rq2_stage1_v2/development_calibration_images.txt", 821,
            "1985f0d4097812462069be9c97cb672c4f6b3257ba8cf6a2d55243d3a7f3a6e1"),
}
NAMES = dict(enumerate(("SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO")))
METRIC = "metrics/mAP50-95(B)"
SELECTION = "highest stored development metrics/mAP50-95(B); earliest epoch on exact decimal tie"


def frozen_inputs(root):
    """Validate exact frozen membership/order using LF-canonical hashes; read text only."""
    root = Path(root)
    template = yaml.safe_load((root / TEMPLATE).read_text(encoding="utf-8"))
    expected = {**{key: spec[0] for key, spec in MANIFESTS.items()}, "names": NAMES}
    if template != expected:
        raise ValueError("RQ2 template must reference only the frozen train/dev manifests and class mapping")
    lists = {}
    for key, (relative, count, digest) in MANIFESTS.items():
        raw = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Frozen Stage 1 manifest hash changed: {relative}")
        lines = raw.decode("utf-8").splitlines()
        if len(lines) != count or len(set(lines)) != count:
            raise ValueError(f"Invalid frozen manifest count/duplicates: {relative}")
        for line in lines:
            path = PurePosixPath(line)
            if (len(path.parts) != 3 or path.parts[0] != "images"
                    or path.parts[1] not in ("train", "val") or "\\" in line
                    or ":" in line or ".." in path.parts or path.as_posix() != line):
                raise ValueError(f"Invalid portable image path: {line!r}")
        lists[key] = lines
    if set(lists["train"]) & set(lists["val"]):
        raise ValueError("Frozen train/dev overlap")
    return lists


def build_plan(root, git_info=None):
    root = Path(root).resolve()
    frozen_inputs(root)
    config = yaml.safe_load((root / rq1.BASELINE_FILE).read_text(encoding="utf-8"))
    if rq1._json(config) != rq1._json(rq1.APPROVED_BASELINE):
        raise ValueError("Training configuration differs from the frozen baseline")
    output = root / "runs/rq2/stage2/seed_24209199"
    data = root / "configs/local/datasets/rq2_stage2/dspcbsd_rq2_train_dev.yaml"
    kwargs = {key: value for key, value in config.items() if key != "model"}
    kwargs.update(data=str(data), project=str(output.parent), name=output.name,
                  exist_ok=False, resume=False, split="val")
    inputs = [TEMPLATE, rq1.BASELINE_FILE, "src/trustpcb/rq2_train.py",
              "src/trustpcb/rq1.py", "src/trustpcb/dataset_config.py"]
    return {
        "schema_version": 1, "stage": "rq2_stage2", "project_root": str(root),
        **(rq1.git_provenance(root) if git_info is None else git_info),
        "stage1_commit": STAGE1_COMMIT,
        "frozen_manifests": {key: {"path": spec[0], "count": spec[1],
                                    "sha256_lf": spec[2]} for key, spec in MANIFESTS.items()},
        "input_hashes": {p: rq1._sha(root / p) for p in inputs},
        "baseline_config": config, "seed": 24209199, "train_kwargs": kwargs,
        "output_dir": str(output), "runtime_dataset_yaml": str(data),
        "metric_selection": SELECTION,
    }


def runtime_contents(plan, dataset_root):
    """Materialize path strings only: no image/annotation enumeration or decoding."""
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError("Configured dataset_root does not exist")
    lists = frozen_inputs(plan["project_root"])
    data = Path(plan["runtime_dataset_yaml"])
    files = {data.parent / Path(MANIFESTS[key][0]).name:
             "".join(str(dataset_root.joinpath(*PurePosixPath(p).parts)) + "\n" for p in lines)
             for key, lines in lists.items()}
    config = {key: str(data.parent / Path(spec[0]).name) for key, spec in MANIFESTS.items()}
    config["names"] = NAMES
    files[data] = yaml.safe_dump(config, sort_keys=False)
    return files


def prepare_runtime(plan, dataset_root):
    files = runtime_contents(plan, dataset_root)
    # Never silently repair altered runtime inputs on restart.
    for path, content in files.items():
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise RuntimeError(f"Existing runtime input differs; inspect manually: {path}")
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="\n") as file:
                file.write(content)
    return {str(path): rq1._sha(path) for path in files}


def pretrained_identity(root):
    """Require existing RQ1 evidence; never establish a new RQ1 identity here."""
    binding = Path(root) / "runs/rq1/pretrained_model.json"
    if not binding.is_file():
        raise FileNotFoundError("RQ1 pretrained_model.json is required to prove the original weight identity")
    actual = rq1.model_provenance(root, "yolov8n.pt")
    if rq1._read_json(binding) != actual:
        raise RuntimeError("Pretrained file differs from the original RQ1 weight binding")
    return actual


def selected_epoch(csv_path, expected_epochs):
    """Compare the stored decimal strings, not an unrounded trainer fitness."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
    if len(rows) != expected_epochs or not rows:
        raise ValueError("Incomplete development metric history")
    scores = []
    for epoch, row in enumerate(rows, 1):
        try:
            score = Decimal(row[METRIC])
            valid = Decimal(row["epoch"]) == epoch and score.is_finite() and 0 <= score <= 1
        except (KeyError, InvalidOperation):
            valid = False
        if not valid:
            raise ValueError("Invalid/noncontiguous development metric history")
        scores.append(score)
    index = max(range(len(scores)), key=lambda i: scores[i])
    return {"epoch": index + 1, "development_map50_95": str(scores[index])}


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as file:
        before = os.fstat(file.fileno())
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(file.fileno())
    signature = lambda st: (st.st_size, st.st_mtime_ns, st.st_ino)
    if (not after.st_size or signature(before) != signature(after)
            or signature(after) != signature(path.stat())):
        raise RuntimeError("Checkpoint is empty or changed while hashing")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": after.st_size}


class CheckpointSelector:
    """on_model_save runs after CSV and last.pt writes; retain only strict improvements."""

    def __init__(self, output):
        self.output = Path(output).resolve()
        self.seen = 0
        self.winner = None
        self.candidate = self.output / "weights/selection_candidate.pt"

    def on_model_save(self, trainer):
        if Path(trainer.save_dir).resolve() != self.output:
            raise RuntimeError("Unexpected training output directory")
        epoch = trainer.epoch + 1
        if epoch != self.seen + 1:
            raise RuntimeError("Missing or repeated checkpoint callback")
        selection = selected_epoch(self.output / "results.csv", epoch)
        last = self.output / "weights/last.pt"
        if Path(trainer.last).resolve() != last:
            raise RuntimeError("Unexpected current-epoch checkpoint path")
        if selection["epoch"] == epoch:
            shutil.copyfile(last, self.candidate)
            self.winner = {**selection, "checkpoint": file_identity(self.candidate)}
        self.seen = epoch

    def freeze(self, epochs):
        selection = selected_epoch(self.output / "results.csv", epochs)
        if (self.seen != epochs or self.winner is None
                or any(self.winner[k] != v for k, v in selection.items())
                or file_identity(self.candidate) != self.winner["checkpoint"]):
            raise RuntimeError("Selected checkpoint does not match complete stored development history")
        target = self.output / "weights/selected.pt"
        with self.candidate.open("rb") as source, target.open("xb") as dest:
            shutil.copyfileobj(source, dest)
        identity = file_identity(target)
        if identity["sha256"] != self.winner["checkpoint"]["sha256"]:
            raise RuntimeError("Selected checkpoint copy changed")
        return {**selection, "checkpoint": identity}


def verify_inputs(plan):
    root = Path(plan["project_root"])
    info = rq1.git_provenance(root)
    if info["git_dirty"]:
        raise RuntimeError("RQ2 execution/scientific inputs must be committed and clean:\n" + info["git_input_status"])
    current = build_plan(root, info)
    if any(current[k] != plan[k] for k in current):
        raise RuntimeError("RQ2 execution configuration/Git state changed")
    if pretrained_identity(root) != plan["pretrained_model"]:
        raise RuntimeError("Original pretrained file changed before training/reuse")
    paths = load_paths(root)
    if paths.project_root != root or str(paths.dataset_root) != plan["dataset_root"]:
        raise RuntimeError("Local project/dataset roots changed")
    expected = runtime_contents(plan, paths.dataset_root)
    if any(not p.is_file() or p.read_bytes() != text.encode("utf-8") for p, text in expected.items()):
        raise RuntimeError("Runtime train/dev inputs changed")
    if {str(p): rq1._sha(p) for p in expected} != plan["runtime_hashes"]:
        raise RuntimeError("Runtime train/dev hashes changed")


def execute(plan, train_call):
    """Injectable training adapter for synthetic tests; fail closed after interruption."""
    verify_inputs(plan)
    output = Path(plan["output_dir"])
    sidecar = output.with_name(output.name + ".provenance.json")
    if sidecar.exists():
        receipt = rq1._read_json(sidecar)
        if receipt.get("plan") != plan or receipt.get("status") != "complete":
            raise RuntimeError("Existing run is incomplete or different; no automatic restart/overwrite")
        chosen = receipt["selected"]
        expected_checkpoint = output / "weights/selected.pt"
        if (chosen["checkpoint"]["path"] != str(expected_checkpoint.resolve())
                or file_identity(expected_checkpoint) != chosen["checkpoint"]
                or selected_epoch(output / "results.csv", 100) != {k: chosen[k] for k in ("epoch", "development_map50_95")}
                or rq1._sha(output / "results.csv") != receipt["results_sha256"]
                or rq1._sha(output / "args.yaml") != receipt["args_sha256"]):
            raise RuntimeError("Frozen completed-run artifacts changed; refusing reuse")
        return receipt
    if output.exists():
        raise RuntimeError("Run directory already exists without complete provenance; inspect manually")
    receipt = {"status": "running", "plan": plan, "environment": rq1._environment(),
               "started_utc": datetime.now(timezone.utc).isoformat()}
    rq1._write_json(sidecar, receipt, exclusive=True)
    try:
        selector = CheckpointSelector(output)
        train_call(plan, selector)
        verify_inputs(plan)
        args = yaml.safe_load((output / "args.yaml").read_text(encoding="utf-8"))
        for key, value in {"model": plan["pretrained_model"]["path"], **plan["train_kwargs"]}.items():
            if args.get(key) != value:
                raise RuntimeError(f"Actual training argument changed: {key}")
        receipt.update(status="complete", selected=selector.freeze(100),
                       actual_training_config=args,
                       results_sha256=rq1._sha(output / "results.csv"),
                       args_sha256=rq1._sha(output / "args.yaml"),
                       completed_utc=datetime.now(timezone.utc).isoformat())
        rq1._write_json(sidecar, receipt)
        return receipt
    except BaseException:
        receipt["status"] = "incomplete"
        rq1._write_json(sidecar, receipt)
        raise


def _train(plan, selector):
    # Only the guarded DICC worker reaches this import. Never call in CPU tests.
    from ultralytics import YOLO

    def guard(trainer):
        if Path(trainer.save_dir).resolve() != Path(plan["output_dir"]):
            raise RuntimeError("Ultralytics changed the output directory")
        for key, value in {"model": plan["pretrained_model"]["path"], **plan["train_kwargs"]}.items():
            if getattr(trainer.args, key, None) != value:
                raise RuntimeError(f"Unexpected training argument before fitting: {key}")
        verify_inputs(plan)

    model = YOLO(plan["pretrained_model"]["path"])
    model.add_callback("on_pretrain_routine_start", guard)
    model.add_callback("on_model_save", selector.on_model_save)
    model.train(**plan["train_kwargs"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "train", "_worker"))
    parser.add_argument("--dicc", action="store_true")
    args = parser.parse_args(argv)
    root = find_project_root()
    if args.command == "plan":
        print(rq1._json(build_plan(root)))
        return
    if not args.dicc or os.name == "nt":
        raise RuntimeError("Training is DICC-only; --dicc is required and Windows execution is prohibited")
    if args.command == "_worker":
        if os.environ.get("TRUSTPCB_RQ2_WORKER") != "1" or os.environ.get("PYTHONHASHSEED") != "24209199":
            raise RuntimeError("Use the guarded train launcher")
        import json
        plan = json.loads(os.environ["TRUSTPCB_RQ2_PLAN"])
        execute(plan, _train)
        return
    info = rq1.git_provenance(root)
    if info["git_dirty"]:
        raise RuntimeError("Commit/resolve RQ2 execution inputs before training:\n" + info["git_input_status"])
    plan = build_plan(root, info)
    plan["pretrained_model"] = pretrained_identity(root)
    paths = load_paths(root)
    if paths.project_root != root:
        raise ValueError("Configured project_root must be the executing repository")
    parent = Path(plan["output_dir"]).parent
    parent.mkdir(parents=True, exist_ok=True)
    lock = parent / ".runner.lock"
    with lock.open("x", encoding="utf-8") as file:
        file.write(str(os.getpid()))
    try:
        plan["dataset_root"] = str(paths.dataset_root)
        plan["runtime_hashes"] = prepare_runtime(plan, paths.dataset_root)
        verify_inputs(plan)
        env = dict(os.environ, PYTHONHASHSEED="24209199", TRUSTPCB_RQ2_WORKER="1",
                   TRUSTPCB_RQ2_PLAN=rq1._json(plan), PYTHONPATH=str(root / "src"))
        subprocess.run([sys.executable, "-B", "-m", "trustpcb.rq2_train", "_worker", "--dicc"],
                       cwd=root, env=env, check=True)
        def require_completed(*_):
            raise RuntimeError("Worker returned without a verified completed run")
        receipt = execute(plan, require_completed)
        print(rq1._json(receipt["selected"]))
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
