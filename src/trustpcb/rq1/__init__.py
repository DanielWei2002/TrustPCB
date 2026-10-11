"""RQ1 package with explicit compatibility for established module interfaces."""

import sys as _sys
from types import ModuleType as _ModuleType

from . import workflow
from .workflow import (
    SEEDS, SPLITS, BASELINE_FILE, EXECUTION_INPUTS, APPROVED_BASELINE, METRICS,
    SELECTION, SOURCE_FILES,
    _json, _sha, _write_json, _read_json, _environment,
    git_provenance, require_clean_inputs, resolve_pretrained_model,
    model_provenance, bind_pretrained_model, verify_pretrained_model,
    build_plan, select_runs, _sidecar, _identity, _runtime_hashes,
    read_metrics, _artifacts, inspect_run, _yolo_train, execute_one,
    _worker_command, run_sequential, extract_results, main,
    find_project_root, generate_runtime_configs, load_paths, subprocess,
)

__all__ = (
    "SEEDS", "SPLITS", "BASELINE_FILE", "EXECUTION_INPUTS", "APPROVED_BASELINE",
    "METRICS", "SELECTION", "SOURCE_FILES",
    "_json", "_sha", "_write_json", "_read_json", "_environment",
    "git_provenance", "require_clean_inputs", "resolve_pretrained_model",
    "model_provenance", "bind_pretrained_model", "verify_pretrained_model",
    "build_plan", "select_runs", "_sidecar", "_identity", "_runtime_hashes",
    "read_metrics", "_artifacts", "inspect_run", "_yolo_train", "execute_one",
    "_worker_command", "run_sequential", "extract_results", "main",
    "find_project_root", "generate_runtime_configs", "load_paths", "subprocess",
)


class _CompatibilityModule(_ModuleType):
    """Keep package patches and workflow globals bound to the same interface.

    Simple function re-exports would leave patches of rq1.git_provenance invisible
    to functions defined in workflow.py. Forward only the explicit export list;
    package metadata and unrelated attributes retain normal module behavior.
    """

    def __getattribute__(self, name):
        if name in super().__getattribute__("__all__"):
            return getattr(super().__getattribute__("workflow"), name)
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in self.__all__:
            setattr(self.workflow, name, value)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self.__all__:
            delattr(self.workflow, name)
        super().__delattr__(name)


_sys.modules[__name__].__class__ = _CompatibilityModule
