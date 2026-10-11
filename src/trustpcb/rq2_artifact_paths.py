"""Legacy import alias for the canonical RQ2 artifact-path implementation."""

import sys

from .rq2 import artifact_paths as _implementation

sys.modules[__name__] = _implementation
