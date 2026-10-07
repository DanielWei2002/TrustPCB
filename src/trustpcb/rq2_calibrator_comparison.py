"""Legacy import and CLI for the canonical RQ2 calibrator-comparison workflow."""

import sys

from trustpcb.rq2 import calibrator_comparison as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
