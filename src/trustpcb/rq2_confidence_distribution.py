"""Legacy import and CLI for the canonical RQ2 confidence-distribution workflow."""

import sys

from trustpcb.rq2 import confidence_distribution as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
