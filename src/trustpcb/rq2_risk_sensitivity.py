"""Legacy import and CLI for the canonical RQ2 risk-sensitivity workflow."""

import sys

from trustpcb.rq2 import risk_sensitivity as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
