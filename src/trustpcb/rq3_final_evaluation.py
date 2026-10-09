"""Legacy import and CLI for the canonical RQ3 final-evaluation workflow."""

import sys

from trustpcb.rq3 import final_evaluation as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
