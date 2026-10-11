"""Preserve the python -m trustpcb.rq1 entry point."""

from .workflow import main


if __name__ == "__main__":
    raise SystemExit(main())
