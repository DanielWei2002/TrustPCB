"""Tiny synthetic reporting fixtures; no scientific evidence is recalculated."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from trustpcb.rq2 import reporting


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "outputs/rq2"
        self.output.mkdir(parents=True)
        (self.root / "source.csv").write_bytes(b"synthetic\r\n")
        (self.root / "reference.csv").write_bytes(b"reference\n")
        (self.output / "copy.csv").write_bytes(b"synthetic\r\n")
        self.record = {"path": "copy.csv", "sha256": reporting.digest(b"synthetic\r\n"),
                       "sources": [reporting.source_record(self.root, "source.csv")]}
        self.manifest = {"artifacts": [self.record],
                         "references": [reporting.source_record(self.root, "reference.csv")]}
        self.write_manifest()

    def write_manifest(self):
        (self.output / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_missing_numbers_remain_missing_and_zero_remains_zero(self):
        for value in (None, "", " ", "nan", "inf"):
            self.assertIsNone(reporting.number(value))
        self.assertEqual(reporting.number("0"), 0)
        self.assertEqual(reporting.number("-0.25"), -0.25)

    def test_incomplete_and_reversed_intervals_rejected(self):
        self.assertEqual(reporting.interval({}, "lo", "hi"), (None, None))
        for row in ({"lo": "0.1"}, {"lo": "0.3", "hi": "0.2"}):
            with self.assertRaises(ValueError):
                reporting.interval(row, "lo", "hi")

    def test_exact_bytes_and_references_verified(self):
        self.assertEqual(reporting.verify(self.root), 1)

    def test_source_changes_rejected(self):
        (self.root / "source.csv").write_bytes(b"synthetic\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            reporting.verify(self.root)

    def test_output_changes_rejected(self):
        (self.output / "copy.csv").write_bytes(b"edited")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            reporting.verify(self.root)

    def test_referenced_evidence_changes_rejected(self):
        (self.root / "reference.csv").write_bytes(b"edited")
        with self.assertRaisesRegex(ValueError, "Referenced evidence changed"):
            reporting.verify(self.root)

    def test_unlisted_file_blocks_rebuild_without_modification(self):
        path = self.output / "user-note.txt"
        path.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "Unlisted"):
            reporting.build(self.root)
        self.assertEqual(path.read_bytes(), b"keep")

    def test_paths_cannot_escape_output_root(self):
        for path in ("../original.csv", "/original.csv", "C:/original.csv", "..\\original.csv"):
            with self.assertRaises(ValueError):
                reporting.inside(self.root, path)

    def test_plot_uses_saved_points_and_absolute_interval_endpoints(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = [{"method": "fixture", "auroc": "0.8", "auprc": "",
                 "auroc_lower_95": "0.6", "auroc_upper_95": "0.75"}]
        # Observed point outside the saved interval must not be clipped or used
        # to replace its endpoints. No derived symmetric error bars are used.
        fig = reporting.plot_metrics(plt, rows, "synthetic")
        self.addCleanup(plt.close, fig)
        self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [0.8])
        self.assertEqual(list(fig.axes[0].collections[0].get_segments()[0][:, 0]), [0.6, 0.75])
        self.assertEqual(len(fig.axes[1].lines), 0)
        self.assertEqual(fig.axes[1].texts[0].get_text(), "undefined")

    def test_paired_plot_keeps_saved_sign_and_bounds(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = [{"method_A": "A", "method_B": "B", "observed_delta_auroc": "-0.02",
                 "lower_95": "-0.03", "upper_95": "-0.01"}]
        fig = reporting.plot_differences(plt, rows, "synthetic")
        self.addCleanup(plt.close, fig)
        self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [-0.02])
        self.assertEqual(list(fig.axes[0].collections[0].get_segments()[0][:, 0]), [-0.03, -0.01])

    def test_reporting_import_does_not_load_ml_or_plotting(self):
        script = "import sys; import trustpcb.rq2.reporting; assert not {'torch','ultralytics','matplotlib'} & sys.modules.keys()"
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
