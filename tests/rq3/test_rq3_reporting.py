"""Synthetic plotting and reporting-provenance checks; no research execution."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from trustpcb.rq3 import reporting as report


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "outputs/rq3"
        self.output.mkdir(parents=True)
        (self.root / "source.csv").write_bytes(b"synthetic\r\n")
        (self.root / "large.csv").write_bytes(b"tiny reference fixture")
        (self.output / "copy.csv").write_bytes(b"synthetic\r\n")
        manifest = {"artifacts": [{"path": "copy.csv", "sha256": report.digest(b"synthetic\r\n"),
                                   "sources": [report.source_record(self.root, "source.csv")]}],
                    "references": [report.source_record(self.root, "large.csv")]}
        (self.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def pyplot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.addCleanup(plt.close, "all")
        return plt

    def test_manifest_verifies_exact_bytes(self):
        self.assertEqual(report.verify(self.root), 1)

    def test_changed_source_and_output_are_rejected(self):
        for path, message in ((self.root / "source.csv", "source changed"),
                              (self.output / "copy.csv", "artifact changed")):
            original = path.read_bytes()
            path.write_bytes(b"synthetic\n")
            with self.assertRaisesRegex(ValueError, message):
                report.verify(self.root)
            path.write_bytes(original)

    def test_reference_and_unlisted_files_block_verification(self):
        reference = self.root / "large.csv"
        original = reference.read_bytes()
        reference.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Referenced evidence changed"):
            report.verify(self.root)
        reference.write_bytes(original)
        (self.output / "user.txt").write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "Unlisted"):
            report.build(self.root)
        self.assertEqual((self.output / "user.txt").read_bytes(), b"keep")

    def test_duplicate_interval_identity_rejected(self):
        row = {"method": "fixture", "budget_percent": "5", "metric": "workload", "lower_95": ".01", "upper_95": ".1"}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            report.interval_index([row, dict(row, budget_percent="5.0")])

    def test_curve_keeps_undefined_zero_and_right_endpoint_steps(self):
        method = "raw_confidence_risk"
        rows = [{"method": method, "coverage": "0", "empirical_risk": ""},
                {"method": method, "coverage": ".4", "empirical_risk": ".2"},
                {"method": method, "coverage": "1", "empirical_risk": ".7"}]
        x, y = report.curve_points(rows, method)
        self.assertEqual(x, [0, .4, 1])
        self.assertTrue(math.isnan(y[0]))
        self.assertEqual(y[1:], [.2, .7])
        fig = report.plot_curve(self.pyplot(), rows, "synthetic")
        self.assertEqual(fig.axes[0].lines[0].get_drawstyle(), "steps-pre")
        self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), x)

    def test_referral_uses_actual_test_workload_and_saved_intervals(self):
        rows = [{"method": m, "budget_percent": str(b), "workload": ".031",
                 "development_achieved_workload": ".05", "error_capture": ".12"}
                for m in report.METHODS for b in (5, 10, 20)]
        intervals = {(m, b, metric): bounds for m in report.METHODS for b in (5, 10, 20)
                     for metric, bounds in (("workload", (.01, .04)), ("error_capture", (.08, .18)))}
        fig = report.plot_referral(self.pyplot(), rows, intervals, "synthetic")
        for ax in fig.axes:
            self.assertEqual(list(ax.lines[0].get_xdata()), [.031])
            self.assertEqual(list(ax.collections[0].get_segments()[0][:, 0]), [.01, .04])

    def test_aurc_uses_saved_value_and_absolute_endpoints(self):
        method = "raw_confidence_risk"
        fig = report.plot_aurc(self.pyplot(), [{"method": method, "aurc": ".7"}],
                              {(method, None, "aurc"): (.6, .65)}, "synthetic")
        self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [.7])
        self.assertEqual(list(fig.axes[0].collections[0].get_segments()[0][:, 0]), [.6, .65])

    def test_paired_difference_sign_is_not_reversed(self):
        rows = [{"method_A": "weighted_trustpcb_risk", "method_B": "raw_confidence_risk", "budget_percent": "5",
                 "metric": "error_capture", "observed_difference": "-.02", "lower_95": "-.03", "upper_95": ".01"}]
        fig = report.plot_paired(self.pyplot(), rows, "error_capture", "synthetic")
        self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [-.02])
        self.assertEqual(list(fig.axes[0].collections[0].get_segments()[0][:, 0]), [-.03, .01])

    def test_runtime_uses_stored_summary_without_reaggregation(self):
        summary = {"pipelines": {p: {"mean_seconds": mean, "standard_deviation_seconds": .5}
                                 for p, mean in (("baseline", 7.0), ("trustpcb", 700.0))}}
        rows = [{"pipeline": p, "repetition": "1", "seconds": seconds}
                for p, seconds in (("baseline", "6"), ("trustpcb", "650"))]
        fig = report.plot_runtime(self.pyplot(), summary, rows)
        self.assertEqual([p.get_height() for p in fig.axes[0].patches], [7, 700])
        self.assertEqual(list(fig.axes[1].lines[0].get_ydata()), [6])
        self.assertEqual(fig.axes[1].get_yscale(), "log")

    def test_reporting_import_is_lazy_and_does_not_import_science(self):
        script = "import sys; import trustpcb.rq3.reporting; assert not {'torch','ultralytics','matplotlib','trustpcb.rq3.selective_review','trustpcb.rq3.computational_efficiency','trustpcb.rq3.final_evaluation'} & sys.modules.keys()"
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
