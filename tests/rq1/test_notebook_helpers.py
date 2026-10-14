"""Synthetic fixtures only; never execute notebook dataset loops or writers."""

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest

import numpy as np

from trustpcb.rq1 import dataset_audit as audit
from trustpcb.rq1 import similarity_split as split
from trustpcb.rq1.archived_sources import read_notebook

REPO = Path(__file__).resolve().parents[2]


def cell(name, index):
    notebook = read_notebook(REPO, name)
    return "".join(notebook["cells"][index]["source"])


class NotebookHelperTests(unittest.TestCase):
    def test_pair_check_stems_and_missing_paths(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            audit.check_pairs([Path("a.jpg"), Path("b.png")], [Path("a.txt"), Path("c.txt")], "fixture")
        self.assertIn("Images without labels: 1", output.getvalue())
        self.assertIn("Labels without images: 1", output.getvalue())

    def test_pixel_hash_includes_dimensions_and_rgb_identity(self):
        data = bytes(range(6))
        self.assertEqual(audit.rgb_pixel_digest(2, 1, data), hashlib.sha256(b"2x1|RGB|" + data).hexdigest())
        self.assertNotEqual(audit.rgb_pixel_digest(2, 1, data), audit.rgb_pixel_digest(1, 2, data))

    def test_similarity_constant_correlation_is_nan(self):
        mae, pearson = audit.calculate_similarity(np.zeros((3, 3)), np.ones((3, 3)))
        self.assertEqual(mae, 1.0)
        self.assertTrue(np.isnan(pearson))

    def test_shift_direction_and_empty_overlap(self):
        a = np.arange(100, dtype=float).reshape(10, 10)
        b = np.zeros_like(a)
        b[:, :-1] = a[:, 1:]
        mae, pearson, overlap = audit.compare_at_shift(a, b, 1, 0)
        self.assertEqual(mae, 0)
        self.assertAlmostEqual(pearson, 1)
        self.assertEqual(overlap, 0.9)
        self.assertIsNone(audit.compare_at_shift(a, b, 10, 0))

    def test_translation_overlap_and_first_tie(self):
        a = np.ones((20, 20))
        best = audit.best_translation_match(a, a, max_shift=1)
        self.assertEqual((best["dx"], best["dy"]), (-1, -1))
        self.assertGreaterEqual(best["overlap_fraction"], 0.85)

    def test_original_array_functions_on_synthetic_inputs(self):
        namespace = {"np": np}
        for index in (9, 12):
            source = cell("01_dataset_structure_audit.ipynb", index)
            nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                     and n.name in {"calculate_similarity", "compare_at_shift", "best_translation_match"}]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "pure_notebook_fixture", "exec"), namespace)
        a = np.arange(400, dtype=np.float32).reshape(20, 20) / 400
        b = np.roll(a, 1, axis=1)
        np.testing.assert_allclose(audit.calculate_similarity(a, b), namespace["calculate_similarity"](a, b))
        for dx, dy in [(-1, 0), (0, 1), (1, -1)]:
            np.testing.assert_allclose(audit.compare_at_shift(a, b, dx, dy),
                                       namespace["compare_at_shift"](a, b, dx, dy))
        self.assertEqual(audit.best_translation_match(a, b, 1), namespace["best_translation_match"](a, b, 1))

    def test_connected_components_order_and_transitive_pairs(self):
        self.assertEqual(split.similarity_groups([("c", "b"), ("a", "b"), ("y", "z"), ("a", "b")]),
                         [["a", "b", "c"], ["y", "z"]])
        self.assertEqual(split.similarity_groups([]), [])

    def test_original_allocation_search_and_stable_tie(self):
        counts = {i: np.array([1, 1]) for i in range(1, 11)}
        base = np.array([10, 10])
        target = np.array([50.0, 50.0])
        actual = split.select_group_allocation(counts, base, base, target, target)
        source = cell("02_similarity_aware_split.ipynb", 16)
        source = source[source.index("group_ids = sorted"):source.index('print("=== BEST')]
        from itertools import combinations
        namespace = dict(np=np, combinations=combinations, group_annotation_counts=counts,
                         base_train_counts=base, base_val_counts=base,
                         target_train_pct=target, target_val_pct=target)
        exec(compile(source, "pure_allocation_fixture", "exec"), namespace)
        expected = namespace["best_allocation"]
        for key in actual:
            np.testing.assert_equal(actual[key], expected[key])
        self.assertEqual(actual["train_groups"], tuple(range(1, 10)))
        np.testing.assert_array_equal(base, [10, 10])


if __name__ == "__main__":
    unittest.main()
