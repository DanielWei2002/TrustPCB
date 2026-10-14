"""Tiny synthetic fixtures only; no research data, model execution or GPU work."""

import ast
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

import numpy as np

from trustpcb.rq1 import archived_sources as archive
from trustpcb.rq1 import audit_workflow as audit
from trustpcb.rq1 import split_workflow as split
from trustpcb.rq1 import preliminary_baseline as baseline
from trustpcb.rq1 import reconstruction

REPO = Path(__file__).resolve().parents[2]


class ArchiveTests(unittest.TestCase):
    def test_all_three_originals_validate_from_immutable_history(self):
        records = json.loads((REPO / 'docs/rq1/notebook_archive.json').read_text())['sources']
        self.assertEqual(len(records), 3)
        for record in records:
            data = archive.read_source(REPO, record)
            self.assertEqual(hashlib.sha256(data).hexdigest(), record['sha256'])
            self.assertIn('cells', json.loads(data))

    def test_mixed_newline_transform_preserves_exact_old_hash(self):
        blob, checkout = b'one\ntwo\nthree\n', b'one\r\ntwo\nthree\r\n'
        record = dict(kind='git_blob', path='old.ipynb', commit='a' * 40,
                      blob_sha256=archive.digest(blob), sha256=archive.digest(checkout),
                      checkout_transform={'mode': 'lf_to_crlf_except', 'lf_lines': [1]})
        with patch.object(archive.subprocess, 'check_output', return_value=blob):
            self.assertEqual(archive.read_source(REPO, record), checkout)
            record['blob_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'blob changed'):
                archive.read_source(REPO, record)

    def test_missing_history_and_original_hash_mismatch_fail(self):
        record = archive.notebook_record(REPO, '01_dataset_structure_audit.ipynb')
        with patch.object(archive.subprocess, 'check_output', side_effect=subprocess.CalledProcessError(1, 'git')):
            with self.assertRaises(subprocess.CalledProcessError): archive.read_source(REPO, record)
        record['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'source changed'): archive.read_source(REPO, record)

    def test_mutable_revision_and_traversal_rejected(self):
        record = archive.notebook_record(REPO, '01_dataset_structure_audit.ipynb')
        record['commit'] = 'main'
        with self.assertRaises(ValueError): archive.read_source(REPO, record)
        record['path'] = '../secret'
        with self.assertRaises(ValueError): archive.read_source(REPO, record)


class ConvertedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output_capture = contextlib.redirect_stdout(io.StringIO())
        self.output_capture.__enter__()
        self.addCleanup(self.output_capture.__exit__, None, None, None)

    def make_dataset(self, suffix='.png', pairs=1):
        from PIL import Image
        dataset = self.root / 'dataset'
        for partition in ('train', 'val'):
            (dataset / 'images' / partition).mkdir(parents=True)
            (dataset / 'labels' / partition).mkdir(parents=True)
            for i in range(pairs):
                # Same nonconstant image at both sides; strictly tiny 20x20 data.
                pixels = np.arange(400, dtype=np.uint8).reshape(20, 20)
                Image.fromarray(pixels).save(dataset / 'images' / partition / f'{partition}{i}{suffix}')
                (dataset / 'labels' / partition / f'{partition}{i}.txt').write_text(
                    '\n'.join(f'{c} 0.5 0.5 0.2 0.2' for c in range(9)) + '\n')
        return dataset

    def test_complete_audit_on_two_synthetic_images_stops_before_manual_decisions(self):
        dataset = self.make_dataset()
        dest = self.root / 'audit'
        with patch.object(audit, 'require_dicc'):
            state = audit.run(dataset, dest, self.root / 'checkout', dicc=True)
        self.assertEqual(len(state['final_shortlist']), 1)
        self.assertEqual(len(state['duplicate_groups']), 1)
        self.assertEqual(state['valid_boxes'], 18)
        self.assertEqual(len(list(dest.glob('*.png'))), 2)
        self.assertTrue((dest / 'manual_decisions_required.csv').is_file())
        self.assertFalse((dest / 'cross_split_high_confidence_near_duplicates.csv').exists())
        self.assertFalse((dest / 'dataset_audit_summary.csv').exists())

    def test_annotation_validation_keeps_overflow_and_error_categories(self):
        dataset = self.make_dataset()
        label = dataset / 'labels/train/train0.txt'
        label.write_text('0 0.01 0.5 0.1 0.2\n\nbad line\nx y z p q\n0.5 0.5 0.5 0.2 0.2\n0 0.5 0.5 0 0.1\n0 2 0.5 0.1 0.1\n0 0.5 0.5 2 0.1\n')
        result = audit.validate_annotations([('train', dataset / 'images/train/train0.png', label)])
        self.assertEqual(result['valid_boxes'], 1)
        self.assertEqual(len(result['issues']), 7)
        self.assertAlmostEqual(result['outside_boxes'][0]['max_overflow_px'], .8)

    def test_phash_and_display_alignment_match_original_functions(self):
        import cv2
        dataset = self.make_dataset()
        original = archive.read_notebook(REPO, '01_dataset_structure_audit.ipynb')
        namespace = {'cv2': cv2, 'np': np}
        for index, name in [(8, 'compute_phash'), (13, 'align_for_display')]:
            tree = ast.parse(''.join(original['cells'][index]['source']))
            nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), 'synthetic_reference', 'exec'), namespace)
        path = dataset / 'images/train/train0.png'
        np.testing.assert_array_equal(audit.compute_phash(path), namespace['compute_phash'](path))
        a, b = np.arange(1200, dtype=np.uint8).reshape(20, 20, 3), np.ones((15, 15, 3), dtype=np.uint8)
        for actual, expected in zip(audit.align_for_display(a, b, -1, 2), namespace['align_for_display'](a, b, -1, 2)):
            np.testing.assert_array_equal(actual, expected)

    def test_manual_decisions_require_exact_filename_identity(self):
        import pandas as pd
        shortlist = pd.DataFrame({'train_file': ['a.jpg', 'b.jpg'], 'val_file': ['c.jpg', 'd.jpg']})
        reviewed, confirmed = audit.record_manual_review(shortlist, {('b.jpg', 'd.jpg'): 'B', ('a.jpg', 'c.jpg'): 'A'})
        self.assertEqual(list(reviewed.pair_id), [1, 2])
        self.assertEqual(list(confirmed.train_file), ['a.jpg'])
        for invalid in [{}, {('a.jpg', 'c.jpg'): 'A'}, {('a.jpg', 'c.jpg'): '', ('b.jpg', 'd.jpg'): 'B'}]:
            with self.assertRaises(ValueError): audit.record_manual_review(shortlist, invalid)

    def test_summary_requires_explicit_rows_and_never_overwrites(self):
        path = self.root / 'summary.csv'
        audit.write_audit_summary([('fixture metric', 2)], path)
        self.assertIn('fixture metric,2', path.read_text())
        with self.assertRaises(FileExistsError): audit.write_audit_summary([], path)

    def test_split_stages_on_ten_tiny_groups_reuse_original_search(self):
        import pandas as pd
        dataset = self.make_dataset('.jpg', pairs=11)
        state = dict(TRAIN_IMAGES=dataset/'images/train', VAL_IMAGES=dataset/'images/val',
                     TRAIN_LABELS=dataset/'labels/train', VAL_LABELS=dataset/'labels/val')
        reconstruction.apply_stage(state, split.supplied_inventory)
        state['confirmed_pairs'] = pd.DataFrame({'train_file': [f'train{i}.jpg' for i in range(10)],
                                                  'val_file': [f'val{i}.jpg' for i in range(10)]})
        for function in (split.construct_groups, split.group_classes, split.supplied_distribution,
                         split.feasible_allocations, split.allocate, split.compare_distribution):
            reconstruction.apply_stage(state, function)
        self.assertEqual(state['best_allocation']['train_groups'], tuple(range(1, 10)))
        state['SPLIT_DIR'] = self.root / 'splits'
        reconstruction.apply_stage(state, split.write_manifests)
        reconstruction.apply_stage(state, split.validate_split)
        self.assertFalse(state['validation_passed'])  # never claim 22 images satisfy the frozen 10,259 audit
        self.assertEqual(len(state['new_train_names']), 19)
        self.assertEqual(len(state['new_val_names']), 3)
        self.assertEqual(len(list(state['SPLIT_DIR'].iterdir())), 5)

    def test_historical_pair_assumptions_are_enforced(self):
        train = [f't{i}.jpg' for i in range(19)]
        val = [f'v{i}.jpg' for i in range(19)]
        groups = [list(pair) for pair in zip(train, val)]
        split.validate_historical_groups(groups, train, val)
        for bad in [groups[:-1], [groups[0] + groups[1]] + groups[2:]]:
            with self.assertRaises(ValueError): split.validate_historical_groups(bad, train, val)
        with self.assertRaises(ValueError): split.validate_historical_groups(groups, train + ['extra.png'], val)

    def test_preliminary_options_match_historical_constants(self):
        notebook = archive.read_notebook(REPO, '03_yolo_baseline.ipynb')
        tree = ast.parse(''.join(notebook['cells'][12]['source']))
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BASELINE_CONFIG' for t in n.targets))
        for node in ast.walk(assignment.value):
            if isinstance(node, ast.Name): self.assertEqual(node.id, 'EXPERIMENT_SEED')
        expected = eval(compile(ast.Expression(assignment.value), 'historical_configuration', 'eval'), {'EXPERIMENT_SEED': 24209199})
        self.assertEqual(baseline.APPROVED_BASELINE, expected)
        for run in ('smoke_test', 'supplied', 'similarity_aware'):
            options = baseline.training_options(run, self.root, self.root / 'v1')
            self.assertEqual(options['epochs'], 2 if run == 'smoke_test' else 100)
            self.assertEqual(options['seed'], 24209199)
            self.assertFalse(options['exist_ok'])
            self.assertNotIn('model', options)

    def test_preliminary_dataset_and_class_checks_on_tiny_or_mocked_inputs(self):
        from collections import Counter
        dataset = self.make_dataset('.jpg')
        directory = self.root / 'splits'
        directory.mkdir()
        for name, entry in [('supplied_train', 'train0.jpg'), ('supplied_val', 'val0.jpg'),
                             ('similarity_aware_train', 'train0.jpg'), ('similarity_aware_val', 'val0.jpg')]:
            (directory / (name + '.txt')).write_text(entry + '\n')
        self.assertEqual(len(baseline.verify_split_inputs(dataset, directory)['splits']), 4)
        with patch.object(baseline, 'count_classes', side_effect=[Counter({0: 746}), Counter({0: 169})]):
            self.assertEqual(baseline.verify_class_mapping(dataset)['class_mapping'], {0: 'SH'})
        with patch.object(baseline, 'count_classes', return_value=Counter({0: 1})):
            self.assertEqual(baseline.verify_class_mapping(dataset)['class_mapping'], {})

    def test_runtime_config_validation_uses_tiny_counts_and_external_checker(self):
        dataset = self.make_dataset('.jpg')
        directory = self.root / 'configs'
        directory.mkdir()
        for split_name in ('supplied', 'similarity_aware'):
            for partition in ('train', 'val'):
                (directory / f'{split_name}_{partition}_images.txt').write_text(
                    str(dataset / 'images' / partition / f'{partition}0.jpg') + '\n')
        fake = types.ModuleType('ultralytics.data.utils')
        fake.check_det_dataset = lambda path: {'nc': 9, 'names': {i: str(i) for i in range(9)}}
        with patch.dict(sys.modules, {'ultralytics': types.ModuleType('ultralytics'),
                                     'ultralytics.data': types.ModuleType('ultralytics.data'),
                                     'ultralytics.data.utils': fake}):
            baseline.validate_runtime_configs(directory, {i: str(i) for i in range(9)},
                                               {'supplied': (1, 1), 'similarity_aware': (1, 1)})
            (dataset / 'labels/train/train0.txt').unlink()
            with self.assertRaises(AssertionError):
                baseline.validate_runtime_configs(directory, {i: str(i) for i in range(9)},
                                                   {'supplied': (1, 1), 'similarity_aware': (1, 1)})

    def test_training_dispatch_with_fake_model_only_and_no_overwrite(self):
        import yaml
        from trustpcb import dataset_config
        project = self.root / 'checkout'
        config = project / 'configs/experiments/yolov8n_preliminary_baseline_v1.yaml'
        config.parent.mkdir(parents=True)
        config.write_text(yaml.safe_dump(baseline.APPROVED_BASELINE))
        (project / 'yolov8n.pt').write_bytes(b'synthetic checkpoint; never loaded')
        runtime = self.root / 'supplied.yaml'
        runtime.write_text('names: {0: synthetic}\n')
        paths = dataset_config.ProjectPaths(project, self.root)
        seen = []
        fake = types.ModuleType('ultralytics')
        fake.YOLO = lambda model: types.SimpleNamespace(train=lambda **options: seen.append((model, options)))
        with contextlib.ExitStack() as stack:
            for name in ('require_dicc', 'verify_environment', 'verify_split_inputs',
                         'verify_class_mapping', 'validate_runtime_configs', 'seed_environment'):
                stack.enter_context(patch.object(baseline, name))
            stack.enter_context(patch.object(dataset_config, 'generate_runtime_configs', return_value={'supplied': runtime}))
            stack.enter_context(patch.dict(sys.modules, {'ultralytics': fake}))
            baseline.train('smoke_test', paths, self.root / 'new-version', dicc=True)
            self.assertEqual(seen[0][1]['epochs'], 2)
            self.assertEqual(seen[0][0], str((project / 'yolov8n.pt').resolve()))
            (self.root / 'new-version/smoke_test').mkdir()
            with self.assertRaises(FileExistsError): baseline.train('smoke_test', paths, self.root / 'new-version', dicc=True)
            self.assertEqual(len(seen), 1)

    def test_comparison_preserves_first_tie_and_difference_direction(self):
        from trustpcb.rq1.workflow import METRICS
        for name, value in [('supplied', .4), ('similarity_aware', .6)]:
            directory = self.root / name
            directory.mkdir()
            with (directory / 'results.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['epoch', *METRICS.values()])
                for epoch in range(1, 101): writer.writerow([epoch, *([value] * 4)])
        table = baseline.compare_results(self.root, self.root / 'comparison.csv')
        self.assertEqual(table.iloc[0]['Supplied'], 1)
        self.assertTrue(np.isnan(table.iloc[0]['Difference']))
        self.assertAlmostEqual(table.iloc[1]['Difference'], .2)

    def test_windows_and_missing_dicc_flag_are_rejected(self):
        with patch.object(reconstruction.os, 'name', 'nt'):
            with self.assertRaises(RuntimeError): reconstruction.require_dicc(True)
        with patch.object(reconstruction.os, 'name', 'posix'):
            with self.assertRaises(RuntimeError): reconstruction.require_dicc(False)

    def test_protected_or_existing_output_destinations_rejected(self):
        for name in ('outputs/rq1/new', 'data/splits', 'runs/new', 'configs/local/new'):
            with self.assertRaises(ValueError): reconstruction.fresh_output(self.root / name, self.root)
        dest = self.root / 'fresh'
        reconstruction.fresh_output(dest, self.root / 'checkout')
        self.assertFalse(json.loads((dest / 'RECONSTRUCTION.json').read_text())['historical_execution'])
        with self.assertRaises(FileExistsError): reconstruction.fresh_output(dest, self.root / 'checkout')


if __name__ == '__main__':
    unittest.main()
