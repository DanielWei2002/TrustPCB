"""Path-only tests: temporary empty dataset directories, never real images."""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trustpcb.dataset_config import generate_runtime_configs, load_paths


class DatasetConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "empty dataset"
        self.dataset.mkdir()
        shutil.copytree(REPO / "configs" / "datasets", self.root / "configs" / "datasets")
        self.local = self.root / "configs" / "local"
        self.local.mkdir()
        self.config = self.local / "paths.yaml"
        self.config.write_text("project_root: .\ndataset_root: empty dataset\n", encoding="utf-8")

    def test_complete_ordered_round_trip(self):
        paths = load_paths(self.root)
        self.assertEqual(paths.dataset_root, self.dataset.resolve())
        result = generate_runtime_configs(paths)
        self.assertEqual(len(list((self.local / "datasets").iterdir())), 6)
        for split, generated in result.items():
            original = yaml.safe_load((self.root / "configs/datasets" / generated.name).read_text())
            runtime = yaml.safe_load(generated.read_text())
            self.assertEqual(original["names"], runtime["names"])
            for partition in ("train", "val"):
                source = self.root / "configs/datasets" / original[partition]
                entries = Path(runtime[partition]).read_text().splitlines()
                relative = [Path(p).relative_to(paths.dataset_root).as_posix() for p in entries]
                self.assertEqual(source.read_text().splitlines(), relative)
        self.assertEqual(list(self.dataset.iterdir()), [])

    def test_unsorted_entries_and_duplicates_are_not_rewritten(self):
        source = self.root / "configs/datasets/supplied_train_images.txt"
        entries = ["images/val/z.jpg", "images/train/a.jpg", "images/val/z.jpg"]
        source.write_text("\n".join(entries) + "\n", encoding="utf-8")
        paths = load_paths(self.root)
        generate_runtime_configs(paths)
        output = (self.local / "datasets" / source.name).read_text().splitlines()
        self.assertEqual([Path(p).relative_to(paths.dataset_root).as_posix() for p in output], entries)

    def test_missing_and_malformed_settings(self):
        self.config.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "paths.yaml"):
            load_paths(self.root)
        for content in ("[", "[]", "project_root: .", "project_root: 5\ndataset_root: x"):
            self.config.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                load_paths(self.root)

    def test_missing_dataset(self):
        self.config.write_text("project_root: .\ndataset_root: absent\n", encoding="utf-8")
        with self.assertRaisesRegex(FileNotFoundError, "dataset_root"):
            load_paths(self.root)

    def test_invalid_entries_fail_before_writing(self):
        source = self.root / "configs/datasets/supplied_train_images.txt"
        for entry in ("", "../escape.jpg", "/images/train/a.jpg", "images/train/../a.jpg", "images//train/a.jpg"):
            source.write_text(entry + "\n", encoding="utf-8")
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                generate_runtime_configs(load_paths(self.root))
            self.assertFalse((self.local / "datasets").exists())


if __name__ == "__main__":
    unittest.main()
