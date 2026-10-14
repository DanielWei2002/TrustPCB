"""Reconstructed notebook 03 preliminary baseline, not historical execution.

Uses the frozen baseline settings and shared runtime-config/metric helpers.
Training, validation and dataset inspection belong on DICC, even the two-epoch
historical smoke run. Ordinary imports never import torch or Ultralytics.
"""
from pathlib import Path
from trustpcb.rq1.workflow import APPROVED_BASELINE, read_metrics, resolve_pretrained_model
from trustpcb.rq1.reconstruction import require_dicc, fresh_output
from trustpcb.rq1.split_workflow import count_classes


def training_options(run, dataset_config_dir, version_run_dir):
    """Cells 12/14/16/18: one implementation of the three frozen configurations."""
    if run not in {'smoke_test', 'supplied', 'similarity_aware'}:
        raise ValueError('Unknown preliminary run')
    options = {k: v for k, v in APPROVED_BASELINE.items() if k != 'model'}
    split = 'supplied' if run == 'smoke_test' else run
    options.update(data=str(Path(dataset_config_dir) / f'dspcbsd_{split}.yaml'),
                   project=str(version_run_dir), name=run, exist_ok=False)
    if run == 'smoke_test':
        options['epochs'] = 2
    return options


def seed_environment():
    """Cell 12 seeding; call only on DICC immediately before a training stage."""
    import os
    import random
    import numpy as np
    import torch
    seed = APPROVED_BASELINE['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_environment(paths):
    """Cell 2 diagnostic; imports ML libraries only when explicitly called."""
    import sys
    import torch
    import ultralytics
    print('Python:', sys.executable)
    print('PyTorch:', torch.__version__, 'Ultralytics:', ultralytics.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0), 'CUDA runtime:', torch.version.cuda)
    for name, path in {'Project': paths.project_root, 'Dataset': paths.dataset_root,
                       'Splits': paths.project_root / 'data/splits',
                       'Configs': paths.project_root / 'configs', 'Runs': paths.project_root / 'runs'}.items():
        print(name, path.exists(), path)


def compare_results(version_run_dir, output_path):
    """Cell 20 first best-mAP50–95 row via workflow.read_metrics; no model evaluation.

    Shared validation requires complete 100-epoch logs. Incomplete preliminary
    logs are rejected instead of silently presenting them as completed evidence.
    """
    import pandas as pd
    root = Path(version_run_dir)
    best = {split: read_metrics(root / split / 'results.csv', 100)
            for split in ('supplied', 'similarity_aware')}
    fields = [('Best epoch', 'best_epoch'), ('Precision', 'precision'), ('Recall', 'recall'),
              ('mAP50', 'mAP@0.50'), ('mAP50-95', 'mAP@0.50:0.95')]
    table = pd.DataFrame({'Metric': [label for label, key in fields],
                          'Supplied': [best['supplied'][key] for label, key in fields],
                          'Similarity-aware': [best['similarity_aware'][key] for label, key in fields]})
    table['Difference'] = table['Similarity-aware'] - table['Supplied']
    table.loc[table.Metric == 'Best epoch', 'Difference'] = pd.NA
    with Path(output_path).open('x', encoding='utf-8', newline='') as stream:
        table.to_csv(stream, index=False)
    return table


def train(run, paths, version_run_dir, *, dicc=False):
    require_dicc(dicc)
    import yaml
    from trustpcb.dataset_config import generate_runtime_configs
    frozen = yaml.safe_load((paths.project_root / 'configs/experiments/yolov8n_preliminary_baseline_v1.yaml').read_text())
    if frozen != APPROVED_BASELINE:
        raise ValueError('Frozen preliminary baseline configuration differs')
    root = Path(version_run_dir).resolve()
    # Existing version directories may contain a different completed stage, but
    # never target the frozen runs tree or overwrite the requested run.
    project = paths.project_root.resolve()
    if root == project or project in root.parents or root in project.parents:
        raise ValueError('Use an explicit destination outside the checkout for reconstructed training')
    if root.exists() and not (root / 'RECONSTRUCTION.json').is_file():
        raise FileExistsError('Existing version directory is not a reconstruction destination')
    if (root / run).exists():
        raise FileExistsError(root / run)
    model = resolve_pretrained_model(project, APPROVED_BASELINE['model'])
    verify_environment(paths)
    verify_split_inputs(paths.dataset_root, project / 'data/splits')
    verify_class_mapping(paths.dataset_root)
    configs = generate_runtime_configs(paths)
    config_dir = configs['supplied'].parent
    names = yaml.safe_load(configs['supplied'].read_text())['names']
    validate_runtime_configs(config_dir, names)
    options = training_options(run, config_dir, root)
    if not root.exists():
        fresh_output(root, project)
    from ultralytics import YOLO
    seed_environment()
    return YOLO(str(model)).train(**options)


def main(argv=None):
    import argparse
    from trustpcb.dataset_config import load_paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('verify', 'smoke_test', 'supplied', 'similarity_aware', 'compare'))
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--version-run-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dicc', action='store_true')
    args = parser.parse_args(argv)
    require_dicc(args.dicc)
    if args.stage == 'compare':
        if args.version_run_dir is None or args.output is None:
            parser.error('compare requires --version-run-dir and a fresh --output directory')
        dest = fresh_output(args.output, args.project_root)
        compare_results(args.version_run_dir, dest / 'preliminary_baseline_comparison_v1.csv')
    else:
        paths = load_paths(args.project_root)
        if args.stage == 'verify':
            verify_environment(paths)
            verify_split_inputs(paths.dataset_root, paths.project_root / 'data/splits')
            verify_class_mapping(paths.dataset_root)
        else:
            if args.version_run_dir is None:
                parser.error('training requires --version-run-dir outside the checkout')
            train(args.stage, paths, args.version_run_dir, dicc=True)


def verify_split_inputs(DATASET_ROOT, SPLIT_DIR):
    """Notebook 03 cell 4 verification, retained without inference."""
    dataset_dirs = {
        "train_images": DATASET_ROOT / "images/train",
        "train_labels": DATASET_ROOT / "labels/train",
        "val_images": DATASET_ROOT / "images/val",
        "val_labels": DATASET_ROOT / "labels/val",
    }

    split_files = {
        "supplied_train": SPLIT_DIR / "supplied_train.txt",
        "supplied_val": SPLIT_DIR / "supplied_val.txt",
        "similarity_train": SPLIT_DIR / "similarity_aware_train.txt",
        "similarity_val": SPLIT_DIR / "similarity_aware_val.txt",
    }

    print("=== DATASET DIRECTORIES ===")

    for name, path in dataset_dirs.items():
        count = sum(1 for p in path.iterdir() if p.is_file())
        print(f"{name:18s}: {count:5d} | {path}")


    train_images = {p.stem for p in dataset_dirs["train_images"].glob("*.jpg")}
    train_labels = {p.stem for p in dataset_dirs["train_labels"].glob("*.txt")}
    val_images = {p.stem for p in dataset_dirs["val_images"].glob("*.jpg")}
    val_labels = {p.stem for p in dataset_dirs["val_labels"].glob("*.txt")}

    print("\n=== IMAGE-LABEL MATCHING ===")
    print("Train images without labels:", len(train_images - train_labels))
    print("Train labels without images:", len(train_labels - train_images))
    print("Val images without labels  :", len(val_images - val_labels))
    print("Val labels without images  :", len(val_labels - val_images))


    splits = {}

    for name, path in split_files.items():
        entries = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        splits[name] = entries

    print("\n=== SPLIT COUNTS ===")

    expected = {
        "supplied_train": 8208,
        "supplied_val": 2051,
        "similarity_train": 8207,
        "similarity_val": 2052,
    }

    for name, expected_count in expected.items():
        actual = len(splits[name])
        status = "OK" if actual == expected_count else "MISMATCH"
        print(f"{name:20s}: {actual:5d} | expected {expected_count:5d} | {status}")


    supplied_train = set(splits["supplied_train"])
    supplied_val = set(splits["supplied_val"])
    similarity_train = set(splits["similarity_train"])
    similarity_val = set(splits["similarity_val"])

    supplied_all = supplied_train | supplied_val
    similarity_all = similarity_train | similarity_val

    print("\n=== SPLIT INTEGRITY ===")
    print("Supplied train-val overlap        :", len(supplied_train & supplied_val))
    print("Similarity-aware train-val overlap:", len(similarity_train & similarity_val))
    print("Supplied total images             :", len(supplied_all))
    print("Similarity-aware total images     :", len(similarity_all))
    print("Same image universe               :", supplied_all == similarity_all)

    changed = supplied_train ^ similarity_train

    print("\n=== ASSIGNMENT CHANGES ===")
    print("Moved into similarity-aware train:", len(similarity_train - supplied_train))
    print("Moved into similarity-aware val  :", len(similarity_val - supplied_val))
    print("Images with changed assignment   :", len(changed))
    return {'splits': splits}


def verify_class_mapping(DATASET_ROOT):
    """Notebook 03 cell 6 verification, retained without inference."""
    from collections import Counter

    train_label_dir = DATASET_ROOT / "labels/train"
    val_label_dir = DATASET_ROOT / "labels/val"



    train_counts = count_classes(train_label_dir.glob("*.txt"))
    val_counts = count_classes(val_label_dir.glob("*.txt"))

    all_ids = sorted(set(train_counts) | set(val_counts))

    print("=== OBSERVED YOLO CLASS COUNTS ===")

    for class_id in all_ids:
        train = train_counts[class_id]
        val = val_counts[class_id]
        total = train + val

        print(
            f"ID {class_id}: "
            f"train={train:4d} | "
            f"val={val:4d} | "
            f"total={total:4d}"
        )


    published_counts = {
        "SH":   (746, 169, 915),
        "SP":   (3655, 929, 4584),
        "SC":   (1308, 285, 1593),
        "OP":   (1432, 338, 1770),
        "MB":   (1983, 546, 2529),
        "HB":   (2275, 608, 2883),
        "CS":   (2042, 448, 2490),
        "CFO":  (1409, 423, 1832),
        "BMFO": (1334, 346, 1680),
    }

    print("\n=== VERIFIED CLASS MAPPING ===")

    class_mapping = {}

    for class_id in all_ids:
        observed = (
            train_counts[class_id],
            val_counts[class_id],
            train_counts[class_id] + val_counts[class_id],
        )

        matches = [
            name
            for name, expected in published_counts.items()
            if observed == expected
        ]

        if len(matches) == 1:
            class_mapping[class_id] = matches[0]
            print(f"{class_id} -> {matches[0]} | counts match publication")
        else:
            print(f"{class_id} -> UNRESOLVED | observed={observed}")


    print("\n=== TOTAL ANNOTATIONS ===")
    print("Training  :", sum(train_counts.values()), "| expected: 16184")
    print("Validation:", sum(val_counts.values()), "| expected: 4092")
    print(
        "Total     :",
        sum(train_counts.values()) + sum(val_counts.values()),
        "| expected: 20276"
    )
    return {'class_mapping': class_mapping}


def validate_runtime_configs(DATASET_CONFIG_DIR, class_names, expected_counts=None):
    """Notebook 03 cell 9 verification, retained without inference."""
    from ultralytics.data.utils import check_det_dataset

    config_specs = {
        "supplied": {
            "yaml": DATASET_CONFIG_DIR / "dspcbsd_supplied.yaml",
            "train": DATASET_CONFIG_DIR / "supplied_train_images.txt",
            "val": DATASET_CONFIG_DIR / "supplied_val_images.txt",
            "expected_train": 8208,
            "expected_val": 2051,
        },
        "similarity_aware": {
            "yaml": DATASET_CONFIG_DIR / "dspcbsd_similarity_aware.yaml",
            "train": DATASET_CONFIG_DIR / "similarity_aware_train_images.txt",
            "val": DATASET_CONFIG_DIR / "similarity_aware_val_images.txt",
            "expected_train": 8207,
            "expected_val": 2052,
        },
    }
    if expected_counts is not None:
        for name, (train_count, val_count) in expected_counts.items():
            config_specs[name]['expected_train'] = train_count
            config_specs[name]['expected_val'] = val_count


    def read_image_paths(path):
        return [
            Path(line.strip())
            for line in path.read_text().splitlines()
            if line.strip()
        ]


    def get_label_path(image_path):
        parts = list(image_path.parts)
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")


    print("=== DATASET CONFIGURATION VALIDATION ===")

    for name, spec in config_specs.items():
        train_paths = read_image_paths(spec["train"])
        val_paths = read_image_paths(spec["val"])
        all_paths = train_paths + val_paths

        missing_images = [p for p in all_paths if not p.exists()]
        missing_labels = [
            get_label_path(p)
            for p in all_paths
            if not get_label_path(p).exists()
        ]

        data = check_det_dataset(str(spec["yaml"]))

        loaded_names = data["names"]
        if isinstance(loaded_names, list):
            loaded_names = {i: value for i, value in enumerate(loaded_names)}
        else:
            loaded_names = {int(k): v for k, v in loaded_names.items()}

        print(f"\n--- {name.upper()} ---")
        print("Train images      :", len(train_paths), "| expected:", spec["expected_train"])
        print("Val images        :", len(val_paths), "| expected:", spec["expected_val"])
        print("Total images      :", len(all_paths))
        print("Duplicate train   :", len(train_paths) - len(set(train_paths)))
        print("Duplicate val     :", len(val_paths) - len(set(val_paths)))
        print("Missing images    :", len(missing_images))
        print("Missing labels    :", len(missing_labels))
        print("Classes           :", data["nc"])
        print("Class names       :", loaded_names)

        assert len(train_paths) == spec["expected_train"]
        assert len(val_paths) == spec["expected_val"]
        assert len(all_paths) == spec['expected_train'] + spec['expected_val']
        assert len(train_paths) == len(set(train_paths))
        assert len(val_paths) == len(set(val_paths))
        assert not missing_images
        assert not missing_labels
        assert data["nc"] == 9
        assert loaded_names == class_names

        print("Validation status : PASSED")
    return {}


if __name__ == '__main__':
    main()
