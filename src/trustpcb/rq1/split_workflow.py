"""Reconstructed notebook 02 workflow, not its historical executed program.

The original 19 disjoint JPG pairs and 9/10 allocation convention are retained.
A different group structure is rejected rather than silently reinterpreted.
"""
from pathlib import Path
import numpy as np
from collections import Counter
from trustpcb.rq1.similarity_split import similarity_groups as connected_groups, select_group_allocation
from trustpcb.rq1.reconstruction import apply_stage, inventory, fresh_output, require_dicc


def validate_historical_groups(groups, train_names, val_names):
    """Reject populations for which the notebook's pair-count arithmetic is unsafe."""
    train, val = set(train_names), set(val_names)
    if train & val or len(train) != len(train_names) or len(val) != len(val_names):
        raise ValueError('Historical split requires unique, disjoint filenames')
    if len(groups) != 19 or any(len(group) != 2 or len(set(group) & train) != 1
                               or len(set(group) & val) != 1 for group in groups):
        raise ValueError('Historical allocation requires 19 disjoint train/val pairs')
    members = [name for group in groups for name in group]
    if len(set(members)) != len(members) or any(not name.endswith('.jpg') for name in train | val):
        raise ValueError('Historical label exclusion assumes unique JPG members')


def run(dataset_root, audit_tables, output, project_root, *, dicc=False):
    require_dicc(dicc)
    import pandas as pd
    root = Path(dataset_root)
    state = dict(TRAIN_IMAGES=root / 'images/train', VAL_IMAGES=root / 'images/val',
                 TRAIN_LABELS=root / 'labels/train', VAL_LABELS=root / 'labels/val',
                 OUTPUT_DIR=Path(audit_tables))
    for function in (supplied_inventory, load_confirmed_pairs, construct_groups):
        apply_stage(state, function)
    validate_historical_groups(state['similarity_groups'], state['train_names'], state['val_names'])
    for function in (group_classes, supplied_distribution, feasible_allocations, allocate, compare_distribution):
        apply_stage(state, function)
    state['SPLIT_DIR'] = fresh_output(output, project_root)
    apply_stage(state, write_manifests)
    apply_stage(state, validate_split)
    if not state['validation_passed']:
        raise ValueError('Historical split validation failed; generated files are not approved evidence')
    return state


def main(argv=None):
    import argparse
    from trustpcb.dataset_config import load_paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--audit-tables', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dicc', action='store_true')
    args = parser.parse_args(argv)
    require_dicc(args.dicc)
    paths = load_paths(args.project_root)
    run(paths.dataset_root, args.audit_tables, args.output, paths.project_root, dicc=True)


def read_label_classes(label_path):
    classes = []

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        parts = line.split()
        classes.append(int(float(parts[0])))

    return classes


def count_classes(label_files):
    counts = Counter()

    for label_path in label_files:
        classes = read_label_classes(label_path)
        counts.update(classes)

    return counts


def supplied_inventory(TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS):
    """Reconstructed from notebook 02 cell 4; explicit inputs, historical arithmetic."""
    import pandas as pd
    # 1.1 Reconstruct supplied train-validation split

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}



    # Load supplied split

    items = inventory(TRAIN_IMAGES.parents[1])
    train_images, val_images = items['train_images'], items['val_images']
    train_labels, val_labels = items['train_labels'], items['val_labels']

    # Basic counts

    total_images = len(train_images) + len(val_images)

    train_percentage = len(train_images) / total_images * 100
    val_percentage = len(val_images) / total_images * 100

    print("=== SUPPLIED SPLIT INVENTORY ===")

    print("\nTraining:")
    print("Images:", len(train_images))
    print("Labels:", len(train_labels))

    print("\nValidation:")
    print("Images:", len(val_images))
    print("Labels:", len(val_labels))

    print("\nTotal images:", total_images)

    print(
        f"Train / validation proportion: "
        f"{train_percentage:.2f}% / {val_percentage:.2f}%"
    )

    # Check filename uniqueness

    train_names = [p.name for p in train_images]
    val_names = [p.name for p in val_images]

    print("\n=== FILENAME CHECK ===")

    print(
        "Unique training filenames:",
        len(set(train_names))
    )

    print(
        "Unique validation filenames:",
        len(set(val_names))
    )

    # Check filename overlap across supplied splits

    cross_split_name_overlap = sorted(
        set(train_names) & set(val_names)
    )

    print(
        "Same filename appearing in both splits:",
        len(cross_split_name_overlap)
    )

    if cross_split_name_overlap:
        print("\nExamples:")
        for name in cross_split_name_overlap[:10]:
            print(" ", name)

    # Check image-label pairing

    train_image_stems = {p.stem for p in train_images}
    train_label_stems = {p.stem for p in train_labels}

    val_image_stems = {p.stem for p in val_images}
    val_label_stems = {p.stem for p in val_labels}

    print("\n=== IMAGE-LABEL PAIRING ===")

    print(
        "Train images without labels:",
        len(train_image_stems - train_label_stems)
    )

    print(
        "Train labels without images:",
        len(train_label_stems - train_image_stems)
    )

    print(
        "Validation images without labels:",
        len(val_image_stems - val_label_stems)
    )

    print(
        "Validation labels without images:",
        len(val_label_stems - val_image_stems)
    )
    return {'train_images': train_images, 'val_images': val_images, 'train_labels': train_labels, 'val_labels': val_labels, 'train_names': train_names, 'val_names': val_names, 'total_images': total_images}


def load_confirmed_pairs(OUTPUT_DIR):
    """Reconstructed from notebook 02 cell 6; explicit inputs, historical arithmetic."""
    import pandas as pd
    # 1.2 Load confirmed cross-split near-duplicate pairs

    CONFIRMED_PAIRS_PATH = (
        OUTPUT_DIR
        / "cross_split_high_confidence_near_duplicates.csv"
    )

    confirmed_pairs = pd.read_csv(CONFIRMED_PAIRS_PATH)

    print("=== CONFIRMED CROSS-SPLIT PAIRS ===")
    print("File:", CONFIRMED_PAIRS_PATH)
    print("Number of confirmed pairs:", len(confirmed_pairs))

    print("\nColumns:")
    print(list(confirmed_pairs.columns))

    print("\nFirst 10 pairs:")
    print(
        confirmed_pairs[
            [
                "train_file",
                "val_file",
                "aligned_mae",
                "aligned_pearson"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nUnique affected training images:")
    print(confirmed_pairs["train_file"].nunique())

    print("\nUnique affected validation images:")
    print(confirmed_pairs["val_file"].nunique())
    return {'confirmed_pairs': confirmed_pairs}


def construct_groups(confirmed_pairs, train_names, val_names):
    """Reconstructed from notebook 02 cell 8; explicit inputs, historical arithmetic."""
    import pandas as pd
    # 1.3 Construct similarity groups

    from collections import defaultdict

    # Collect all confirmed pair relationships

    pair_edges = list(
        zip(
            confirmed_pairs["train_file"],
            confirmed_pairs["val_file"]
        )
    )

    similarity_groups = connected_groups(pair_edges)
    visited = {name for group in similarity_groups for name in group}

    # Map supplied split membership

    train_name_set = set(train_names)
    val_name_set = set(val_names)

    group_records = []

    for group_id, group in enumerate(similarity_groups, start=1):

        train_members = [
            name for name in group
            if name in train_name_set
        ]

        val_members = [
            name for name in group
            if name in val_name_set
        ]

        group_records.append({
            "group_id": group_id,
            "group_size": len(group),
            "train_count": len(train_members),
            "val_count": len(val_members),
            "members": ", ".join(group)
        })

    similarity_groups_df = pd.DataFrame(group_records)

    print("=== SIMILARITY GROUP SUMMARY ===")

    print("Confirmed pair relationships:", len(pair_edges))
    print("Similarity groups:", len(similarity_groups))
    print(
        "Unique images in similarity groups:",
        len(visited)
    )

    print("\nGroup size distribution:")
    print(
        similarity_groups_df["group_size"]
        .value_counts()
        .sort_index()
    )

    print("\n=== GROUP DETAILS ===")

    print(
        similarity_groups_df[
            [
                "group_id",
                "group_size",
                "train_count",
                "val_count",
                "members"
            ]
        ].to_string(index=False)
    )
    return {'similarity_groups': similarity_groups, 'similarity_groups_df': similarity_groups_df, 'train_name_set': train_name_set, 'val_name_set': val_name_set}


def group_classes(similarity_groups, train_name_set, val_name_set, train_labels, val_labels):
    """Reconstructed from notebook 02 cell 10; explicit inputs, historical arithmetic."""
    import pandas as pd
    # 1.4 Analyse class composition of similarity groups

    from collections import Counter


    # Build filename-to-label lookup

    train_label_lookup = {
        p.stem: p
        for p in train_labels
    }

    val_label_lookup = {
        p.stem: p
        for p in val_labels
    }

    # Analyse each similarity group

    class_records = []

    for group_id, group in enumerate(similarity_groups, start=1):

        train_member = next(
            name for name in group
            if name in train_name_set
        )

        val_member = next(
            name for name in group
            if name in val_name_set
        )

        train_classes = read_label_classes(
            train_label_lookup[Path(train_member).stem]
        )

        val_classes = read_label_classes(
            val_label_lookup[Path(val_member).stem]
        )

        train_class_set = sorted(set(train_classes))
        val_class_set = sorted(set(val_classes))

        class_records.append({
            "group_id": group_id,
            "train_file": train_member,
            "val_file": val_member,
            "train_boxes": len(train_classes),
            "val_boxes": len(val_classes),
            "train_classes": train_class_set,
            "val_classes": val_class_set,
            "same_class_set": train_class_set == val_class_set
        })

    group_class_df = pd.DataFrame(class_records)

    print("=== SIMILARITY GROUP CLASS COMPOSITION ===")

    print(
        group_class_df.to_string(index=False)
    )

    print("\n=== CLASS-SET AGREEMENT ===")

    print(
        group_class_df["same_class_set"]
        .value_counts()
    )

    # Count all annotations represented by the affected images

    affected_train_class_counts = Counter()
    affected_val_class_counts = Counter()

    for row in group_class_df.itertuples(index=False):

        train_classes = read_label_classes(
            train_label_lookup[Path(row.train_file).stem]
        )

        val_classes = read_label_classes(
            val_label_lookup[Path(row.val_file).stem]
        )

        affected_train_class_counts.update(train_classes)
        affected_val_class_counts.update(val_classes)

    print("\n=== AFFECTED TRAINING ANNOTATIONS BY CLASS ===")

    for class_id in range(9):
        print(
            f"Class {class_id}: "
            f"{affected_train_class_counts[class_id]}"
        )

    print("\n=== AFFECTED VALIDATION ANNOTATIONS BY CLASS ===")

    for class_id in range(9):
        print(
            f"Class {class_id}: "
            f"{affected_val_class_counts[class_id]}"
        )
    return {'group_class_df': group_class_df, 'train_label_lookup': train_label_lookup, 'val_label_lookup': val_label_lookup}


def supplied_distribution(train_labels, val_labels):
    """Reconstructed from notebook 02 cell 12; explicit inputs, historical arithmetic."""
    import pandas as pd
    # 1.5 Calculate supplied split class distribution


    train_class_counts = count_classes(train_labels)
    val_class_counts = count_classes(val_labels)

    train_total_boxes = sum(train_class_counts.values())
    val_total_boxes = sum(val_class_counts.values())

    class_distribution_records = []

    for class_id in range(9):

        train_count = train_class_counts[class_id]
        val_count = val_class_counts[class_id]

        train_pct = train_count / train_total_boxes * 100
        val_pct = val_count / val_total_boxes * 100

        class_distribution_records.append({
            "class_id": class_id,
            "train_count": train_count,
            "train_pct": train_pct,
            "val_count": val_count,
            "val_pct": val_pct,
            "pct_difference": abs(train_pct - val_pct)
        })

    supplied_class_df = pd.DataFrame(class_distribution_records)

    print("=== SUPPLIED SPLIT CLASS DISTRIBUTION ===")

    print(
        supplied_class_df.round({
            "train_pct": 2,
            "val_pct": 2,
            "pct_difference": 2
        }).to_string(index=False)
    )

    print("\nTraining annotations:", train_total_boxes)
    print("Validation annotations:", val_total_boxes)

    print(
        "\nLargest class percentage difference:",
        f"{supplied_class_df['pct_difference'].max():.2f}%"
    )
    return {'supplied_class_df': supplied_class_df}


def feasible_allocations(similarity_groups, train_images, val_images):
    """Reconstructed from notebook 02 cell 14; explicit inputs, historical arithmetic."""
    import pandas as pd
    # 1.6 Check feasible similarity-group allocation

    n_groups = len(similarity_groups)

    original_train = len(train_images)
    original_val = len(val_images)

    affected_train = 19
    affected_val = 19

    unaffected_train = original_train - affected_train
    unaffected_val = original_val - affected_val

    print("=== SIMILARITY-AWARE SPLIT STRATEGY ===")

    print("Similarity groups:", n_groups)
    print("Images per group: 2")
    print("Images that must change split:", n_groups)

    print("\nOriginal split:")
    print("Training:", original_train)
    print("Validation:", original_val)

    print("\nUnaffected images:")
    print("Training:", unaffected_train)
    print("Validation:", unaffected_val)

    print("\n=== CLOSEST POSSIBLE 80/20 ALLOCATIONS ===")

    for groups_to_train in [9, 10]:

        groups_to_val = n_groups - groups_to_train

        new_train = unaffected_train + (groups_to_train * 2)
        new_val = unaffected_val + (groups_to_val * 2)

        total = new_train + new_val

        train_pct = new_train / total * 100
        val_pct = new_val / total * 100

        print(
            f"\n{groups_to_train} groups -> train, "
            f"{groups_to_val} groups -> validation"
        )

        print(
            f"Training: {new_train} ({train_pct:.2f}%)"
        )

        print(
            f"Validation: {new_val} ({val_pct:.2f}%)"
        )
    return {'unaffected_train': unaffected_train, 'unaffected_val': unaffected_val}


def allocate(group_class_df, train_label_lookup, val_label_lookup, confirmed_pairs, train_labels, val_labels, supplied_class_df, unaffected_train, unaffected_val, total_images):
    """Reconstructed from notebook 02 cell 16; explicit inputs, historical arithmetic."""
    import pandas as pd
    # 1.7 Select similarity group allocation

    from itertools import combinations

    # Get annotation counts for each similarity group

    group_annotation_counts = {}

    for row in group_class_df.itertuples(index=False):

        train_classes = read_label_classes(
            train_label_lookup[Path(row.train_file).stem]
        )

        val_classes = read_label_classes(
            val_label_lookup[Path(row.val_file).stem]
        )

        counts = Counter(train_classes + val_classes)

        group_annotation_counts[row.group_id] = np.array(
            [counts[class_id] for class_id in range(9)],
            dtype=int
        )

    # Get annotation counts from unaffected images

    affected_train_names = set(confirmed_pairs["train_file"])
    affected_val_names = set(confirmed_pairs["val_file"])

    unaffected_train_labels = [
        p for p in train_labels
        if f"{p.stem}.jpg" not in affected_train_names
    ]

    unaffected_val_labels = [
        p for p in val_labels
        if f"{p.stem}.jpg" not in affected_val_names
    ]

    unaffected_train_counts = count_classes(unaffected_train_labels)
    unaffected_val_counts = count_classes(unaffected_val_labels)

    base_train_counts = np.array(
        [unaffected_train_counts[i] for i in range(9)]
    )

    base_val_counts = np.array(
        [unaffected_val_counts[i] for i in range(9)]
    )

    target_train_pct = supplied_class_df["train_pct"].to_numpy()
    target_val_pct = supplied_class_df["val_pct"].to_numpy()

    best_allocation = select_group_allocation(group_annotation_counts, base_train_counts,
        base_val_counts, target_train_pct, target_val_pct)

    print("=== BEST SIMILARITY GROUP ALLOCATION ===")

    print(
        "Groups assigned to training:",
        best_allocation["train_groups"]
    )

    print(
        "Groups assigned to validation:",
        best_allocation["val_groups"]
    )

    print(
        "\nNumber of training groups:",
        len(best_allocation["train_groups"])
    )

    print(
        "Number of validation groups:",
        len(best_allocation["val_groups"])
    )

    print(
        "\nMean class percentage change:",
        f"{best_allocation['mean_pct_difference']:.4f}%"
    )

    print(
        "Largest class percentage change:",
        f"{best_allocation['max_pct_difference']:.4f}%"
    )

    # Calculate final image counts

    new_train_count = (
        unaffected_train
        + 2 * len(best_allocation["train_groups"])
    )

    new_val_count = (
        unaffected_val
        + 2 * len(best_allocation["val_groups"])
    )

    print("\nFinal image counts:")
    print(
        f"Training: {new_train_count} "
        f"({new_train_count / total_images * 100:.2f}%)"
    )

    print(
        f"Validation: {new_val_count} "
        f"({new_val_count / total_images * 100:.2f}%)"
    )
    return {'best_allocation': best_allocation, 'new_train_count': new_train_count, 'new_val_count': new_val_count}


def compare_distribution(best_allocation, supplied_class_df, new_train_count, new_val_count, total_images):
    """Reconstructed from notebook 02 cell 18; explicit inputs, historical arithmetic."""
    import pandas as pd
    # 1.8 Verify selected similarity-aware class distribution

    new_train_counts = best_allocation["train_counts"]
    new_val_counts = best_allocation["val_counts"]

    new_train_pct = (
        new_train_counts
        / new_train_counts.sum()
        * 100
    )

    new_val_pct = (
        new_val_counts
        / new_val_counts.sum()
        * 100
    )

    comparison_records = []

    for class_id in range(9):

        supplied_train_pct = supplied_class_df.loc[
            supplied_class_df["class_id"] == class_id,
            "train_pct"
        ].iloc[0]

        supplied_val_pct = supplied_class_df.loc[
            supplied_class_df["class_id"] == class_id,
            "val_pct"
        ].iloc[0]

        comparison_records.append({
            "class_id": class_id,
            "supplied_train_pct": supplied_train_pct,
            "new_train_pct": new_train_pct[class_id],
            "train_change": (
                new_train_pct[class_id]
                - supplied_train_pct
            ),
            "supplied_val_pct": supplied_val_pct,
            "new_val_pct": new_val_pct[class_id],
            "val_change": (
                new_val_pct[class_id]
                - supplied_val_pct
            )
        })

    split_comparison_df = pd.DataFrame(
        comparison_records
    )

    print("=== CLASS DISTRIBUTION COMPARISON ===")

    print(
        split_comparison_df.round(3).to_string(index=False)
    )

    print("\n=== MAXIMUM ABSOLUTE CHANGE ===")

    print(
        "Training:",
        f"{split_comparison_df['train_change'].abs().max():.4f} percentage points"
    )

    print(
        "Validation:",
        f"{split_comparison_df['val_change'].abs().max():.4f} percentage points"
    )

    print("\n=== FINAL SPLIT SIZE ===")

    print(
        f"Training: {new_train_count} "
        f"({new_train_count / total_images * 100:.2f}%)"
    )

    print(
        f"Validation: {new_val_count} "
        f"({new_val_count / total_images * 100:.2f}%)"
    )
    return {'split_comparison_df': split_comparison_df}


def write_manifests(SPLIT_DIR, train_images, val_images, train_names, val_names, similarity_groups, best_allocation):
    """Reconstructed from notebook 02 cell 20; explicit inputs, historical arithmetic."""
    import pandas as pd
    # 1.9 Create similarity-aware split manifests

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # Map every image filename to its original location

    image_path_lookup = {}

    for path in train_images:
        image_path_lookup[path.name] = path

    for path in val_images:
        image_path_lookup[path.name] = path

    # Start from the supplied split

    new_train_names = set(train_names)
    new_val_names = set(val_names)

    train_group_ids = set(best_allocation["train_groups"])
    val_group_ids = set(best_allocation["val_groups"])

    # Assign every similarity group entirely to one split

    for group_id, group in enumerate(similarity_groups, start=1):

        for image_name in group:
            new_train_names.discard(image_name)
            new_val_names.discard(image_name)

        if group_id in train_group_ids:
            new_train_names.update(group)

        elif group_id in val_group_ids:
            new_val_names.update(group)

        else:
            raise RuntimeError(
                f"Group {group_id} was not assigned."
            )

    # Sort manifests

    new_train_names = sorted(new_train_names)
    new_val_names = sorted(new_val_names)

    # Identify images that changed split

    moved_train_to_val = sorted(
        set(train_names) & set(new_val_names)
    )

    moved_val_to_train = sorted(
        set(val_names) & set(new_train_names)
    )

    print("=== SIMILARITY-AWARE SPLIT ===")

    print("Training images:", len(new_train_names))
    print("Validation images:", len(new_val_names))

    print("\nImages moved train -> validation:")
    print(len(moved_train_to_val))

    print("\nImages moved validation -> train:")
    print(len(moved_val_to_train))

    print("\nTotal images moved:")
    print(
        len(moved_train_to_val)
        + len(moved_val_to_train)
    )

    # Save filename manifests

    (SPLIT_DIR / "supplied_train.txt").write_text(
        "\n".join(sorted(train_names)) + "\n"
    )

    (SPLIT_DIR / "supplied_val.txt").write_text(
        "\n".join(sorted(val_names)) + "\n"
    )

    (SPLIT_DIR / "similarity_aware_train.txt").write_text(
        "\n".join(new_train_names) + "\n"
    )

    (SPLIT_DIR / "similarity_aware_val.txt").write_text(
        "\n".join(new_val_names) + "\n"
    )

    # Save complete split mapping

    manifest_records = []

    for image_name in sorted(set(train_names) | set(val_names)):

        original_split = (
            "train"
            if image_name in set(train_names)
            else "val"
        )

        similarity_split = (
            "train"
            if image_name in set(new_train_names)
            else "val"
        )

        manifest_records.append({
            "image": image_name,
            "supplied_split": original_split,
            "similarity_aware_split": similarity_split,
            "changed_split": original_split != similarity_split
        })

    split_manifest_df = pd.DataFrame(manifest_records)

    split_manifest_path = (
        SPLIT_DIR
        / "similarity_aware_split_manifest.csv"
    )

    split_manifest_df.to_csv(
        split_manifest_path,
        index=False
    )

    print("\n=== SAVED FILES ===")

    for path in [
        SPLIT_DIR / "supplied_train.txt",
        SPLIT_DIR / "supplied_val.txt",
        SPLIT_DIR / "similarity_aware_train.txt",
        SPLIT_DIR / "similarity_aware_val.txt",
        split_manifest_path
    ]:
        print(path)
    return {'new_train_names': new_train_names, 'new_val_names': new_val_names, 'split_manifest_df': split_manifest_df}


def validate_split(new_train_names, new_val_names, train_names, val_names, similarity_groups, split_manifest_df):
    """Reconstructed from notebook 02 cell 22; explicit inputs, historical arithmetic."""
    import pandas as pd
    # 1.10 Validate similarity-aware split

    new_train_set = set(new_train_names)
    new_val_set = set(new_val_names)

    all_original_images = set(train_names) | set(val_names)
    all_new_images = new_train_set | new_val_set

    # Check train-validation overlap

    split_overlap = new_train_set & new_val_set

    # Check missing or unexpected images

    missing_images = all_original_images - all_new_images
    unexpected_images = all_new_images - all_original_images

    # Check confirmed similarity groups

    cross_split_groups = []

    for group_id, group in enumerate(similarity_groups, start=1):

        group_train = [
            name for name in group
            if name in new_train_set
        ]

        group_val = [
            name for name in group
            if name in new_val_set
        ]

        if group_train and group_val:
            cross_split_groups.append({
                "group_id": group_id,
                "train_members": group_train,
                "val_members": group_val
            })

    # Count changed assignments

    changed_images = split_manifest_df[
        split_manifest_df["changed_split"]
    ]

    print("=== FINAL SPLIT VALIDATION ===")

    print("Training images:", len(new_train_set))
    print("Validation images:", len(new_val_set))
    print("Total images:", len(all_new_images))

    print(
        "\nTrain / validation proportion:",
        f"{len(new_train_set) / len(all_new_images) * 100:.2f}% / "
        f"{len(new_val_set) / len(all_new_images) * 100:.2f}%"
    )

    print("\nImages appearing in both splits:", len(split_overlap))
    print("Missing images:", len(missing_images))
    print("Unexpected images:", len(unexpected_images))

    print(
        "Confirmed similarity groups crossing splits:",
        len(cross_split_groups)
    )

    print(
        "Images with changed split assignment:",
        len(changed_images)
    )

    if cross_split_groups:
        print("\nCross-split similarity groups:")
        for group in cross_split_groups:
            print(group)

    # Final pass/fail

    validation_passed = (
        len(new_train_set) + len(new_val_set) == 10259
        and len(split_overlap) == 0
        and len(missing_images) == 0
        and len(unexpected_images) == 0
        and len(cross_split_groups) == 0
        and len(changed_images) == 19
    )

    print("\n=== VALIDATION RESULT ===")

    if validation_passed:
        print("PASS")
    else:
        print("FAIL")
    return {'validation_passed': validation_passed}


if __name__ == '__main__':
    main()
