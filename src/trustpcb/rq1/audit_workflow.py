"""Reconstructed notebook 01 workflow; NOT the originally executed program.

Real-data stages belong on DICC. Numerical kernels are reused from dataset_audit.
CLI writes only to a new explicit destination; historical evidence is never updated.
Manual decisions and the fixed historical summary are not inferred by this workflow.
"""
from pathlib import Path
from trustpcb.rq1.dataset_audit import (check_pairs, rgb_pixel_digest, calculate_similarity,
                                       compare_at_shift, best_translation_match)
from trustpcb.rq1.reconstruction import apply_stage, inventory, fresh_output, require_dicc


def record_manual_review(final_shortlist, decisions):
    """Cell 14 with explicit, filename-keyed human input instead of fixed labels.

    A means confirmed; B means related but unconfirmed in the historical review.
    No old pair-number decisions are silently applied to a newly ordered shortlist.
    """
    keys = list(zip(final_shortlist.train_file, final_shortlist.val_file))
    if len(set(keys)) != len(keys) or set(decisions) != set(keys):
        raise ValueError('Manual decisions must match every shortlisted filename pair exactly')
    if any(label not in {'A', 'B'} for label in decisions.values()):
        raise ValueError('Every manual decision must explicitly be A or B')
    reviewed = final_shortlist.reset_index(drop=True).copy()
    reviewed.insert(0, 'pair_id', range(1, len(reviewed) + 1))
    reviewed['manual_label'] = [decisions[key] for key in keys]
    return reviewed, reviewed[reviewed.manual_label == 'A'].copy()


def write_audit_summary(rows, output_path):
    """Cell 15 table schema, with explicit reviewed rows rather than fixed results.

    The old cell hard-coded results and a conclusion, not an aggregation algorithm.
    Its exact historical table remains in outputs/tables and Git history.
    """
    import pandas as pd
    with Path(output_path).open('x', encoding='utf-8', newline='') as stream:
        pd.DataFrame(rows, columns=['metric', 'value']).to_csv(stream, index=False)


def run(dataset_root, output, project_root, *, dicc=False):
    require_dicc(dicc)
    # Fail on missing runtime dependencies before creating an output destination.
    import cv2
    import pandas
    import PIL.Image
    import tqdm
    import matplotlib
    matplotlib.use('Agg')
    state = inventory(dataset_root)
    destination = fresh_output(output, project_root)
    state['OUTPUT_DIR'] = destination
    state['REVIEW_DIR'] = destination
    for partition in ('train', 'val'):
        check_pairs(state[partition + '_images'], state[partition + '_labels'], partition)
        print('Example images:', state[partition + '_images'][:5])
        print('Example labels:', state[partition + '_labels'][:5])
    for function in (inspect_images, validate_annotations, describe_boxes, exact_duplicates,
                     screen_phash, verify_candidates, shortlist, show_tier_a,
                     align_candidates, show_final_review):
        apply_stage(state, function)
    template = state['final_shortlist'][['train_file', 'val_file']].copy()
    template['manual_label'] = ''
    template.to_csv(destination / 'manual_decisions_required.csv', index=False)
    print('STOP: human review is required; no labels, fixed summary or conclusions inferred.')
    return state


def main(argv=None):
    import argparse
    import pandas as pd
    from trustpcb.dataset_config import load_paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('audit', 'review'))
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dicc', action='store_true')
    parser.add_argument('--shortlist', type=Path)
    parser.add_argument('--decisions', type=Path)
    args = parser.parse_args(argv)
    require_dicc(args.dicc)
    if args.stage == 'audit':
        paths = load_paths(args.project_root)
        run(paths.dataset_root, args.output, paths.project_root, dicc=True)
    else:
        if args.shortlist is None or args.decisions is None:
            parser.error('review needs --shortlist and --decisions')
        decisions = pd.read_csv(args.decisions, keep_default_na=False)
        keys = list(zip(decisions.train_file, decisions.val_file))
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate manual decision pair')
        reviewed, confirmed = record_manual_review(pd.read_csv(args.shortlist),
                                                   dict(zip(keys, decisions.manual_label)))
        dest = fresh_output(args.output, args.project_root)
        reviewed.to_csv(dest / 'cross_split_manual_review.csv', index=False)
        confirmed.to_csv(dest / 'cross_split_high_confidence_near_duplicates.csv', index=False)
        print('Confirmed pairs:', len(confirmed))
        print('Unique affected train/validation images:', confirmed.train_file.nunique(), confirmed.val_file.nunique())


def compute_phash(path):
    import cv2
    import numpy as np
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Could not read: {path}")

    # Standard pHash-style preprocessing
    img = cv2.resize(
        img,
        (32, 32),
        interpolation=cv2.INTER_AREA
    )

    img = np.float32(img)

    # 2D Discrete Cosine Transform
    dct = cv2.dct(img)

    # Low-frequency 8x8 block
    low_freq = dct[:8, :8].flatten()

    # Exclude DC coefficient when computing threshold
    median = np.median(low_freq[1:])

    bits = low_freq > median

    # Ignore DC coefficient itself
    bits[0] = False

    # Pack 64 bits into 8 bytes
    return np.packbits(bits.astype(np.uint8))


def load_for_similarity(path, size=128):
    import cv2
    import numpy as np
    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise RuntimeError(f"Cannot read {path}")

    img = cv2.resize(
        img,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    return img.astype(np.float32) / 255.0


def load_rgb(path):
    import cv2
    import numpy as np
    img = cv2.imread(str(path))

    if img is None:
        raise RuntimeError(f"Cannot read {path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_gray_original(path):
    import cv2
    import numpy as np
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Could not read: {path}")

    return img.astype(np.float32) / 255.0


def align_for_display(train_img, val_img, dx, dy):
    import cv2
    import numpy as np
    """
    Produce overlapping train/validation crops using the
    shift found during the alignment audit.
    """

    if train_img.shape != val_img.shape:
        val_img = cv2.resize(
            val_img,
            (train_img.shape[1], train_img.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    h, w = train_img.shape[:2]

    if dx >= 0:
        ax1, ax2 = dx, w
        bx1, bx2 = 0, w - dx
    else:
        ax1, ax2 = 0, w + dx
        bx1, bx2 = -dx, w

    if dy >= 0:
        ay1, ay2 = dy, h
        by1, by2 = 0, h - dy
    else:
        ay1, ay2 = 0, h + dy
        by1, by2 = -dy, h

    a = train_img[ay1:ay2, ax1:ax2]
    b = val_img[by1:by2, bx1:bx2]

    return a, b


def inspect_images(all_images):
    """Reconstructed from notebook 01 cell 4; explicit inputs, historical arithmetic."""
    from collections import Counter
    from PIL import Image
    from tqdm import tqdm

    unreadable = []
    dimensions = Counter()
    modes = Counter()

    for split, path in tqdm(all_images, desc="Checking images"):
        try:
            # Verify that the image file is readable
            with Image.open(path) as img:
                img.verify()

            # Reopen after verify() to inspect properties
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode

            dimensions[(width, height)] += 1
            modes[mode] += 1

        except Exception as e:
            unreadable.append({
                "split": split,
                "file": path.name,
                "error": str(e)
            })

    print("\n=== IMAGE READABILITY ===")
    print("Total images checked:", len(all_images))
    print("Readable images:", len(all_images) - len(unreadable))
    print("Unreadable images:", len(unreadable))

    print("\n=== IMAGE DIMENSIONS ===")
    for size, count in dimensions.most_common():
        print(f"{size[0]} x {size[1]}: {count}")

    print("\n=== IMAGE MODES ===")
    for mode, count in modes.most_common():
        print(f"{mode}: {count}")

    if unreadable:
        print("\n=== UNREADABLE IMAGE EXAMPLES ===")
        for item in unreadable[:10]:
            print(item)
    return {'unreadable': unreadable, 'dimensions': dimensions, 'modes': modes}


def validate_annotations(all_pairs):
    """Reconstructed from notebook 01 cell 5; explicit inputs, historical arithmetic."""
    from collections import Counter
    from PIL import Image
    from tqdm import tqdm

    issues = []
    class_counts = Counter()

    total_boxes = 0
    valid_boxes = 0
    outside_boxes = []

    for split, image_path, label_path in tqdm(
        all_pairs,
        desc="Validating YOLO annotations"
    ):
        with Image.open(image_path) as img:
            img_w, img_h = img.size

        lines = label_path.read_text().splitlines()

        for line_no, line in enumerate(lines, start=1):

            line = line.strip()

            if not line:
                issues.append(
                    (split, label_path.name, line_no, "empty_line", line)
                )
                continue

            parts = line.split()

            if len(parts) != 5:
                issues.append(
                    (
                        split,
                        label_path.name,
                        line_no,
                        f"expected_5_values_got_{len(parts)}",
                        line,
                    )
                )
                continue

            try:
                cls_raw, xc, yc, bw, bh = map(float, parts)
            except ValueError:
                issues.append(
                    (split, label_path.name, line_no, "non_numeric", line)
                )
                continue

            total_boxes += 1

            # Class ID must be an integer
            if not cls_raw.is_integer():
                issues.append(
                    (
                        split,
                        label_path.name,
                        line_no,
                        "non_integer_class",
                        line,
                    )
                )
                continue

            cls = int(cls_raw)

            # Width and height must be positive
            if bw <= 0 or bh <= 0:
                issues.append(
                    (
                        split,
                        label_path.name,
                        line_no,
                        "non_positive_box_size",
                        line,
                    )
                )
                continue

            # Basic YOLO normalized values
            if not (0 <= xc <= 1 and 0 <= yc <= 1):
                issues.append(
                    (
                        split,
                        label_path.name,
                        line_no,
                        "center_outside_normalized_range",
                        line,
                    )
                )
                continue

            if not (0 < bw <= 1 and 0 < bh <= 1):
                issues.append(
                    (
                        split,
                        label_path.name,
                        line_no,
                        "size_outside_normalized_range",
                        line,
                    )
                )
                continue

            # Convert YOLO box to image-relative boundaries
            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            overflow_left = max(0, -x1) * img_w
            overflow_top = max(0, -y1) * img_h
            overflow_right = max(0, x2 - 1) * img_w
            overflow_bottom = max(0, y2 - 1) * img_h

            max_overflow_px = max(
                overflow_left,
                overflow_top,
                overflow_right,
                overflow_bottom,
            )

            if max_overflow_px > 0:
                outside_boxes.append(
                    {
                        "split": split,
                        "file": label_path.name,
                        "line": line_no,
                        "class": cls,
                        "max_overflow_px": max_overflow_px,
                    }
                )

            class_counts[cls] += 1
            valid_boxes += 1


    print("\n=== YOLO ANNOTATION SUMMARY ===")
    print("Total parsed boxes:", total_boxes)
    print("Structurally valid boxes:", valid_boxes)
    print("Other annotation issues:", len(issues))
    print("Boxes extending outside image:", len(outside_boxes))

    print("\n=== CLASS IDs FOUND ===")
    for cls, count in sorted(class_counts.items()):
        print(f"Class {cls}: {count}")

    if outside_boxes:
        overflow_values = [
            x["max_overflow_px"] for x in outside_boxes
        ]

        print("\n=== OUT-OF-BOUND BOX DETAILS ===")
        print("Maximum overflow (pixels):", max(overflow_values))
        print(
            "Boxes with overflow > 0.1 pixel:",
            sum(x > 0.1 for x in overflow_values)
        )
        print(
            "Boxes with overflow > 1 pixel:",
            sum(x > 1 for x in overflow_values)
        )

        print("\nFirst 10 examples:")
        for item in outside_boxes[:10]:
            print(item)

    if issues:
        print("\n=== FIRST 10 OTHER ISSUES ===")
        for item in issues[:10]:
            print(item)
    return {'issues': issues, 'class_counts': class_counts, 'total_boxes': total_boxes, 'valid_boxes': valid_boxes, 'outside_boxes': outside_boxes}


def describe_boxes(all_pairs):
    """Reconstructed from notebook 01 cell 6; explicit inputs, historical arithmetic."""
    import pandas as pd
    from PIL import Image

    records = []

    print("Starting class and bounding-box analysis...")

    for i, (split, image_path, label_path) in enumerate(all_pairs, start=1):

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        for line in label_path.read_text().splitlines():

            if not line.strip():
                continue

            cls, xc, yc, bw, bh = map(float, line.split())

            records.append({
                "split": split,
                "image": image_path.name,
                "class_id": int(cls),
                "image_width": img_w,
                "image_height": img_h,
                "box_width_px": bw * img_w,
                "box_height_px": bh * img_h,
                "box_area_fraction": bw * bh,
                "box_area_px": (bw * img_w) * (bh * img_h),
            })

        if i % 1000 == 0:
            print(f"Processed {i:,} / {len(all_pairs):,} images")

    print(f"Processed {len(all_pairs):,} / {len(all_pairs):,} images")
    print("Analysis complete.")

    df_boxes = pd.DataFrame(records)

    print("\n=== OVERALL ===")
    print("Total boxes:", len(df_boxes))

    print("\n=== CLASS DISTRIBUTION ===")

    class_summary = (
        df_boxes["class_id"]
        .value_counts()
        .sort_index()
        .rename("count")
        .to_frame()
    )

    class_summary["percentage"] = (
        class_summary["count"] / len(df_boxes) * 100
    )

    print(class_summary.round(2))

    smallest = class_summary["count"].min()
    largest = class_summary["count"].max()

    print("\n=== CLASS IMBALANCE ===")
    print("Smallest class:", smallest)
    print("Largest class:", largest)
    print("Largest / smallest ratio:", round(largest / smallest, 2))

    print("\n=== RELATIVE BOX AREA ===")
    print(
        df_boxes["box_area_fraction"].describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
        )
    )

    small_1 = (df_boxes["box_area_fraction"] < 0.01).sum()
    small_025 = (df_boxes["box_area_fraction"] < 0.0025).sum()

    print("\nBoxes occupying <1% of crop:")
    print(small_1, f"({small_1 / len(df_boxes) * 100:.2f}%)")

    print("\nBoxes occupying <0.25% of crop:")
    print(small_025, f"({small_025 / len(df_boxes) * 100:.2f}%)")

    print("\n=== TRAIN / VALIDATION BOX COUNTS ===")
    print(pd.crosstab(df_boxes["class_id"], df_boxes["split"]))
    return {'df_boxes': df_boxes, 'class_summary': class_summary}


def exact_duplicates(all_images):
    """Reconstructed from notebook 01 cell 7; explicit inputs, historical arithmetic."""
    import hashlib
    from collections import defaultdict
    from PIL import Image

    print("Starting exact pixel-duplicate audit...")

    hash_groups = defaultdict(list)

    for i, (split, path) in enumerate(all_images, start=1):

        with Image.open(path) as img:
            img = img.convert("RGB")

            # Include dimensions so structurally different images
            # cannot share the same pixel-byte interpretation.
            pixel_hash = rgb_pixel_digest(img.width, img.height, img.tobytes())

        hash_groups[pixel_hash].append({
            "split": split,
            "file": path.name,
        })

        if i % 1000 == 0:
            print(f"Processed {i:,} / {len(all_images):,} images")

    print(f"Processed {len(all_images):,} / {len(all_images):,} images")


    # Keep only hashes belonging to more than one image
    duplicate_groups = {
        h: items
        for h, items in hash_groups.items()
        if len(items) > 1
    }


    within_train_groups = []
    within_val_groups = []
    cross_split_groups = []

    for h, items in duplicate_groups.items():

        splits = {item["split"] for item in items}

        if splits == {"train"}:
            within_train_groups.append((h, items))

        elif splits == {"val"}:
            within_val_groups.append((h, items))

        elif "train" in splits and "val" in splits:
            cross_split_groups.append((h, items))


    duplicate_images = sum(
        len(items)
        for items in duplicate_groups.values()
    )


    print("\n=== EXACT PIXEL-DUPLICATE SUMMARY ===")

    print("Total images:", len(all_images))
    print("Unique pixel hashes:", len(hash_groups))
    print("Duplicate hash groups:", len(duplicate_groups))
    print("Images belonging to duplicate groups:", duplicate_images)

    print("\n=== BY LOCATION ===")

    print("Within-train duplicate groups:", len(within_train_groups))
    print("Within-validation duplicate groups:", len(within_val_groups))
    print("Cross-split duplicate groups:", len(cross_split_groups))


    if duplicate_groups:

        print("\n=== EXAMPLE DUPLICATE GROUPS ===")

        for group_no, (h, items) in enumerate(
            list(duplicate_groups.items())[:10],
            start=1
        ):
            print(f"\nGroup {group_no}:")
            for item in items:
                print(
                    f"  {item['split']}: {item['file']}"
                )

    else:
        print("\nNo exact pixel-identical duplicate images detected.")
    return {'hash_groups': hash_groups, 'duplicate_groups': duplicate_groups, 'within_train_groups': within_train_groups, 'within_val_groups': within_val_groups, 'cross_split_groups': cross_split_groups}


def screen_phash(train_images, val_images):
    """Reconstructed from notebook 01 cell 8; explicit inputs, historical arithmetic."""
    import cv2
    import numpy as np
    import pandas as pd

    print("Starting cross-split perceptual-hash screening...")


    # ============================================================
    # 1. 64-bit perceptual hash
    # ============================================================



    # ============================================================
    # 2. Calculate hashes
    # ============================================================

    print("\n[1/3] Hashing training images...")

    train_hashes = []

    for i, path in enumerate(train_images, start=1):
        train_hashes.append(compute_phash(path))

        if i % 1000 == 0:
            print(f"Train: {i:,} / {len(train_images):,}")

    train_hashes = np.stack(train_hashes)


    print("\n[2/3] Hashing validation images...")

    val_hashes = []

    for i, path in enumerate(val_images, start=1):
        val_hashes.append(compute_phash(path))

        if i % 500 == 0:
            print(f"Validation: {i:,} / {len(val_images):,}")

    val_hashes = np.stack(val_hashes)


    print("\nHashing complete.")
    print("Train hash matrix:", train_hashes.shape)
    print("Validation hash matrix:", val_hashes.shape)


    # ============================================================
    # 3. Cross-split Hamming distances
    # ============================================================

    print("\n[3/3] Comparing train vs validation hashes...")

    # Fast lookup table for number of set bits in each byte
    POPCOUNT = np.array(
        [int(i).bit_count() for i in range(256)],
        dtype=np.uint8
    )

    thresholds = [0, 2, 4, 6, 8, 10]

    pair_counts = {t: 0 for t in thresholds}

    nearest_distances = []

    # Keep candidates at <=10 only for later verification
    candidate_records = []

    BATCH_SIZE = 64

    for start in range(0, len(val_images), BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            len(val_images)
        )

        # Shape:
        # batch x train x 8 bytes
        xor = np.bitwise_xor(
            val_hashes[start:end, None, :],
            train_hashes[None, :, :]
        )

        distances = POPCOUNT[xor].sum(axis=2)

        # Nearest training image for each validation image
        nearest_distances.extend(
            distances.min(axis=1).tolist()
        )

        # Count pairs at several thresholds
        for threshold in thresholds:
            pair_counts[threshold] += int(
                (distances <= threshold).sum()
            )

        # Save liberal candidate set for next audit stage
        rows, cols = np.where(distances <= 10)

        for row, col in zip(rows, cols):
            candidate_records.append({
                "train_file": train_images[col].name,
                "val_file": val_images[start + row].name,
                "phash_distance": int(
                    distances[row, col]
                )
            })

        print(
            f"Compared validation images "
            f"{start + 1:,}-{end:,} / {len(val_images):,}"
        )


    # ============================================================
    # Results
    # ============================================================

    nearest_distances = np.array(nearest_distances)

    phash_candidates = pd.DataFrame(candidate_records)

    print("\n=== CROSS-SPLIT pHASH SCREENING ===")

    print(
        "Total train-validation comparisons:",
        len(train_images) * len(val_images)
    )

    print("\nCandidate pair counts:")

    for threshold in thresholds:
        print(
            f"Hamming distance <= {threshold:2d}: "
            f"{pair_counts[threshold]:,}"
        )


    print("\n=== NEAREST TRAIN IMAGE PER VALIDATION IMAGE ===")

    print(
        pd.Series(nearest_distances).describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )


    print("\nValidation images with nearest train pHash distance:")

    for threshold in [2, 4, 6, 8, 10]:
        count = int(
            (nearest_distances <= threshold).sum()
        )

        print(
            f"<= {threshold:2d}: "
            f"{count:,} / {len(val_images):,}"
        )


    print("\nCandidate pairs retained for secondary checks:")
    print(len(phash_candidates))

    if len(phash_candidates) > 0:
        print("\nFirst 20 candidates:")
        print(
            phash_candidates
            .sort_values("phash_distance")
            .head(20)
            .to_string(index=False)
        )

    print("\nPerceptual-hash screening complete.")
    return {'phash_candidates': phash_candidates, 'pair_counts': pair_counts, 'nearest_distances': nearest_distances}


def verify_candidates(train_images, val_images, phash_candidates):
    """Reconstructed from notebook 01 cell 9; explicit inputs, historical arithmetic."""
    import cv2
    import numpy as np
    import pandas as pd

    print("Starting secondary verification of pHash candidates...")

    # ------------------------------------------------------------
    # File lookup
    # ------------------------------------------------------------

    train_lookup = {
        p.name: p
        for p in train_images
    }

    val_lookup = {
        p.name: p
        for p in val_images
    }


    # ------------------------------------------------------------
    # Load image into common comparison representation
    # ------------------------------------------------------------



    # ------------------------------------------------------------
    # Similarity metrics
    # ------------------------------------------------------------



    # ------------------------------------------------------------
    # Cache images so the same files are not repeatedly decoded
    # ------------------------------------------------------------

    train_cache = {}
    val_cache = {}


    def get_train(name):
        if name not in train_cache:
            train_cache[name] = load_for_similarity(
                train_lookup[name]
            )
        return train_cache[name]


    def get_val(name):
        if name not in val_cache:
            val_cache[name] = load_for_similarity(
                val_lookup[name]
            )
        return val_cache[name]


    # ------------------------------------------------------------
    # Evaluate pHash candidates
    # ------------------------------------------------------------

    secondary_records = []

    total = len(phash_candidates)

    for i, row in enumerate(
        phash_candidates.itertuples(index=False),
        start=1
    ):

        train_img = get_train(row.train_file)
        val_img = get_val(row.val_file)

        mae, pearson = calculate_similarity(
            train_img,
            val_img
        )

        secondary_records.append({
            "train_file": row.train_file,
            "val_file": row.val_file,
            "phash_distance": row.phash_distance,
            "mae": mae,
            "pearson": pearson,
        })

        if i % 500 == 0:
            print(
                f"Verified {i:,} / {total:,} candidate pairs"
            )


    print(
        f"Verified {total:,} / {total:,} candidate pairs"
    )

    similarity_df = pd.DataFrame(
        secondary_records
    )


    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print("\n=== SECONDARY SIMILARITY SUMMARY ===")

    print("Candidate pairs:", len(similarity_df))

    print("\nNormalized MAE:")
    print(
        similarity_df["mae"].describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )

    print("\nPearson correlation:")
    print(
        similarity_df["pearson"].describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )


    print("\n=== SIMILARITY COUNTS ===")

    for threshold in [0.10, 0.075, 0.05, 0.03, 0.02, 0.01]:
        count = int(
            (similarity_df["mae"] <= threshold).sum()
        )

        print(
            f"MAE <= {threshold:0.3f}: {count:,}"
        )


    print()

    for threshold in [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]:
        count = int(
            (similarity_df["pearson"] >= threshold).sum()
        )

        print(
            f"Pearson >= {threshold:0.3f}: {count:,}"
        )


    # ------------------------------------------------------------
    # Joint screening — not yet a final duplicate definition
    # ------------------------------------------------------------

    joint_screen = similarity_df[
        (similarity_df["mae"] <= 0.05)
        &
        (similarity_df["pearson"] >= 0.95)
    ].copy()

    joint_screen = joint_screen.sort_values(
        ["mae", "pearson", "phash_distance"],
        ascending=[True, False, True]
    )


    print("\n=== JOINT HIGH-SIMILARITY SCREEN ===")
    print(
        "Pairs with MAE <= 0.05 AND Pearson >= 0.95:",
        len(joint_screen)
    )

    print("\nTop candidate pairs:")

    print(
        similarity_df
        .sort_values(
            ["mae", "pearson", "phash_distance"],
            ascending=[True, False, True]
        )
        .head(30)
        .to_string(index=False)
    )

    print("\nSecondary verification complete.")
    return {'similarity_df': similarity_df, 'joint_screen': joint_screen, 'train_lookup': train_lookup, 'val_lookup': val_lookup}


def shortlist(similarity_df, OUTPUT_DIR):
    """Reconstructed from notebook 01 cell 10; explicit inputs, historical arithmetic."""
    # ============================================================
    # High-confidence near-duplicate shortlist
    # ============================================================

    print("Building tiered near-duplicate shortlist...")


    # Tier A: extremely strong similarity
    tier_a = similarity_df[
        (similarity_df["mae"] <= 0.01)
        & (similarity_df["pearson"] >= 0.99)
    ].copy()


    # Tier B: very strong similarity
    tier_b = similarity_df[
        (similarity_df["mae"] <= 0.02)
        & (similarity_df["pearson"] >= 0.98)
        & (similarity_df["phash_distance"] <= 6)
    ].copy()


    # Tier C: broader high-similarity review pool
    tier_c = similarity_df[
        (similarity_df["mae"] <= 0.03)
        & (similarity_df["pearson"] >= 0.97)
        & (similarity_df["phash_distance"] <= 6)
    ].copy()


    def sort_candidates(df):
        return df.sort_values(
            ["mae", "pearson", "phash_distance"],
            ascending=[True, False, True]
        )


    tier_a = sort_candidates(tier_a)
    tier_b = sort_candidates(tier_b)
    tier_c = sort_candidates(tier_c)


    print("\n=== TIER COUNTS ===")
    print("Tier A:", len(tier_a))
    print("Tier B:", len(tier_b))
    print("Tier C:", len(tier_c))


    print("\n=== TIER A CANDIDATES ===")

    if len(tier_a):
        print(
            tier_a.to_string(index=False)
        )
    else:
        print("None")


    print("\n=== TOP 30 TIER C CANDIDATES ===")

    print(
        tier_c
        .head(30)
        .to_string(index=False)
    )


    # Save shortlist for reproducibility

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tier_c.to_csv(
        OUTPUT_DIR / "cross_split_near_duplicate_candidates.csv",
        index=False
    )

    print(
        "\nSaved:",
        (OUTPUT_DIR / "cross_split_near_duplicate_candidates.csv")

    )
    return {'tier_a': tier_a, 'tier_b': tier_b, 'tier_c': tier_c}


def show_tier_a(tier_a, train_lookup, val_lookup, REVIEW_DIR):
    """Reconstructed from notebook 01 cell 11; explicit inputs, historical arithmetic."""
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    print("Displaying Tier A candidates for manual review...")




    for i, row in enumerate(tier_a.itertuples(index=False), start=1):

        train_path = train_lookup[row.train_file]
        val_path = val_lookup[row.val_file]

        train_img = load_rgb(train_path)
        val_img = load_rgb(val_path)

        # Resize validation image only for visual pixel-difference comparison
        if train_img.shape != val_img.shape:
            val_for_diff = cv2.resize(
                val_img,
                (train_img.shape[1], train_img.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        else:
            val_for_diff = val_img

        diff = cv2.absdiff(
            train_img,
            val_for_diff
        )

        # Amplify small differences for visibility
        diff_amplified = np.clip(
            diff.astype(np.float32) * 5,
            0,
            255
        ).astype(np.uint8)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4)
        )

        axes[0].imshow(train_img)
        axes[0].set_title(
            f"TRAIN\n{row.train_file}"
        )

        axes[1].imshow(val_img)
        axes[1].set_title(
            f"VALIDATION\n{row.val_file}"
        )

        axes[2].imshow(diff_amplified)
        axes[2].set_title(
            "Absolute difference ×5"
        )

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"Tier A Pair {i}\n"
            f"pHash={row.phash_distance} | "
            f"MAE={row.mae:.6f} | "
            f"Pearson={row.pearson:.6f}"
        )

        plt.tight_layout()
        fig.savefig(REVIEW_DIR / f"tier_a_{i:03d}.png")
        plt.close(fig)
    return {}


def align_candidates(tier_c, train_lookup, val_lookup, OUTPUT_DIR):
    """Reconstructed from notebook 01 cell 12; explicit inputs, historical arithmetic."""
    import cv2
    import numpy as np
    import pandas as pd

    print("Starting translation-aware similarity analysis...")








    alignment_records = []

    total = len(tier_c)

    for i, row in enumerate(tier_c.itertuples(index=False), start=1):

        train_path = train_lookup[row.train_file]
        val_path = val_lookup[row.val_file]

        train_img = load_gray_original(train_path)
        val_img = load_gray_original(val_path)

        best = best_translation_match(
            train_img,
            val_img,
            max_shift=10
        )

        alignment_records.append({
            "train_file": row.train_file,
            "val_file": row.val_file,
            "phash_distance": row.phash_distance,
            "original_mae": row.mae,
            "original_pearson": row.pearson,
            "best_dx": best["dx"],
            "best_dy": best["dy"],
            "aligned_mae": best["aligned_mae"],
            "aligned_pearson": best["aligned_pearson"],
            "overlap_fraction": best["overlap_fraction"],
        })

        if i % 20 == 0:
            print(f"Checked {i:,} / {total:,} pairs")


    print(f"Checked {total:,} / {total:,} pairs")

    alignment_df = pd.DataFrame(alignment_records)

    alignment_df["shift_magnitude"] = np.sqrt(
        alignment_df["best_dx"] ** 2
        + alignment_df["best_dy"] ** 2
    )

    alignment_df["mae_improvement"] = (
        alignment_df["original_mae"]
        - alignment_df["aligned_mae"]
    )


    print("\n=== ALIGNMENT-AWARE SUMMARY ===")

    print("Tier-C pairs checked:", len(alignment_df))

    print("\nAligned MAE:")
    print(
        alignment_df["aligned_mae"].describe(
            percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
    )

    print("\nAligned Pearson:")
    print(
        alignment_df["aligned_pearson"].describe(
            percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
    )


    print("\n=== HIGH-SIMILARITY AFTER ALIGNMENT ===")

    for mae_t, corr_t in [
        (0.010, 0.990),
        (0.015, 0.985),
        (0.020, 0.980),
    ]:
        count = (
            (alignment_df["aligned_mae"] <= mae_t)
            & (alignment_df["aligned_pearson"] >= corr_t)
        ).sum()

        print(
            f"MAE <= {mae_t:.3f} AND Pearson >= {corr_t:.3f}: "
            f"{count}"
        )


    final_shortlist = alignment_df[
        (alignment_df["aligned_mae"] <= 0.015)
        & (alignment_df["aligned_pearson"] >= 0.985)
    ].copy()

    final_shortlist = final_shortlist.sort_values(
        ["aligned_mae", "aligned_pearson"],
        ascending=[True, False]
    )


    print("\n=== FINAL MANUAL-REVIEW SHORTLIST ===")
    print("Pairs:", len(final_shortlist))

    print(
        final_shortlist[
            [
                "train_file",
                "val_file",
                "best_dx",
                "best_dy",
                "aligned_mae",
                "aligned_pearson",
                "overlap_fraction",
            ]
        ].to_string(index=False)
    )




    alignment_df.to_csv(
        OUTPUT_DIR / "cross_split_alignment_analysis.csv",
        index=False
    )

    final_shortlist.to_csv(
        OUTPUT_DIR / "cross_split_final_manual_review.csv",
        index=False
    )

    print("\nSaved alignment results.")
    return {'alignment_df': alignment_df, 'final_shortlist': final_shortlist}


def show_final_review(final_shortlist, train_lookup, val_lookup, REVIEW_DIR):
    """Reconstructed from notebook 01 cell 13; explicit inputs, historical arithmetic."""
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    import math

    print("Creating final manual-review pages...")






    review_df = final_shortlist.reset_index(drop=True)

    PAIRS_PER_PAGE = 6
    pages = math.ceil(len(review_df) / PAIRS_PER_PAGE)

    for page in range(pages):

        start = page * PAIRS_PER_PAGE
        end = min(start + PAIRS_PER_PAGE, len(review_df))

        subset = review_df.iloc[start:end]

        fig, axes = plt.subplots(
            len(subset),
            3,
            figsize=(12, 3.2 * len(subset))
        )

        if len(subset) == 1:
            axes = np.array([axes])

        for row_idx, (_, row) in enumerate(subset.iterrows()):

            train_img = load_rgb(train_lookup[row["train_file"]])
            val_img = load_rgb(val_lookup[row["val_file"]])

            aligned_train, aligned_val = align_for_display(
                train_img,
                val_img,
                int(row["best_dx"]),
                int(row["best_dy"])
            )

            diff = cv2.absdiff(
                aligned_train,
                aligned_val
            )

            diff = np.clip(
                diff.astype(np.float32) * 5,
                0,
                255
            ).astype(np.uint8)

            pair_number = start + row_idx + 1

            axes[row_idx, 0].imshow(train_img)
            axes[row_idx, 0].set_title(
                f"Pair {pair_number} TRAIN\n{row['train_file']}"
            )

            axes[row_idx, 1].imshow(val_img)
            axes[row_idx, 1].set_title(
                f"VALIDATION\n{row['val_file']}"
            )

            axes[row_idx, 2].imshow(diff)
            axes[row_idx, 2].set_title(
                f"Aligned diff ×5\n"
                f"shift=({int(row['best_dx'])},{int(row['best_dy'])}) | "
                f"MAE={row['aligned_mae']:.4f} | "
                f"r={row['aligned_pearson']:.4f}"
            )

            for ax in axes[row_idx]:
                ax.axis("off")

        plt.tight_layout()
        fig.savefig(REVIEW_DIR / f"final_review_{page + 1:03d}.png")
        plt.close(fig)

        print(
            f"Displayed review page {page + 1}/{pages}: "
            f"pairs {start + 1}-{end}"
        )
    return {}


if __name__ == "__main__":
    main()
