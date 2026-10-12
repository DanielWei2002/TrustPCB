"""Research helpers extracted/reconstructed from historical notebook 01.

Source: notebooks/01_dataset_structure_audit.ipynb at 1937995fb78d979a66e237b7ab7facbe20ca0513.
Cell indexes below are zero-based. This module did NOT produce the original evidence.
No dataset traversal, CLI, or writes occur here. Real-data use belongs on DICC.
"""

import hashlib
import numpy as np


def rgb_pixel_digest(width, height, rgb_bytes):
    """Cell 7 payload/hash, extracted with decoded RGB bytes as explicit input."""
    payload = f"{width}x{height}|RGB|".encode() + rgb_bytes
    return hashlib.sha256(payload).hexdigest()


# Notebook 01, cell 3; function body retained except lazy OpenCV import.
def check_pairs(images, labels, split_name):
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    print(f"=== {split_name} ===")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Unique image names:", len(image_stems))
    print("Unique label names:", len(label_stems))
    print("Images without labels:", len(missing_labels))
    print("Labels without images:", len(missing_images))

    if missing_labels:
        print("\nExample images without labels:")
        for x in missing_labels[:10]:
            print(" ", x)

    if missing_images:
        print("\nExample labels without images:")
        for x in missing_images[:10]:
            print(" ", x)

    print()


# Notebook 01, cell 9; function body retained except lazy OpenCV import.
def calculate_similarity(img1, img2):

    # Mean absolute pixel difference
    mae = float(
        np.mean(np.abs(img1 - img2))
    )

    # Pearson correlation
    a = img1.ravel()
    b = img2.ravel()

    if np.std(a) == 0 or np.std(b) == 0:
        pearson = np.nan
    else:
        pearson = float(
            np.corrcoef(a, b)[0, 1]
        )

    return mae, pearson


# Notebook 01, cell 12; function body retained except lazy OpenCV import.
def compare_at_shift(a, b, dx, dy):
    """
    Compare overlapping pixels after shifting b relative to a.
    dx > 0 means b is shifted right.
    dy > 0 means b is shifted down.
    """

    h, w = a.shape

    # Horizontal overlap
    if dx >= 0:
        ax1, ax2 = dx, w
        bx1, bx2 = 0, w - dx
    else:
        ax1, ax2 = 0, w + dx
        bx1, bx2 = -dx, w

    # Vertical overlap
    if dy >= 0:
        ay1, ay2 = dy, h
        by1, by2 = 0, h - dy
    else:
        ay1, ay2 = 0, h + dy
        by1, by2 = -dy, h

    a_crop = a[ay1:ay2, ax1:ax2]
    b_crop = b[by1:by2, bx1:bx2]

    if a_crop.size == 0 or b_crop.size == 0:
        return None

    mae = float(np.mean(np.abs(a_crop - b_crop)))

    aa = a_crop.ravel()
    bb = b_crop.ravel()

    if np.std(aa) == 0 or np.std(bb) == 0:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(aa, bb)[0, 1])

    overlap_fraction = a_crop.size / a.size

    return mae, pearson, overlap_fraction


# Notebook 01, cell 12; function body retained except lazy OpenCV import.
def best_translation_match(a, b, max_shift=10):

    # Only compare equal-sized arrays directly.
    # If dimensions differ, resize b only for this diagnostic test.
    if a.shape != b.shape:
        import cv2  # Optional; only the historical resize branch needs OpenCV.
        b = cv2.resize(
            b,
            (a.shape[1], a.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    best = None

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):

            result = compare_at_shift(a, b, dx, dy)

            if result is None:
                continue

            mae, pearson, overlap = result

            # Require substantial overlap
            if overlap < 0.85:
                continue

            if best is None or mae < best["aligned_mae"]:
                best = {
                    "dx": dx,
                    "dy": dy,
                    "aligned_mae": mae,
                    "aligned_pearson": pearson,
                    "overlap_fraction": overlap,
                }

    return best
