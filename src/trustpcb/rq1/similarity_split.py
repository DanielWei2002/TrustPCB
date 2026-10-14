"""Pure kernels reconstructed from historical notebook 02, not its executed program.

Source: notebooks/02_similarity_aware_split.ipynb at 1937995fb78d979a66e237b7ab7facbe20ca0513.
Explicit arguments replace notebook globals. No manifests are read or written.
The allocation search is combinatorial: real-data execution belongs on DICC.
"""

from collections import defaultdict
from itertools import combinations
import numpy as np


def similarity_groups(pair_edges):
    """Cell 8 connected components; only endpoints in confirmed pairs participate."""
    graph = defaultdict(set)

    for image_a, image_b in pair_edges:
        graph[image_a].add(image_b)
        graph[image_b].add(image_a)

    # Find connected components

    visited = set()
    similarity_groups = []

    for image_name in sorted(graph):

        if image_name in visited:
            continue

        stack = [image_name]
        group = []

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            group.append(current)

            for neighbour in graph[current]:
                if neighbour not in visited:
                    stack.append(neighbour)

        similarity_groups.append(sorted(group))

    return similarity_groups


def select_group_allocation(group_annotation_counts, base_train_counts, base_val_counts,
                            target_train_pct, target_val_pct):
    """Cell 16 search, with precomputed arrays supplied explicitly.

    Preserves [9, 10] group counts, percentage arithmetic, sorted group order,
    mean/max ranking and stable first-candidate tie handling. This does not
    generalize the notebook's pair-based construction of group annotation counts.
    """
    group_ids = sorted(group_annotation_counts.keys())

    results = []

    # Check both near-80/20 group allocations

    for n_train_groups in [9, 10]:

        for train_group_tuple in combinations(
            group_ids,
            n_train_groups
        ):

            train_group_set = set(train_group_tuple)

            train_counts_candidate = base_train_counts.copy()
            val_counts_candidate = base_val_counts.copy()

            for group_id in group_ids:

                if group_id in train_group_set:
                    train_counts_candidate += group_annotation_counts[group_id]
                else:
                    val_counts_candidate += group_annotation_counts[group_id]

            train_pct_candidate = (
                train_counts_candidate
                / train_counts_candidate.sum()
                * 100
            )

            val_pct_candidate = (
                val_counts_candidate
                / val_counts_candidate.sum()
                * 100
            )

            train_diff = np.abs(
                train_pct_candidate - target_train_pct
            )

            val_diff = np.abs(
                val_pct_candidate - target_val_pct
            )

            mean_difference = np.mean(
                np.concatenate([train_diff, val_diff])
            )

            max_difference = np.max(
                np.concatenate([train_diff, val_diff])
            )

            results.append({
                "train_groups": tuple(sorted(train_group_set)),
                "val_groups": tuple(
                    sorted(set(group_ids) - train_group_set)
                ),
                "n_train_groups": n_train_groups,
                "mean_pct_difference": mean_difference,
                "max_pct_difference": max_difference,
                "train_counts": train_counts_candidate,
                "val_counts": val_counts_candidate
            })

    allocation_results = sorted(
        results,
        key=lambda x: (
            x["mean_pct_difference"],
            x["max_pct_difference"]
        )
    )

    best_allocation = allocation_results[0]

    return best_allocation
