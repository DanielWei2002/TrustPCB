"""DICC-only development OOF calibrator comparison; never fits a final calibrator."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import random

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from trustpcb import rq1, rq2_calibration_analysis as analysis
from trustpcb import rq2_confidence_distribution as raw, rq2_data_partition as partition
from trustpcb.dataset_config import find_project_root

SEED = 24209199
FOLDS = 5
RESAMPLES = 10000
EXPECTED_PREDICTIONS = 7131
EPSILON = 1e-15
MIN_SLOPE = 1e-8
METHODS = ("Temperature", "Platt", "Beta", "Isotonic")
ALL_METHODS = ("Raw", *METHODS)
OUTPUT = "runs/rq2/calibrator_comparison"


def assign_folds(images, edges, seed=SEED, n_folds=FOLDS):
    """Largest connected components first; only image counts and seeded tie ranks."""
    if len(set(images)) != len(images) or len(images) < n_folds:
        raise ValueError("Expected unique images and enough images for folds")
    groups = partition.components(images, edges)
    if len(groups) < n_folds:
        raise ValueError("Too few independent similarity components for five folds")
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda g: -len(g))  # stable seeded order for equal sizes
    priority = list(range(n_folds))
    rng.shuffle(priority)
    sizes = [0] * n_folds
    membership = {}
    for group in groups:
        fold = min(priority, key=lambda f: sizes[f])
        component = hashlib.sha256("\n".join(group).encode()).hexdigest()
        for image in group:
            membership[image] = {"image": image, "fold_id": fold, "component_id": component}
        sizes[fold] += len(group)
    return [membership[image] for image in images]


def design(method, confidence):
    p = np.clip(np.asarray(confidence, dtype=float), EPSILON, 1 - EPSILON)
    logit = np.log(p) - np.log1p(-p)
    if method == "Temperature":
        return logit[:, None]
    if method == "Platt":
        return np.column_stack((logit, np.ones(len(p))))
    if method == "Beta":
        return np.column_stack((np.log(p), -np.log1p(-p), np.ones(len(p))))
    raise ValueError("Unknown parametric calibrator")


def fit_calibrator(method, confidence, labels):
    """Fold-training data only. No regularisation, method search or final refit."""
    p, y = np.asarray(confidence, dtype=float), np.asarray(labels, dtype=float)
    if (p.ndim != 1 or p.shape != y.shape or not len(p) or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all()):
        raise ValueError("Invalid fold training data")
    if len(np.unique(y)) != 2:
        raise ValueError("Single-class training fold requires scientific review; no fallback fit")
    if method == "Isotonic":
        # Weighted PAVA after aggregating duplicate raw confidences.
        x, inverse, weights = np.unique(p, return_inverse=True, return_counts=True)
        positives = np.bincount(inverse, weights=y, minlength=len(x))
        blocks = []
        for i in range(len(x)):
            blocks.append([i, i, float(weights[i]), float(positives[i])])
            while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
                right = blocks.pop()
                left = blocks.pop()
                blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
        fitted = np.empty(len(x))
        for lo, hi, weight, positive in blocks:
            fitted[lo:hi + 1] = positive / weight
        return {"method": method, "x": x.tolist(), "y": fitted.tolist()}
    x = design(method, p)
    initial, bounds = {
        "Temperature": ([1.], [(MIN_SLOPE, None)]),
        "Platt": ([1., 0.], [(MIN_SLOPE, None), (None, None)]),
        "Beta": ([1., 1., 0.], [(0., None), (0., None), (None, None)]),
    }[method]
    def objective(theta):
        z = x @ theta
        loss = np.mean(np.logaddexp(0., z) - y * z)
        gradient = x.T @ (expit(z) - y) / len(y)
        return float(loss), gradient
    result = minimize(objective, initial, jac=True, bounds=bounds, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50})
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError(f"{method} fold fit failed; scientific review required: {result.message}")
    return {"method": method, "parameters": result.x.tolist(), "iterations": int(result.nit),
            "training_nll": float(result.fun), "optimizer_message": str(result.message)}


def apply_calibrator(model, confidence):
    p = np.asarray(confidence, dtype=float)
    if model["method"] == "Isotonic":
        # Linear interpolation between fitted unique scores, constant extension.
        return np.interp(p, model["x"], model["y"])
    return expit(design(model["method"], p) @ np.asarray(model["parameters"]))


def cross_fit(images, predictions, folds, fitter=fit_calibrator):
    if [r["image"] for r in folds] != images or len(set(images)) != len(images):
        raise ValueError("Fold inventory must cover every development image once in manifest order")
    lookup = {r["image"]: r["fold_id"] for r in folds}
    if set(lookup.values()) != set(range(FOLDS)):
        raise ValueError("Expected all five nonempty image folds")
    if any(r["image"] not in lookup for r in predictions):
        raise ValueError("Prediction outside development set")
    p = np.array([r["raw_confidence"] for r in predictions])
    y = np.array([r["correct_iou50"] for r in predictions])
    prediction_folds = np.array([lookup[r["image"]] for r in predictions])
    probabilities = {"Raw": p.copy(), **{m: np.full(len(p), np.nan) for m in METHODS}}
    fits = []
    for fold in range(FOLDS):
        train, held = prediction_folds != fold, prediction_folds == fold
        if not train.any():
            raise ValueError("Empty prediction training fold")
        for method in METHODS:
            model = fitter(method, p[train], y[train])
            probabilities[method][held] = apply_calibrator(model, p[held])
            fits.append({"fold_id": fold, "training_predictions": int(train.sum()),
                         "held_out_predictions": int(held.sum()), "model": model})
    if any(not np.isfinite(v).all() or np.any((v < 0) | (v > 1)) for v in probabilities.values()):
        raise RuntimeError("Missing/invalid OOF probability")
    return probabilities, prediction_folds, fits


def losses(probabilities, labels):
    p, y = np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid probabilities or primary labels")
    safe = np.clip(p, EPSILON, 1 - EPSILON)
    return {"nll": -(y * np.log(safe) + (1 - y) * np.log1p(-safe)), "brier": (p - y) ** 2}


def metric_summary(probabilities, labels):
    values = losses(probabilities, labels)
    if not len(probabilities):
        raise ValueError("Empty scientific prediction population")
    p, y = np.asarray(probabilities), np.asarray(labels)
    indices = np.minimum(9, np.searchsorted(analysis.BIN_EDGES, p, side="right") - 1)
    bins = []
    for index in range(10):
        mask = indices == index
        bins.append({"lower_inclusive": analysis.BIN_EDGES[index], "upper": analysis.BIN_EDGES[index + 1],
                     "upper_inclusive": index == 9, "count": int(mask.sum()),
                     "mean_confidence": float(p[mask].mean()) if mask.any() else None,
                     "correctness_rate": float(y[mask].mean()) if mask.any() else None})
    ece = sum(b["count"] * abs(b["mean_confidence"] - b["correctness_rate"]) for b in bins if b["count"]) / len(p)
    return {"nll": float(values["nll"].mean()), "brier": float(values["brier"].mean()),
            "ece": float(ece), "count": len(p), "reliability_bins": bins}


def cluster_bootstrap(images, predictions, probabilities, resamples=RESAMPLES, seed=SEED):
    """Shared image multiplicities applied to per-image sums (exact cluster resampling)."""
    lookup = {image: i for i, image in enumerate(images)}
    ids = np.array([lookup[r["image"]] for r in predictions])
    counts = np.bincount(ids, minlength=len(images))
    y = [r["correct_iou50"] for r in predictions]
    totals = np.empty((len(images), len(ALL_METHODS), 2))
    for j, method in enumerate(ALL_METHODS):
        values = losses(probabilities[method], y)
        for k, metric in enumerate(("nll", "brier")):
            totals[:, j, k] = np.bincount(ids, weights=values[metric], minlength=len(images))
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap = np.empty((resamples, len(ALL_METHODS), 2))
    draw_digest = hashlib.sha256()
    for start in range(0, resamples, 128):
        size = min(128, resamples - start)
        draws = rng.integers(0, len(images), size=(size, len(images)), dtype=np.int64)
        draw_digest.update(draws.astype("<i8", copy=False).tobytes())
        weights = np.zeros((size, len(images)), dtype=np.int64)
        np.add.at(weights, (np.arange(size)[:, None], draws), 1)
        denominators = weights @ counts
        if np.any(denominators == 0):
            raise ValueError("Bootstrap sample contains no predictions; scientific review required")
        bootstrap[start:start + size] = np.einsum("bi,imk->bmk", weights, totals) / denominators[:, None, None]
    comparisons = []
    for i, competitor in enumerate(ALL_METHODS):
        for j, reference in enumerate(ALL_METHODS):
            if i == j:
                continue
            for k, metric in enumerate(("nll", "brier")):
                delta = bootstrap[:, i, k] - bootstrap[:, j, k]
                low, high = np.percentile(delta, [2.5, 97.5], method="linear")
                point = (totals[:, i, k].sum() - totals[:, j, k].sum()) / counts.sum()
                comparisons.append({"competitor": competitor, "reference": reference, "metric": metric,
                                    "difference": float(point), "ci_low": float(low), "ci_high": float(high)})
    return comparisons, {"resamples": resamples, "seed": seed, "rng": "PCG64",
                         "draw_indices_sha256": draw_digest.hexdigest(), "cluster_count": len(images),
                         "percentiles": [2.5, 97.5], "percentile_method": "linear"}


def select_method(metrics, comparisons):
    """Frozen hierarchy, no ECE or sensitivity labels; never performs final fitting."""
    leader = min(METHODS, key=lambda m: (metrics[m]["nll"], METHODS.index(m)))
    result = {"numerical_nll_leader": leader, "selected_method": None, "final_fit_performed": False}
    if not any(metrics[m]["nll"] < metrics["Raw"]["nll"] for m in METHODS):
        return {**result, "status": "scientific_review_required", "reason": "No calibrator improves raw point-estimate OOF NLL"}
    def comparison(method, reference, metric):
        rows = [r for r in comparisons if (r["competitor"], r["reference"], r["metric"]) == (method, reference, metric)]
        if len(rows) != 1:
            raise ValueError("Missing/ambiguous paired comparison")
        return rows[0]
    def indistinguishable(method, reference, metric):
        if method == reference:
            return True
        row = comparison(method, reference, metric)
        return row["ci_low"] <= 0 <= row["ci_high"]
    nll_set = [m for m in METHODS if indistinguishable(m, leader, "nll")]
    brier_leader = min(nll_set, key=lambda m: (metrics[m]["brier"], METHODS.index(m)))
    brier_set = [m for m in nll_set if indistinguishable(m, brier_leader, "brier")]
    selected = next(m for m in METHODS if m in brier_set)
    return {**result, "status": "comparison_complete_pending_review", "selected_method": selected,
            "nll_indistinguishable": nll_set, "numerical_brier_leader": brier_leader,
            "brier_indistinguishable": brier_set,
            "selected_minus_raw": {metric: comparison(selected, "Raw", metric) for metric in ("nll", "brier")},
            "reason": "NLL CI, then Brier CI within NLL set, then pre-fixed simplicity order"}


def load_inputs(root):
    """Read only development manifest, labelled predictions/provenance and confirmed-link metadata."""
    root = Path(root)
    images = raw.development_images(root)
    source = root / analysis.OUTPUT
    provenance_path = source / "provenance.json"
    provenance = rq1._read_json(provenance_path)
    expected = {"status": "complete", "inclusion_threshold": 0.01, "primary_iou": 0.50,
                "checkpoint_epoch": raw.EPOCH, "checkpoint_sha256": raw.CHECKPOINT_SHA,
                "manifest_sha256_lf": raw.MANIFEST_SHA}
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError("Input is not the frozen complete development correctness analysis")
    table = source / "labelled_predictions.csv"
    if rq1._sha(table) != provenance.get("output_sha256", {}).get(table.name):
        raise ValueError("Labelled prediction hash mismatch")
    allowed, seen, predictions = set(images), set(), []
    with table.open(encoding="utf-8", newline="") as file:
        for record in csv.DictReader(file):
            # Intentionally never parse or consult the sensitivity correctness field.
            image = record["image"]
            score = float(record["confidence"])
            label = int(record["correct_at_iou50"])
            number = int(record["source_row"])
            identifier = f"{image}#{number}"
            if (image not in allowed or number <= 0 or identifier in seen or not np.isfinite(score)
                    or not 0.01 <= score <= 1 or label not in (0, 1)):
                raise ValueError("Invalid development population; no dropping/refiltering is allowed")
            seen.add(identifier)
            predictions.append({"image": image, "prediction_id": identifier, "raw_confidence": score, "correct_iou50": label})
    if len(images) != 821 or len(predictions) != EXPECTED_PREDICTIONS:
        raise ValueError("Expected exactly 821 development images and 7131 retained predictions")
    pair_path = root / partition.PAIRS
    with pair_path.open(encoding="utf-8-sig", newline="") as file:
        edges = [(r["train_file"], r["val_file"]) for r in csv.DictReader(file)]
    names = {PurePosixPath(image).name: image for image in images}
    if len(names) != len(images):
        raise ValueError("Ambiguous development image basenames")
    # Entire link graph is needed for transitive components through intermediate nodes.
    # This is confirmed-link metadata only; no test manifest, labels or images are opened.
    nodes = set(names) | {name for edge in edges for name in edge}
    development_edges = []
    for group in partition.components(nodes, edges):
        inside = [n for n in group if n in names]
        if inside and len(inside) != len(group):
            raise ValueError("Confirmed similarity component crosses frozen development boundary")
        development_edges.extend((names[inside[0]], names[n]) for n in inside[1:])
    hashes = {raw.DEVELOPMENT: rq1._sha(root / raw.DEVELOPMENT),
              partition.PAIRS: rq1._sha(pair_path),
              f"{analysis.OUTPUT}/provenance.json": rq1._sha(provenance_path),
              f"{analysis.OUTPUT}/labelled_predictions.csv": rq1._sha(table)}
    return images, predictions, development_edges, hashes


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_reliability(metrics, destination):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    for method in ALL_METHODS:
        bins = [b for b in metrics[method]["reliability_bins"] if b["count"]]
        ax.plot([b["mean_confidence"] for b in bins], [b["correctness_rate"] for b in bins], "o-", label=method)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean probability", ylabel="Primary correctness rate",
           title="Development out-of-fold reliability; 10 equal-width bins")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run(root):
    root = Path(root).resolve()
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution inputs before DICC comparison")
    images, predictions, edges, hashes = load_inputs(root)
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "matplotlib")}
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"experiment": "rq2_calibrator_comparison", "status": "running", **git,
                  "input_sha256": hashes, "versions": versions, "seed": SEED, "folds": FOLDS,
                  "resamples": RESAMPLES, "inclusion": 0.01, "primary_iou": 0.50,
                  "log_epsilon": EPSILON, "minimum_positive_slope": MIN_SLOPE,
                  "complexity_order": METHODS, "started_utc": datetime.now(timezone.utc).isoformat()}
    target = output / "provenance.json"
    rq1._write_json(target, provenance, exclusive=True)
    try:
        folds = assign_folds(images, edges)
        write_csv(output / "fold_manifest.csv", folds)  # persisted before any fitting
        probabilities, row_folds, fits = cross_fit(images, predictions, folds)
        oof = [{**row, "fold_id": int(row_folds[i]), "method": method,
                "calibrated_probability": float(probabilities[method][i])}
               for i, row in enumerate(predictions) for method in METHODS]
        write_csv(output / "oof_calibrated_predictions.csv", oof)
        labels = [r["correct_iou50"] for r in predictions]
        metrics = {method: metric_summary(probabilities[method], labels) for method in ALL_METHODS}
        write_csv(output / "method_metrics.csv", [{"method": m, **{k: v for k, v in metrics[m].items() if k != "reliability_bins"}} for m in ALL_METHODS])
        comparisons, bootstrap = cluster_bootstrap(images, predictions, probabilities)
        write_csv(output / "paired_comparisons.csv", comparisons)
        rq1._write_json(output / "selection_summary.json", select_method(metrics, comparisons), exclusive=True)
        rq1._write_json(output / "reliability_data.json", metrics, exclusive=True)
        rq1._write_json(output / "fold_fit_parameters.json", {"fold_fits_only": fits, "final_fit": None}, exclusive=True)
        plot_reliability(metrics, output / "reliability_comparison.png")
        if load_inputs(root)[3] != hashes or rq1.git_provenance(root) != git:
            raise RuntimeError("Inputs/source changed during comparison")
        provenance.update(status="complete", bootstrap=bootstrap,
                          fold_image_counts={str(f): sum(r["fold_id"] == f for r in folds) for f in range(FOLDS)},
                          output_sha256={p.name: rq1._sha(p) for p in output.iterdir() if p != target},
                          completed_utc=datetime.now(timezone.utc).isoformat(), final_fit_performed=False)
        rq1._write_json(target, provenance)
        return output
    except BaseException:
        provenance["status"] = "incomplete"
        rq1._write_json(target, provenance)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicc", action="store_true")
    args = parser.parse_args(argv)
    if os.name == "nt" or not args.dicc:
        raise RuntimeError("Real comparison is DICC-only; --dicc required")
    print(run(find_project_root()))


if __name__ == "__main__":
    main()
