"""DICC-only development ranking evaluation; no fitting or threshold selection."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform

import numpy as np
from scipy.stats import rankdata

from trustpcb import rq1, rq2_final_calibrator as final
from trustpcb.rq2_artifact_paths import resolve_input
from trustpcb.dataset_config import find_project_root

OUTPUT = "runs/rq2/risk_evaluation"
STABILITY = "runs/rq2/transformation_stability"
INPUT = f"{STABILITY}/prediction_stability_scores.csv"
SEED, REPLICATES = 24209199, 10000
IMAGE_COUNT, PREDICTION_COUNT = 821, 7131
SIGNALS = ("calibrated_confidence", "class_consistency", "localisation_stability")
FORMULAS = {
    "raw_confidence_risk": "1 - raw_confidence",
    "calibrated_confidence_risk": "1 - calibrated_confidence",
    "class_consistency_risk": "1 - class_consistency",
    "localisation_risk": "1 - localisation_stability",
    "calibrated_plus_class_risk": "(calibrated_confidence_risk + class_consistency_risk) / 2",
    "calibrated_plus_localisation_risk": "(calibrated_confidence_risk + localisation_risk) / 2",
    "class_plus_localisation_risk": "(class_consistency_risk + localisation_risk) / 2",
    "trustpcb_risk": "(calibrated_confidence_risk + class_consistency_risk + localisation_risk) / 3",
}
METHODS = tuple(FORMULAS)
PAIRS = ((METHODS[7], METHODS[0]), (METHODS[7], METHODS[1]),
         (METHODS[4], METHODS[1]), (METHODS[5], METHODS[1]), (METHODS[6], METHODS[1]))
DEFINITIONS = {
    "target": "incorrect_prediction = 1 - correct_iou50",
    "auroc": "P(risk_incorrect > risk_correct) + 0.5 * P(tied risks); prediction-weighted",
    "auprc": "non-interpolated average precision: sum(delta_recall * precision) at distinct descending score groups",
    "undefined": "AUROC undefined without both target classes; AUPRC undefined without positives; empty population undefined for both",
    "interval": "paired image-cluster bootstrap; percentile 2.5/97.5 with linear interpolation; omit undefined replicates without redrawing",
    "descriptives": "population standard deviation ddof=0; median/Q1/Q3 use linear quantiles",
    "spearman": "Pearson correlation of average tied ranks; undefined for a constant signal or fewer than two predictions",
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def validate_rows(images, rows, expected_count=PREDICTION_COUNT):
    if len(images) != IMAGE_COUNT or len(set(images)) != IMAGE_COUNT or len(rows) != expected_count:
        raise ValueError("Expected frozen 821-image / 7131-prediction population")
    allowed, seen, parsed = set(images), set(), []
    for row in rows:
        identity = row.get("prediction_id")
        if not identity or identity in seen or row.get("image") not in allowed:
            raise ValueError("Duplicate/missing prediction identity or foreign development image")
        seen.add(identity)
        try:
            values = {key: float(row[key]) for key in ("raw_confidence", *SIGNALS)}
            label = float(row["correct_iou50"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("Missing/invalid required signal or primary correctness") from exc
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
            raise ValueError("Required signals must be finite and within [0,1]")
        if values["raw_confidence"] < .01 or label not in (0, 1):
            raise ValueError("Frozen inclusion/primary correctness differs")
        parsed.append({**row, **values, "correct_iou50": int(label), "incorrect_prediction": 1 - int(label)})
    return parsed


def load_inputs(root):
    """Text artifacts only: no checkpoint/image/annotation read or model import."""
    root = Path(root)
    images, source, originals, hashes, _, _ = final.load_inputs(root, approve_beta_selection=True)
    folder = resolve_input(root, final.OUTPUT)
    provenance = rq1._read_json(folder / "provenance.json")
    if (provenance.get("status") != "complete" or provenance.get("final_fit_performed") is not True
            or provenance.get("input_sha256") != hashes):
        raise ValueError("Final calibrator provenance does not bind the frozen population")
    for name in ("final_calibrator.json", "calibrated_development_predictions.csv", "fit_summary.json"):
        digest = rq1._sha(folder / name)
        if provenance.get("output_sha256", {}).get(name) != digest:
            raise ValueError("Final calibrator artifact hash mismatch")
        hashes[f"{final.OUTPUT}/{name}"] = digest
    if rq1._read_json(folder / "final_calibrator.json").get("method") != "Beta":
        raise ValueError("Expected frozen Beta calibrator")
    hashes[f"{final.OUTPUT}/provenance.json"] = rq1._sha(folder / "provenance.json")
    frozen = read_csv(folder / "calibrated_development_predictions.csv")
    stability = resolve_input(root, STABILITY)
    metadata = rq1._read_json(stability / "provenance.json")
    if metadata.get("status") != "complete" or metadata.get("input_sha256") != hashes:
        raise ValueError("Incomplete/unbound transformation stability provenance")
    table = resolve_input(root, INPUT)
    digest = rq1._sha(table)
    if metadata.get("output_sha256", {}).get(table.name) != digest:
        raise ValueError("Stability score table hash mismatch")
    rows = read_csv(table)
    if not len(rows) == len(frozen) == len(source) == len(originals):
        raise ValueError("Reference population count differs")
    for row, reference, original, source_row in zip(rows, frozen, originals, source):
        if (any(reference.get(k) != v for k, v in original.items())
                or any(row.get(k) != v for k, v in reference.items())
                or row.get("prediction_id") != source_row["prediction_id"]
                or float(row["raw_confidence"]) != source_row["raw_confidence"]
                or float(row["correct_iou50"]) != source_row["correct_iou50"]):
            raise ValueError("Reference identity/order/values differ from frozen population")
    hashes[INPUT] = digest
    hashes[f"{STABILITY}/provenance.json"] = rq1._sha(stability / "provenance.json")
    return images, validate_rows(images, rows, expected_count=PREDICTION_COUNT), hashes


def risk_scores(rows):
    raw, calibrated, cls, loc = (1 - np.array([r[key] for r in rows], dtype=float)
                                for key in ("raw_confidence", *SIGNALS))
    return dict(zip(METHODS, (raw, calibrated, cls, loc, (calibrated + cls) / 2,
                             (calibrated + loc) / 2, (cls + loc) / 2, (calibrated + cls + loc) / 3)))


def ranking_plan(scores, labels):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    if scores.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid ranking scores/targets")
    order = np.argsort(-scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order]) != 0) + 1] if len(scores) else np.array([], dtype=int)
    return order, starts, labels[order]


def weighted_metrics(plan, weights):
    """Batch exact frequency-weighted AUROC/AP; whole tie groups enter together."""
    order, starts, y = plan
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    if weights.shape[1] != len(order) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Invalid frequency weights")
    result = np.full((len(weights), 2), np.nan)
    if not len(order):
        return result
    sorted_weights = weights[:, order]
    positive = np.add.reduceat(sorted_weights * y, starts, axis=1)
    negative = np.add.reduceat(sorted_weights * (1 - y), starts, axis=1)
    p, n = positive.sum(axis=1), negative.sum(axis=1)
    cum_p, cum_n = positive.cumsum(axis=1), negative.cumsum(axis=1)
    valid = (p > 0) & (n > 0)
    result[valid, 0] = np.sum(positive[valid] * (n[valid, None] - cum_n[valid] + .5 * negative[valid]), axis=1) / (p[valid] * n[valid])
    precision = np.divide(cum_p, cum_p + cum_n, out=np.zeros_like(cum_p), where=(cum_p + cum_n) > 0)
    valid = p > 0
    result[valid, 1] = np.sum(positive[valid] * precision[valid], axis=1) / p[valid]
    return result


def nullable(value):
    return float(value) if np.isfinite(value) else None


def metrics(scores, labels):
    values = weighted_metrics(ranking_plan(scores, labels), np.ones(len(labels)))[0]
    return dict(zip(("auroc", "auprc"), map(nullable, values)))


def interval(values):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    return {"bootstrap_mean": float(valid.mean()) if len(valid) else None,
            "lower_95": float(np.percentile(valid, 2.5, method="linear")) if len(valid) else None,
            "upper_95": float(np.percentile(valid, 97.5, method="linear")) if len(valid) else None,
            "valid_replicates": len(valid), "invalid_replicates": len(values) - len(valid)}


def bootstrap(images, rows, scores, *, replicates=REPLICATES, seed=SEED):
    if not images or len(set(images)) != len(images) or replicates < 1:
        raise ValueError("Invalid bootstrap image inventory/replicates")
    lookup = {image: i for i, image in enumerate(images)}
    ids = np.array([lookup[r["image"]] for r in rows], dtype=int)
    y = np.array([r["incorrect_prediction"] for r in rows])
    plans = {method: ranking_plan(scores[method], y) for method in METHODS}
    rng = np.random.Generator(np.random.PCG64(seed))
    digest = hashlib.sha256()
    samples = np.empty((replicates, len(METHODS), 2))
    for start in range(0, replicates, 64):
        size = min(64, replicates - start)
        draws = rng.integers(0, len(images), size=(size, len(images)), dtype=np.int64)
        digest.update(draws.astype("<i8", copy=False).tobytes())
        multiplicity = np.zeros((size, len(images)), dtype=np.int64)
        np.add.at(multiplicity, (np.arange(size)[:, None], draws), 1)
        weights = multiplicity[:, ids]  # all predictions of an image share its draw multiplicity
        for index, method in enumerate(METHODS):
            samples[start:start + size, index] = weighted_metrics(plans[method], weights)
    return samples, {"seed": seed, "replicates": replicates, "rng": "PCG64", "sampling_unit": "development image",
                     "sampled_images_per_replicate": len(images), "draw_indices_sha256": digest.hexdigest(),
                     "redraw_invalid": False, "percentile_method": "linear"}


def evaluation_tables(rows, scores, samples):
    y = [r["incorrect_prediction"] for r in rows]
    observed = {m: metrics(scores[m], y) for m in METHODS}
    methods = []
    for j, method in enumerate(METHODS):
        row = {"method": method, **observed[method]}
        for k, metric in enumerate(("auroc", "auprc")):
            row.update({f"{metric}_{key}": value for key, value in interval(samples[:, j, k]).items()})
        methods.append(row)
    differences = []
    for first, second in PAIRS:
        a, b = observed[first]["auroc"], observed[second]["auroc"]
        delta = samples[:, METHODS.index(first), 0] - samples[:, METHODS.index(second), 0]
        differences.append({"method_A": first, "method_B": second,
                            "observed_delta_auroc": a - b if a is not None and b is not None else None,
                            **interval(delta)})
    return methods, differences


def diagnostics(rows, scores):
    descriptives, correlations, classes = [], [], []
    arrays = {key: np.array([r[key] for r in rows], dtype=float) for key in SIGNALS}
    for key, values in arrays.items():
        for correct in (1, 0):
            selected = values[[r["correct_iou50"] == correct for r in rows]]
            descriptives.append({"signal": key, "group": "correct" if correct else "incorrect", "count": len(selected),
                "mean": float(selected.mean()) if len(selected) else None,
                "standard_deviation": float(selected.std(ddof=0)) if len(selected) else None,
                **dict(zip(("q1", "median", "q3"), np.quantile(selected, [.25, .5, .75], method="linear").tolist() if len(selected) else [None] * 3))})
    for i, first in enumerate(SIGNALS):
        for second in SIGNALS[i + 1:]:
            a, b = rankdata(arrays[first]), rankdata(arrays[second])
            valid = len(a) >= 2 and np.ptp(a) > 0 and np.ptp(b) > 0
            correlations.append({"signal_A": first, "signal_B": second, "count": len(a),
                                 "spearman_rho": float(np.corrcoef(a, b)[0, 1]) if valid else None,
                                 "status": "defined" if valid else "undefined_constant_or_insufficient"})
    if all(r.get("class_id") not in (None, "") for r in rows):
        identifiers = [int(r["class_id"]) for r in rows]
        for cls in sorted(set(identifiers)):
            mask = np.array([value == cls for value in identifiers])
            target = np.array([r["incorrect_prediction"] for r in rows])[mask]
            for method in (METHODS[1], METHODS[2], METHODS[3], METHODS[7]):
                value = metrics(scores[method][mask], target)["auroc"]
                classes.append({"class_id": cls, "method": method, "prediction_count": int(mask.sum()),
                                "correct": int((target == 0).sum()), "incorrect": int(target.sum()), "auroc": value,
                                "status": "defined" if value is not None else "undefined_single_class"})
    return descriptives, correlations, classes


def run(root):
    root = Path(root).resolve()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError("Existing complete/incomplete risk evaluation; archive explicitly before retry")
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution inputs before DICC evaluation")
    images, rows, hashes = load_inputs(root)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy")}
    provenance = {"experiment": "rq2_risk_evaluation", "status": "running", **git, "input_path": INPUT,
                  "input_sha256": hashes, "development_images": len(images), "prediction_count": len(rows),
                  "correctness_definition": final.TARGET, "bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES,
                  "metric_definitions": DEFINITIONS, "risk_formulas": FORMULAS, "versions": versions,
                  "python": platform.python_version(), "started_utc": datetime.now(timezone.utc).isoformat(),
                  "scope": "development ranking diagnostics; no threshold/model/weight selection"}
    output.mkdir(parents=True, exist_ok=False)
    target = output / "provenance.json"
    rq1._write_json(target, provenance, exclusive=True)
    try:
        scores = risk_scores(rows)
        samples, bootstrap_info = bootstrap(images, rows, scores, replicates=REPLICATES)
        methods, differences = evaluation_tables(rows, scores, samples)
        descriptives, correlations, classes = diagnostics(rows, scores)
        writer = final.comparison.write_csv
        for name, records in (("method_metrics.csv", methods), ("pairwise_auroc_differences.csv", differences),
                              ("signal_descriptives.csv", descriptives), ("signal_correlations.csv", correlations)):
            writer(output / name, records)
        if classes:
            writer(output / "per_class_diagnostics.csv", classes)
        # Persist undefined cells as blank and retain every replicate index; never redraw.
        with (output / "bootstrap_metrics.csv").open("x", newline="", encoding="utf-8") as file:
            csv_writer = csv.DictWriter(file, fieldnames=("replicate", "method", "auroc", "auprc", "auroc_valid", "auprc_valid"))
            csv_writer.writeheader()
            for i in range(len(samples)):
                for j, method in enumerate(METHODS):
                    csv_writer.writerow({"replicate": i, "method": method, "auroc": nullable(samples[i, j, 0]),
                                         "auprc": nullable(samples[i, j, 1]), "auroc_valid": bool(np.isfinite(samples[i, j, 0])),
                                         "auprc_valid": bool(np.isfinite(samples[i, j, 1]))})
        bootstrap_info["validity_by_method"] = [{"method": r["method"], **{k: v for k, v in r.items() if k.endswith("replicates")}} for r in methods]
        rq1._write_json(output / "bootstrap_summary.json", bootstrap_info, exclusive=True)
        incorrect = sum(r["incorrect_prediction"] for r in rows)
        rq1._write_json(output / "summary.json", {"development_images": len(images), "prediction_count": len(rows),
            "correct": len(rows) - incorrect, "incorrect": incorrect, "incorrect_prevalence": incorrect / len(rows),
            "auprc_reference_level": incorrect / len(rows), "metric_definitions": DEFINITIONS,
            "interpretation": "development-set diagnostics; not held-out test generalisation evidence",
            "per_class_diagnostics_available": bool(classes), "threshold_selected": False, "weights_tuned": False}, exclusive=True)
        if load_inputs(root)[2] != hashes or rq1.git_provenance(root) != git:
            raise RuntimeError("Inputs/source changed during evaluation")
        provenance.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                          output_sha256={p.name: rq1._sha(p) for p in output.iterdir() if p != target})
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
        raise RuntimeError("Real development evaluation is DICC-only; --dicc required")
    print(run(find_project_root()))


if __name__ == "__main__":
    main()
