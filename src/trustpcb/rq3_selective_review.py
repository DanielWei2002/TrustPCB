"""Development-only selective review of existing frozen RQ2 text artifacts."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform

import numpy as np

from trustpcb import rq1, rq2_risk_evaluation as risk, rq2_weighted_fusion as fusion
from trustpcb.rq2_weighted_final_evaluation import selected_identity
from trustpcb.dataset_config import find_project_root

OUTPUT = "runs/rq3/development_selective_review"
SEED, REPLICATES = 24209199, 10000
BUDGETS = (5, 10, 20)
METHODS = ("raw_confidence_risk", "calibrated_confidence_risk",
           "class_consistency_risk", "localisation_stability_risk", "weighted_trustpcb_risk")
REFERRAL_BOOTSTRAP_METRICS = ("workload", "error_capture", "random_expected_error_capture",
    "automatic_predictions", "automatic_incorrect", "remaining_automatic_error_rate", "absolute_error_reduction")
DEFINITIONS = {
    "scope": "development only; 821 manifest images are 821 reviewable crops",
    "target": "incorrect_prediction = 1 - correct_iou50; existing same-class one-to-one IoU >= 0.50",
    "inclusion": "existing raw confidence >= 0.01 population only",
    "scores": "higher = higher risk; 1-raw, 1-calibrated, 1-class, 1-localisation, frozen 89/1/10 fusion",
    "crop_risk": "maximum retained prediction risk; empty crop is undefined and never referred",
    "referral": "nonempty crop AND crop_risk > threshold",
    "threshold": "largest achievable count <= floor(all crops * budget / 100); whole ties, no ID or label tie-break",
    "curve": "retain ascending risk with entire equal-score groups; coverage=retained/total predictions; risk=retained incorrect/retained",
    "aurc": "right-endpoint step integral over coverage; sum(delta_coverage * retained_error_rate); empty curve undefined",
    "error_capture": "incorrect predictions in referred crops / all incorrect predictions; undefined without errors",
    "automatic_error": "non-referred incorrect / non-referred predictions; undefined with zero automatic predictions",
    "absolute_error_reduction": "no-referral error rate minus automatic error rate; probability-point difference; replicate-specific reference",
    "paired_difference": "weighted TrustPCB minus reference on the same replicate; negative favours weighted for AURC/automatic error, positive for capture",
    "random": "uniform selection without replacement among nonempty crops at achieved workload; analytic expected capture=k/nonempty crops",
    "bootstrap": "paired whole-image resampling with replacement; fixed original development thresholds; no threshold rederivation; percentile CIs; no redraw",
}


def scores_for(rows):
    original = risk.risk_scores(rows)
    return dict(zip(METHODS, (original["raw_confidence_risk"], original["calibrated_confidence_risk"],
        original["class_consistency_risk"], original["localisation_risk"],
        fusion.weighted_risk(fusion.component_matrix(rows), (89, 1, 10)))))


def prepare(images, rows, scores):
    if not images or len(set(images)) != len(images):
        raise ValueError("Expected nonempty unique crop inventory")
    lookup = {image: i for i, image in enumerate(images)}
    if any(r["image"] not in lookup for r in rows):
        raise ValueError("Foreign image outside development inventory")
    identities = [r["prediction_id"] for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate prediction identity")
    labels = np.asarray([1-r["correct_iou50"] for r in rows], dtype=float)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid frozen primary correctness")
    ids = np.asarray([lookup[r["image"]] for r in rows], dtype=int)
    counts = np.bincount(ids, minlength=len(images))
    errors = np.bincount(ids, weights=labels, minlength=len(images))
    plans, crops = {}, {}
    for method in METHODS:
        values = np.asarray(scores[method], dtype=float)
        if values.shape != labels.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Invalid prediction risk")
        crop = np.full(len(images), -np.inf)
        np.maximum.at(crop, ids, values)
        crops[method] = crop
        order = np.argsort(values, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(values[order]) != 0), len(order)-1] if len(order) else np.array([], dtype=int)
        plans[method] = (order, ends, values[order][ends])
    return ids, labels, counts, errors, crops, plans


def curve(plan, labels, weights):
    order, ends, boundaries = plan
    w = np.asarray(weights, dtype=float)
    if w.shape != labels.shape or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Invalid prediction multiplicities")
    total = float(w.sum())
    records = [{"coverage": 0.0, "retained_predictions": 0.0, "retained_incorrect": 0.0,
                "empirical_risk": None, "max_retained_score": None}]
    if not total:
        return records, None
    retained = np.cumsum(w[order])[ends]
    incorrect = np.cumsum((w*labels)[order])[ends]
    previous, area = 0.0, 0.0
    for count, error, boundary in zip(retained, incorrect, boundaries):
        if count == previous:
            continue
        rate = float(error/count)
        area += (count-previous)/total * rate
        records.append({"coverage": float(count/total), "retained_predictions": float(count),
            "retained_incorrect": float(error), "empirical_risk": rate, "max_retained_score": float(boundary)})
        previous = count
    return records, float(area)


def threshold_for(crop_risks, budget_percent):
    values = np.asarray(crop_risks, dtype=float)
    if values.ndim != 1 or not len(values) or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("Invalid crop risks; empty crops must use -inf")
    if budget_percent not in BUDGETS:
        raise ValueError("Only predeclared review budgets permitted")
    target = len(values)*budget_percent//100
    finite = np.sort(values[np.isfinite(values)])[::-1]
    # The first excluded risk is the strict boundary. Ties are excluded together.
    threshold = float(finite[target]) if target < len(finite) else -1.0
    achieved = int(np.sum(np.isfinite(values) & (values > threshold)))
    return {"budget_percent": budget_percent, "nominal_crops": target, "threshold": threshold,
            "achieved_crops": achieved, "achieved_workload": achieved/len(values)}


def referral_metrics(crop_risks, counts, errors, threshold, multiplicity):
    m = np.asarray(multiplicity, dtype=float)
    eligible = counts > 0
    referred = eligible & (crop_risks > threshold)
    denominator = float(m.sum())
    if denominator <= 0 or m.shape != counts.shape or not np.isfinite(m).all() or np.any(m < 0):
        raise ValueError("Invalid whole-image multiplicities")
    selected = float(m[referred].sum())
    total_errors = float(np.dot(m, errors))
    captured = float(np.dot(m*referred, errors))
    predictions = float(np.dot(m*referred, counts))
    total_predictions = float(np.dot(m, counts))
    automatic_predictions = total_predictions-predictions
    automatic_incorrect = total_errors-captured
    reference_rate = total_errors/total_predictions if total_predictions else None
    automatic_rate = automatic_incorrect/automatic_predictions if automatic_predictions else None
    nonempty = float(m[eligible].sum())
    probability = selected/nonempty if nonempty else 0.0
    return {"referred_crops": selected, "workload": selected/denominator,
        "referred_predictions": predictions, "captured_incorrect": captured,
        "error_capture": captured/total_errors if total_errors else None,
        "referred_error_rate": captured/predictions if predictions else None,
        "automatic_predictions": automatic_predictions, "automatic_incorrect": automatic_incorrect,
        "no_referral_error_rate": reference_rate, "remaining_automatic_error_rate": automatic_rate,
        "absolute_error_reduction": reference_rate-automatic_rate if automatic_rate is not None else None,
        "random_expected_error_capture": probability if total_errors else None,
        "random_expected_captured_incorrect": probability*total_errors,
        "random_expected_referred_predictions": probability*float(np.dot(m, counts))}


def tables(images, rows, scores):
    ids, labels, counts, errors, crops, plans = prepare(images, rows, scores)
    curves, summaries, referrals, inventory = [], [], [], []
    thresholds = {}
    for method in METHODS:
        records, area = curve(plans[method], labels, np.ones(len(rows)))
        curves.extend({"method": method, **r} for r in records)
        summaries.append({"method": method, "aurc": area})
        thresholds[method] = [threshold_for(crops[method], b) for b in BUDGETS]
        referrals.extend({"method": method, **t, **referral_metrics(crops[method], counts, errors,
            t["threshold"], np.ones(len(images)))} for t in thresholds[method])
        inventory.extend({"method": method, "image": image, "prediction_count": int(counts[i]),
            "incorrect_predictions": int(errors[i]), "crop_risk": float(crops[method][i]) if counts[i] else None,
            "referral_eligible": bool(counts[i])} for i, image in enumerate(images))
    return curves, summaries, referrals, inventory, thresholds


def bootstrap(images, rows, scores, thresholds, *, replicates=REPLICATES, seed=SEED):
    """Optional uncertainty module; thresholds remain fixed across shared image draws."""
    if type(replicates) is not int or replicates < 1:
        raise ValueError("Expected positive replicate count")
    ids, labels, counts, errors, crops, plans = prepare(images, rows, scores)
    rng = np.random.Generator(np.random.PCG64(seed))
    digest, values = hashlib.sha256(), {}
    for method in METHODS:
        values[(method, None, "aurc")] = []
        for t in thresholds[method]:
            for metric in REFERRAL_BOOTSTRAP_METRICS:
                values[(method, t["budget_percent"], metric)] = []
    for _ in range(replicates):
        draw = rng.integers(0, len(images), size=len(images), dtype=np.int64)
        digest.update(draw.astype("<i8", copy=False).tobytes())
        multiplicity = np.bincount(draw, minlength=len(images))
        for method in METHODS:
            _, area = curve(plans[method], labels, multiplicity[ids])
            values[(method, None, "aurc")].append(np.nan if area is None else area)
            for t in thresholds[method]:
                stats = referral_metrics(crops[method], counts, errors, t["threshold"], multiplicity)
                for metric in REFERRAL_BOOTSTRAP_METRICS:
                    value = stats[metric]
                    values[(method, t["budget_percent"], metric)].append(np.nan if value is None else value)
    intervals = [{"method": method, "budget_percent": budget, "metric": metric, **risk.interval(v)}
                 for (method, budget, metric), v in values.items()]
    observed = {}
    for method in METHODS:
        observed[(method, None, "aurc")] = curve(plans[method], labels, np.ones(len(rows)))[1]
        for t in thresholds[method]:
            stats = referral_metrics(crops[method], counts, errors, t["threshold"], np.ones(len(images)))
            for metric in ("error_capture", "remaining_automatic_error_rate"):
                observed[(method, t["budget_percent"], metric)] = stats[metric]
    pairs = []
    for reference in METHODS[:2]:
        for budget, metric in ((None, "aurc"), *((b, m) for b in BUDGETS
                for m in ("error_capture", "remaining_automatic_error_rate"))):
            a, b = (METHODS[-1], budget, metric), (reference, budget, metric)
            first, second = observed[a], observed[b]
            pairs.append({"method_A": METHODS[-1], "method_B": reference, "budget_percent": budget,
                "metric": metric, "observed_difference": first-second if first is not None and second is not None else None,
                **risk.interval(np.asarray(values[a])-np.asarray(values[b]))})
    return intervals, {"seed": seed, "replicates": replicates, "rng": "PCG64",
        "sampling_unit": "whole development image/crop", "draw_indices_sha256": digest.hexdigest(),
        "thresholds_rederived": False, "redraw_invalid": False, "paired_differences": pairs}


def load_inputs(root):
    identity = selected_identity(root)  # evidence bytes only; never final-test inputs
    images, rows, hashes = risk.load_inputs(root)
    return images, rows, hashes, identity


def extend_reporting(root):
    """Separate reports bound to completed evidence; never derive thresholds again."""
    root = Path(root).resolve()
    original = root/OUTPUT
    output = original/"reporting_extension"
    if output.exists():
        raise FileExistsError("Existing reporting extension; archive explicitly before retry")
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution inputs before DICC reporting extension")
    source = rq1._read_json(original/"provenance.json")
    if (source.get("status") != "complete" or source.get("bootstrap_requested") is not True
            or source.get("bootstrap_seed") != SEED or source.get("bootstrap_replicates") != REPLICATES):
        raise ValueError("Expected completed development evidence with frozen 10000-replicate bootstrap")
    images, rows, hashes, identity = load_inputs(root)
    if hashes != source.get("input_sha256") or identity != source.get("selected_weight_identity"):
        raise ValueError("Development inputs differ from completed evidence")
    evidence = {"provenance.json": rq1._sha(original/"provenance.json")}
    for name, digest in source["output_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("Invalid completed output path")
        if rq1._sha(original/name) != digest:
            raise ValueError(f"Completed evidence hash mismatch: {name}")
        evidence[name] = digest
    if not {"development_thresholds.json", "bootstrap_summary.json"}.issubset(evidence):
        raise ValueError("Completed evidence lacks bound thresholds/bootstrap summary")
    saved = rq1._read_json(original/"development_thresholds.json")
    if saved.get("comparison") != ">" or saved.get("test_retuning_permitted") is not False:
        raise ValueError("Expected frozen scalar strict-greater referral protocol")
    thresholds = saved["thresholds"]
    if set(thresholds) != set(METHODS):
        raise ValueError("Frozen threshold method inventory differs")
    for method in METHODS:
        entries = thresholds[method]
        if ([t["budget_percent"] for t in entries] != list(BUDGETS)
                or any(not np.isfinite(t["threshold"]) for t in entries)):
            raise ValueError("Frozen threshold budget/value inventory differs")
    scores = scores_for(rows)
    _, _, counts, errors, crops, _ = prepare(images, rows, scores)
    referrals = [{"method": method, **t, **referral_metrics(crops[method], counts, errors,
        t["threshold"], np.ones(len(images)))} for method in METHODS for t in thresholds[method]]
    if any(r["referred_crops"] != r["achieved_crops"] for r in referrals):
        raise ValueError("Saved thresholds no longer reproduce completed workload")
    metadata = {"experiment": "rq3_development_reporting_extension", "status": "running", **git,
        "source_evidence_sha256": evidence, "input_sha256": hashes, "selected_weight_identity": identity,
        "definitions": DEFINITIONS, "thresholds_rederived": False, "bootstrap_seed": SEED,
        "bootstrap_replicates": REPLICATES, "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
        "started_utc": datetime.now(timezone.utc).isoformat(), "test_population_accessed": False}
    output.mkdir(exist_ok=False)
    target = output/"provenance.json"
    rq1._write_json(target, metadata, exclusive=True)
    try:
        intervals, info = bootstrap(images, rows, scores, thresholds)
        previous = rq1._read_json(original/"bootstrap_summary.json")
        if any(info.get(k) != previous.get(k) for k in ("seed", "replicates", "draw_indices_sha256")):
            raise ValueError("Reporting extension bootstrap draws differ from completed evidence")
        writer = risk.final.comparison.write_csv
        writer(output/"crop_referral_metrics.csv", referrals)
        writer(output/"bootstrap_intervals.csv", intervals)
        writer(output/"paired_bootstrap_differences.csv", info["paired_differences"])
        rq1._write_json(output/"bootstrap_summary.json", info, exclusive=True)
        if (any(rq1._sha(original/name) != digest for name, digest in evidence.items())
                or load_inputs(root)[2:] != (hashes, identity) or rq1.git_provenance(root) != git):
            raise RuntimeError("Completed evidence/inputs changed during reporting extension")
        metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
            output_sha256={p.name: rq1._sha(p) for p in output.iterdir() if p != target})
        rq1._write_json(target, metadata)
        return output
    except BaseException:
        metadata["status"] = "incomplete"
        rq1._write_json(target, metadata)
        raise


def run(root, *, with_bootstrap=False):
    root = Path(root).resolve()
    output = root/OUTPUT
    if output.exists():
        raise FileExistsError("Existing partial/completed selective review; archive explicitly before retry")
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution inputs before DICC development evaluation")
    images, rows, hashes, identity = load_inputs(root)
    provenance = {"experiment": "rq3_development_selective_review", "status": "running", **git,
        "input_sha256": hashes, "selected_weight_identity": identity, "weights": [0.89, 0.01, 0.10],
        "development_images": len(images), "prediction_count": len(rows), "definitions": DEFINITIONS,
        "budgets_percent": BUDGETS, "bootstrap_requested": with_bootstrap,
        "bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES if with_bootstrap else 0,
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
        "python": platform.python_version(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "test_population_accessed": False, "fitting_performed": False, "inference_performed": False}
    output.mkdir(parents=True, exist_ok=False)
    target = output/"provenance.json"
    rq1._write_json(target, provenance, exclusive=True)
    try:
        scores = scores_for(rows)
        curves, summaries, referrals, inventory, thresholds = tables(images, rows, scores)
        writer = risk.final.comparison.write_csv
        for name, records in (("prediction_risk_coverage.csv", curves), ("aurc.csv", summaries),
                              ("crop_referral_metrics.csv", referrals), ("crop_risks.csv", inventory)):
            writer(output/name, records)
        rq1._write_json(output/"development_thresholds.json", {"thresholds": thresholds,
            "comparison": ">", "derived_from": "development scores only", "test_retuning_permitted": False,
            "definitions": DEFINITIONS}, exclusive=True)
        if with_bootstrap:
            intervals, info = bootstrap(images, rows, scores, thresholds)
            writer(output/"bootstrap_intervals.csv", intervals)
            writer(output/"paired_bootstrap_differences.csv", info["paired_differences"])
            rq1._write_json(output/"bootstrap_summary.json", info, exclusive=True)
        rq1._write_json(output/"summary.json", {"development_images": len(images), "prediction_count": len(rows),
            "empty_crops": len(images)-len({r["image"] for r in rows}), "definitions": DEFINITIONS,
            "interpretation": "development protocol analysis; not independent test generalisation evidence"}, exclusive=True)
        if load_inputs(root)[2:] != (hashes, identity) or rq1.git_provenance(root) != git:
            raise RuntimeError("Frozen inputs/source changed during development evaluation")
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
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--extend-report", action="store_true", help="Extend completed evidence with fixed saved thresholds and shared bootstrap")
    args = parser.parse_args(argv)
    if os.name == "nt" or not args.dicc:
        raise RuntimeError("Real development selective review is DICC-only; --dicc required")
    print(extend_reporting(find_project_root()) if args.extend_report
          else run(find_project_root(), with_bootstrap=args.bootstrap))


if __name__ == "__main__":
    main()
