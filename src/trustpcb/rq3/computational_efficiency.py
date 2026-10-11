"""DICC GPU benchmark of frozen development inference; synthetic local tests only."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from trustpcb import rq1
from trustpcb.rq2 import confidence_distribution as raw
from trustpcb.rq2 import transformation_stability as stability
from trustpcb.rq2 import risk_evaluation as risk, weighted_fusion as fusion
from trustpcb.rq2.weighted_final_evaluation import selected_identity
from trustpcb.dataset_config import find_project_root, load_paths
from trustpcb.rq2.artifact_paths import resolve_input

OUTPUT = "runs/rq3/computational_efficiency"
IMAGE_COUNT, WARMUP_CROPS, REPETITIONS = 821, 20, 5
PASSES = {"baseline": 1, "trustpcb": 12}
PROTOCOL = {
    "population": "821 frozen development-manifest images; deterministic manifest order",
    "batch": 1, "warmup": "first 20 development crops for each pipeline; untimed",
    "order": "repetition 1 baseline then TrustPCB, repeated for all five repetitions",
    "timing": "wall-clock perf_counter; CUDA synchronize after warmup and immediately before/after every repetition",
    "included": "crop loading/decode, preprocessing, detector inference, postprocessing; TrustPCB also transforms, matching, aggregation, Beta application, weighted risk",
    "excluded": "model loading/startup, provenance validation, file hashing, output-file writing",
    "standard_deviation": "sample standard deviation across five repetitions, ddof=1",
    "reference_inclusion": "original post-NMS confidence >= 0.01; original/transformed extraction floor 0.001",
    "weights": [0.89, 0.01, 0.10], "transformed_passes": 11,
}


def validate_specs():
    expected = (
        ("brightness_0.90", "brightness", "factor", .90),
        ("brightness_1.10", "brightness", "factor", 1.10),
        ("contrast_0.90", "contrast", "factor", .90),
        ("contrast_1.10", "contrast", "factor", 1.10),
        ("blur_0.6", "blur", "sigma", .6),
        ("rotation_-2", "rotation", "degrees", -2),
        ("rotation_+2", "rotation", "degrees", 2),
        ("translation_x_-0.02", "translation", "fraction", -.02),
        ("translation_x_+0.02", "translation", "fraction", .02),
        ("translation_y_-0.02", "translation", "fraction", -.02),
        ("translation_y_+0.02", "translation", "fraction", .02))
    if len(stability.SPECS) != 11:
        raise ValueError("Expected eleven frozen transforms")
    for spec, (identifier, family, key, value) in zip(stability.SPECS, expected):
        required = {"id": identifier, "family": family, key: value}
        if family == "translation":
            required["axis"] = identifier.split("_")[1]
        if spec != required:
            raise ValueError("Frozen transformation specifications differ")


def summarize(records, image_count=IMAGE_COUNT):
    if image_count <= 0:
        raise ValueError("Expected positive crop count")
    summary = {}
    for name in PASSES:
        rows = [r for r in records if r["pipeline"] == name]
        if ([r["repetition"] for r in rows] != list(range(1, REPETITIONS+1))
                or any(r["crops"] != image_count or r["detector_passes"] != image_count*PASSES[name] for r in rows)):
            raise ValueError("Incomplete/mismatched repetition inventory or pass counts")
        values = np.asarray([r["seconds"] for r in rows], dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Expected finite positive runtimes")
        mean = float(values.mean())
        summary[name] = {"total_seconds_per_repetition": values.tolist(), "mean_seconds": mean,
            "standard_deviation_seconds": float(values.std(ddof=1)),
            "mean_ms_per_crop": mean*1000/image_count, "throughput_crops_per_second": image_count/mean,
            "detector_passes_per_crop": PASSES[name], "detector_passes_per_repetition": image_count*PASSES[name]}
    ratio = summary["trustpcb"]["mean_seconds"]/summary["baseline"]["mean_seconds"]
    return {"pipelines": summary, "relative_overhead": ratio, "percentage_increase": (ratio-1)*100}


def measure(images, pipelines, synchronize, *, clock=perf_counter):
    """Injectable orchestration: no model imports or output writing here."""
    if len(images) < WARMUP_CROPS or len(set(images)) != len(images) or set(pipelines) != set(PASSES):
        raise ValueError("Expected unique crop inventory and both pipelines")
    for name in PASSES:
        for image in images[:WARMUP_CROPS]:
            if pipelines[name](image) != PASSES[name]:
                raise RuntimeError("Unexpected warmup detector pass count")
        synchronize()
    records = []
    for repetition in range(1, REPETITIONS+1):
        for name in PASSES:
            synchronize()
            start = clock()
            passes = sum(pipelines[name](image) for image in images)
            synchronize()
            elapsed = clock()-start
            if passes != len(images)*PASSES[name]:
                raise RuntimeError("Unexpected timed detector pass count")
            records.append({"pipeline": name, "repetition": repetition, "crops": len(images),
                "seconds": elapsed, "detector_passes": passes,
                "milliseconds_per_crop": elapsed*1000/len(images),
                "throughput_crops_per_second": len(images)/elapsed if elapsed > 0 else None})
    return records


def predict(model, source):
    results = model.predict(source=source, **raw.SETTINGS)
    if len(results) != 1:
        raise RuntimeError("Expected one batch-1 result")
    for key, value in raw.SETTINGS.items():
        if not raw.rq2_detector_training._argument_matches(key, getattr(model.predictor.args, key, None), value):
            raise RuntimeError(f"Frozen prediction setting differs: {key}")
    result = results[0]
    detections = []
    if result.boxes is not None:
        for box, confidence, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            detections.append({"box": box, "confidence": confidence, "class_id": int(cls)})
    return detections, tuple(result.orig_shape)


def full_crop(model, source, image, beta, *, pixel_loader):
    detections, shape = predict(model, str(source))
    references = [{**r, "prediction_id": f"{image}#{i}"} for i, r in enumerate(detections) if r["confidence"] >= .01]
    pixels = pixel_loader(source)
    height, width = pixels.shape[:2]
    if shape != (height, width):
        raise RuntimeError("Original image dimensions differ from frozen RGB/EXIF decoding")
    audits = {r["prediction_id"]: [] for r in references}
    for spec in stability.SPECS:
        view, _, _ = stability.transform(pixels, spec)
        transformed, transformed_shape = predict(model, view)
        if transformed_shape != (height, width):
            raise RuntimeError("Transformed image dimensions differ")
        _, matches = stability.match_transform(references, transformed, spec, width, height)
        for audit in matches:
            audits[audit["prediction_id"]].append(audit)
    if references:
        calibrated = stability.final.comparison.apply_calibrator(beta, [r["confidence"] for r in references])
        signals = [{"calibrated_confidence": float(c), **stability.aggregate(audits[r["prediction_id"]])}
                   for r, c in zip(references, calibrated)]
        risks = fusion.weighted_risk(fusion.component_matrix(signals), (89, 1, 10))
        if not np.isfinite(risks).all():
            raise RuntimeError("Undefined frozen reliability score")
    return 1+len(stability.SPECS)


def adapter(root, sources, beta):
    if os.name == "nt":
        raise RuntimeError("Real benchmark is DICC GPU only")
    import torch
    from ultralytics import YOLO
    from PIL import Image, ImageOps
    if not torch.cuda.is_available():
        raise RuntimeError("DICC CUDA GPU required; CPU fallback prohibited")
    model = YOLO(str(resolve_input(root, raw.CHECKPOINT)))
    if model.names != raw.rq2_detector_training.NAMES:
        raise ValueError("Frozen detector class mapping differs")
    def load_pixels(source):
        with Image.open(source) as opened:
            return np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))[:, :, ::-1].copy()
    def baseline(image):
        predict(model, str(sources[image]))
        return 1
    def full(image):
        return full_crop(model, sources[image], image, beta, pixel_loader=load_pixels)
    return {"baseline": baseline, "trustpcb": full}, lambda: torch.cuda.synchronize(0), model


def load_inputs(root):
    validate_specs()
    identity = selected_identity(root)
    images, _, hashes = risk.load_inputs(root)  # existing development text-artifact validation
    checked, _, checkpoint_hashes, checkpoint = stability.load_inputs(root)
    if images != checked or any(hashes.get(k) != v for k, v in checkpoint_hashes.items()):
        raise ValueError("Frozen development evidence differs between validators")
    artifact = rq1._read_json(resolve_input(root, f"{stability.final.OUTPUT}/final_calibrator.json"))
    beta = {"method": "Beta", "parameters": [float(artifact[k]) for k in ("a", "b", "c")]}
    return images, hashes, identity, checkpoint, beta


def run(root, dataset, *, adapter_factory=adapter):
    root, dataset = Path(root).resolve(), Path(dataset).resolve()
    output = root/OUTPUT
    if output.exists():
        raise FileExistsError("Existing partial/completed benchmark; explicit archival required")
    git = rq1.git_provenance(root)
    if git["git_dirty"]:
        raise RuntimeError("Commit/resolve execution inputs before DICC benchmark")
    images, hashes, identity, checkpoint, beta = load_inputs(root)
    if len(images) != IMAGE_COUNT or len(set(images)) != IMAGE_COUNT:
        raise ValueError("Expected exactly 821 development crops")
    sources = {image: (dataset/image).resolve() for image in images}
    if any(not p.is_relative_to(dataset) or not p.is_file() for p in sources.values()):
        raise ValueError("Missing/escaped development image")
    image_hashes = {image: rq1._sha(path) for image, path in sources.items()}
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "Pillow", "ultralytics", "torch")}
    if versions["ultralytics"] != raw.VERSION:
        raise RuntimeError(f"Expected ultralytics=={raw.VERSION}")
    pipelines, synchronize, model = adapter_factory(root, sources, beta)
    metadata = {"experiment": "rq3_computational_efficiency", "status": "running", **git,
        "protocol": PROTOCOL, "repetitions": REPETITIONS, "warmup_images": images[:WARMUP_CROPS],
        "development_images": images, "input_sha256": hashes, "source_image_sha256": image_hashes,
        "selected_weight_identity": identity, "checkpoint_identity": checkpoint, "beta": beta,
        "transformations": stability.SPECS, "transformation_rules": stability.RULES,
        "prediction_settings": raw.SETTINGS, "versions": versions, "python": platform.python_version(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "test_population_accessed": False,
        "fitting_performed": False, "threshold_selection_performed": False}
    output.mkdir(parents=True, exist_ok=False)
    target = output/"provenance.json"
    rq1._write_json(target, metadata, exclusive=True)
    try:
        records = measure(images, pipelines, synchronize)
        # Audit/hash/output work deliberately follows the synchronized timed regions.
        if load_inputs(root) != (images, hashes, identity, checkpoint, beta) or rq1.git_provenance(root) != git:
            raise RuntimeError("Frozen evidence/source changed during benchmark")
        if any(rq1._sha(sources[image]) != digest for image, digest in image_hashes.items()):
            raise RuntimeError("Development images changed during benchmark")
        device = model.predictor.device
        if device.type != "cuda" or device.index not in (None, 0):
            raise RuntimeError("Benchmark did not use requested GPU 0")
        import torch
        metadata["gpu"] = {"resolved_device": str(device), "name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "effective_prediction_settings": {k: getattr(model.predictor.args, k) for k in raw.SETTINGS}}
        risk.final.comparison.write_csv(output/"repetition_timings.csv", records)
        rq1._write_json(output/"summary.json", {**summarize(records, len(images)), "protocol": PROTOCOL}, exclusive=True)
        metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
            output_sha256={p.name: rq1._sha(p) for p in output.iterdir() if p != target})
        rq1._write_json(target, metadata)
        return output
    except BaseException:
        metadata["status"] = "incomplete"
        rq1._write_json(target, metadata)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicc", action="store_true")
    args = parser.parse_args(argv)
    if os.name == "nt" or not args.dicc:
        raise RuntimeError("Real computational benchmark is DICC GPU only; --dicc required")
    root = find_project_root()
    paths = load_paths(root)
    if paths.project_root != root:
        raise ValueError("project_root must match this checkout")
    print(run(root, paths.dataset_root))


if __name__ == "__main__":
    main()
