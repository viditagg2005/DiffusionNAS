#!/usr/bin/env python3
"""Score a set of runs with every quality metric used by the study.

For each succeeded run under the given roots:
- CLIP prompt alignment and safety-black count (``quality_metrics.clip_score``),
- CLIP image fidelity to the reference run's image for the same prompt and seed,
- optional KID/FID against the reference run's images ("relative") and
  against real COCO images ("vs real").

The reference run is the 20-step DDIM guidance-7.5 fp16 run without
architecture edits found among the inputs (or ``--reference-run``). Metrics
are written into each run's ``summary.json``; Inception features are cached
per run in ``inception_features.npy``. Generation latency is never touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas.quality import DEFAULT_CLIP_MODEL, ClipScorer, image_records  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_reference(config: dict) -> bool:
    inference = config["inference"]
    architecture = config.get("architecture") or {}
    return (
        config["method"] == "baseline"
        and inference["scheduler"] == "ddim"
        and inference["num_inference_steps"] == 20
        and inference["guidance_scale"] == 7.5
        and not architecture.get("removed_units")
        and architecture.get("weight_precision", "fp16") == "fp16"
    )


def discover(roots: list[Path]) -> list[Path]:
    runs = []
    for root in roots:
        for manifest in sorted(root.glob("**/manifest.json")):
            if read(manifest).get("status") == "succeeded" and (manifest.parent / "images").is_dir():
                runs.append(manifest.parent)
    return runs


def cached_features(run_dir: Path, device: str) -> np.ndarray:
    from diffusionnas.distribution import inception_features

    cache = run_dir / "inception_features.npy"
    if cache.exists():
        return np.load(cache)
    features = inception_features([record.image_path for record in image_records(run_dir)], device=device)
    np.save(cache, features)
    return features


def real_features(real_dir: Path, device: str) -> np.ndarray:
    from diffusionnas.distribution import inception_features

    cache = real_dir / "inception_features.npy"
    if cache.exists():
        return np.load(cache)
    paths = sorted(p for p in real_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    features = inception_features(paths, device=device)
    np.save(cache, features)
    return features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--real-dir", type=Path, help="directory of real COCO images for KID/FID vs real")
    parser.add_argument("--distribution", action="store_true", help="also compute KID/FID (needs pytorch-fid)")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    runs = discover(args.runs_root)
    if not runs:
        raise SystemExit("no succeeded runs with images found")
    reference = args.reference_run
    if reference is None:
        candidates = [run for run in runs if is_reference(read(run / "config.json"))]
        if len(candidates) != 1:
            raise SystemExit(f"expected exactly one reference run, found {len(candidates)}; pass --reference-run")
        reference = candidates[0]
    reference = reference.resolve()
    print(f"reference run: {reference.name}")

    scorer = ClipScorer(model_id=args.clip_model, device=args.device, batch_size=args.batch_size)
    for run in runs:
        summary = read(run / "summary.json")
        if "black_images" not in summary.get("quality_metrics", {}):
            scorer.score_run(run)
        scorer.fidelity_run(run, reference)
    del scorer

    if args.distribution:
        from diffusionnas.distribution import distribution_metrics

        reference_features = cached_features(reference, args.device)
        real = real_features(args.real_dir, args.device) if args.real_dir else None
        for run in runs:
            features = cached_features(run, args.device)
            summary = read(run / "summary.json")
            metrics = summary.setdefault("quality_metrics", {})
            if run.resolve() != reference:
                metrics["relative_distribution"] = distribution_metrics(features, reference_features)
            if real is not None:
                metrics["real_distribution"] = distribution_metrics(features, real)
            (run / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for run in runs:
        metrics = read(run / "summary.json")["quality_metrics"]
        line = f"{run.name}: CLIP {metrics['clip_score']['mean']:.3f}  fidelity {metrics['clip_image_fidelity']['mean']:.2f}  black {metrics['black_images']}"
        if "relative_distribution" in metrics:
            line += f"  rKID {metrics['relative_distribution']['kid']['mean_x1000']:.2f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
