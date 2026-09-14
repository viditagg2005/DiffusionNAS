from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    # Nearest-rank percentile: conservative for small benchmark samples.
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _scheduler(pipe: Any, name: str) -> None:
    if name == "ddim":
        from diffusers import DDIMScheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif name == "dpm_solver++":
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++"
        )


def run(config: dict[str, Any], output_dir: Path) -> None:
    import torch
    from diffusers import StableDiffusionPipeline

    runtime = config["runtime"]
    dtype = torch.float16 if runtime["precision"] == "float16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        config["model"]["id"], torch_dtype=dtype, use_safetensors=True
    ).to(runtime["device"])
    pipe.set_progress_bar_config(disable=True)
    _scheduler(pipe, config["inference"]["scheduler"])

    helper = None
    if config["method"] == "deepcache":
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(
            cache_interval=config["deepcache"]["interval"],
            cache_branch_id=config["deepcache"]["branch_id"],
            skip_mode=config["deepcache"].get("skip_mode", "uniform"),
        )
        helper.enable()

    call_args = {
        "height": config["model"]["height"],
        "width": config["model"]["width"],
        "num_inference_steps": config["inference"]["num_inference_steps"],
        "guidance_scale": config["inference"]["guidance_scale"],
    }
    if config["method"] == "autodiffusion":
        call_args["timesteps"] = config["inference"]["timesteps"]

    warmup_prompt = config["prompts"][0]
    for warmup_index in range(runtime["warmup_runs"]):
        generator = torch.Generator(device=runtime["device"]).manual_seed(config["seeds"][0] + warmup_index)
        pipe(warmup_prompt, generator=generator, output_type="latent", **call_args)
    torch.cuda.synchronize()

    image_dir = output_dir / "images"
    if config.get("save_images", True):
        image_dir.mkdir(exist_ok=True)
    measurements: list[dict[str, Any]] = []
    measurement_path = output_dir / "measurements.jsonl"
    with measurement_path.open("w", encoding="utf-8") as stream:
        for prompt_index, prompt in enumerate(config["prompts"]):
            for seed in config["seeds"]:
                for repetition in range(runtime["repetitions"]):
                    torch.cuda.reset_peak_memory_stats()
                    generator = torch.Generator(device=runtime["device"]).manual_seed(seed)
                    torch.cuda.synchronize()
                    started = time.perf_counter_ns()
                    result = pipe(prompt, generator=generator, output_type="pil", **call_args)
                    torch.cuda.synchronize()
                    latency_ms = (time.perf_counter_ns() - started) / 1_000_000
                    row = {
                        "schema_version": 1,
                        "method": config["method"],
                        "model_id": config["model"]["id"],
                        "prompt_index": prompt_index,
                        "seed": seed,
                        "repetition": repetition,
                        "latency_ms": latency_ms,
                        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
                        "num_inference_steps": config["inference"]["num_inference_steps"],
                    }
                    measurements.append(row)
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    if config.get("save_images", True) and repetition == 0:
                        result.images[0].save(image_dir / f"p{prompt_index:04d}_s{seed}.png")

    latencies = [row["latency_ms"] for row in measurements]
    summary = {
        "schema_version": 1,
        "status": "succeeded",
        "method": config["method"],
        "sample_count": len(measurements),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": statistics.median(latencies),
            "p90": _percentile(latencies, 0.9),
        },
        "peak_allocated_mb": max(row["peak_allocated_mb"] for row in measurements),
        "peak_reserved_mb": max(row["peak_reserved_mb"] for row in measurements),
        "quality_metrics": {},
        "energy_joules_per_image": None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if helper is not None:
        helper.disable()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run(config, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
