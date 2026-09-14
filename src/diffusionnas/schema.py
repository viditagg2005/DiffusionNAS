from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
METHODS = {"baseline", "autodiffusion", "deepcache"}
SD15_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {config_path}: {exc}") from exc
    validate_config(data)
    return data


def validate_config(config: dict[str, Any]) -> None:
    errors: list[str] = []
    method = config.get("method")
    if method not in METHODS:
        errors.append(f"method must be one of {sorted(METHODS)}")

    model = config.get("model", {})
    if model.get("id") != SD15_MODEL:
        errors.append(f"model.id must be the locked SD1.5 repository: {SD15_MODEL}")
    if model.get("height") != 512 or model.get("width") != 512:
        errors.append("the initial matched benchmark is locked to 512x512")

    prompts = config.get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        errors.append("prompts must be a non-empty list of strings")

    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) and s >= 0 for s in seeds):
        errors.append("seeds must be a non-empty list of non-negative integers")

    inference = config.get("inference", {})
    steps = inference.get("num_inference_steps")
    if not isinstance(steps, int) or steps < 1:
        errors.append("inference.num_inference_steps must be a positive integer")
    if inference.get("scheduler") not in {"ddim", "dpm_solver++"}:
        errors.append("inference.scheduler must be 'ddim' or 'dpm_solver++'")

    if method == "autodiffusion":
        timesteps = inference.get("timesteps")
        if inference.get("scheduler") != "dpm_solver++":
            errors.append("AutoDiffusion SD1.5 adaptation requires DPM-Solver++, which accepts custom timesteps")
        if not isinstance(timesteps, list) or not timesteps:
            errors.append("AutoDiffusion requires inference.timesteps")
        elif not all(isinstance(t, int) and 0 <= t <= 999 for t in timesteps):
            errors.append("AutoDiffusion timesteps must be integers in [0, 999]")
        elif any(a <= b for a, b in zip(timesteps, timesteps[1:])):
            errors.append("AutoDiffusion timesteps must be strictly descending")
        elif steps != len(timesteps):
            errors.append("num_inference_steps must equal len(timesteps)")

    if method == "deepcache":
        cache = config.get("deepcache", {})
        if not isinstance(cache.get("interval"), int) or cache.get("interval", 0) < 1:
            errors.append("deepcache.interval must be a positive integer")
        if not isinstance(cache.get("branch_id"), int) or not 0 <= cache.get("branch_id", -1) <= 11:
            errors.append("deepcache.branch_id must be an integer in [0, 11]")

    runtime = config.get("runtime", {})
    if runtime.get("device") != "cuda":
        errors.append("the benchmark timing protocol currently requires runtime.device='cuda'")
    if runtime.get("precision") not in {"float16", "float32"}:
        errors.append("runtime.precision must be float16 or float32")
    if not isinstance(runtime.get("warmup_runs"), int) or runtime.get("warmup_runs", -1) < 0:
        errors.append("runtime.warmup_runs must be a non-negative integer")
    if not isinstance(runtime.get("repetitions"), int) or runtime.get("repetitions", 0) < 1:
        errors.append("runtime.repetitions must be a positive integer")

    if errors:
        raise ConfigError("; ".join(errors))
