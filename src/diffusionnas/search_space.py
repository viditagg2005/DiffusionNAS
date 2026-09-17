"""Shared policy search space for benchmark optimisers.

Defines the chromosome encoding used by both Random Search and ECAD.
A chromosome is a list of 16 binary integers whose fields map onto
the inference-policy JSON schema used by the rest of the harness.

Chromosome layout
-----------------
Bit   Field                 Width  Decoded range
0-1   method                2      0-2 → METHOD_CHOICES (3 wraps to 0)
2     scheduler             1      0-1 → SCHEDULER_CHOICES
3-4   nfe_index             2      0-3 → NFE_CHOICES
5-6   guidance_index        2      0-3 → GUIDANCE_CHOICES
7-9   deepcache_interval    3      raw 0-7, decoded+1 then clamped [1, 5]
10-13 deepcache_branch_id   4      raw 0-15, clamped [0, 11]
14-15 timestep_pattern      2      0-2 → TIMESTEP_PATTERNS (3 wraps to 0)
"""
from __future__ import annotations

import json
import math
import random as _random
from pathlib import Path
from typing import Any

from .schema import SD15_MODEL, validate_config


# ---- Search-space dimensions -------------------------------------------

NFE_CHOICES: tuple[int, ...] = (4, 8, 12, 20)
GUIDANCE_CHOICES: tuple[float, ...] = (5.0, 7.5, 10.0, 12.0)
SCHEDULER_CHOICES: tuple[str, ...] = ("ddim", "dpm_solver++")
METHOD_CHOICES: tuple[str, ...] = ("baseline", "deepcache", "autodiffusion")
TIMESTEP_PATTERNS: tuple[str, ...] = ("linear", "trailing", "leading")

DEEPCACHE_INTERVAL_MIN = 1
DEEPCACHE_INTERVAL_MAX = 5
DEEPCACHE_BRANCH_MIN = 0
DEEPCACHE_BRANCH_MAX = 11

CHROMOSOME_LENGTH = 16


# ---- Bit manipulation helpers ------------------------------------------

def _bits_to_int(bits: list[int], start: int, width: int) -> int:
    """Decode *width* bits starting at *start* as a little-endian integer."""
    value = 0
    for i in range(width):
        value |= bits[start + i] << i
    return value


def _int_to_bits(value: int, width: int) -> list[int]:
    """Encode *value* into *width* little-endian bits."""
    return [(value >> i) & 1 for i in range(width)]


# ---- Timestep generation -----------------------------------------------

def generate_timesteps(nfe: int, pattern: str) -> list[int]:
    """Generate *nfe* strictly-descending timesteps in [0, 999].

    Parameters
    ----------
    nfe : int
        Number of function evaluations (inference steps).
    pattern : str
        One of ``"linear"``, ``"trailing"`` (quadratic bias toward
        lower timesteps), or ``"leading"`` (sqrt bias toward higher
        timesteps).
    """
    if nfe < 1:
        raise ValueError("nfe must be >= 1")
    if nfe == 1:
        return [999]
    indices = list(range(nfe))
    raw: list[float]
    if pattern == "trailing":
        raw = [999.0 * ((nfe - 1 - i) / (nfe - 1)) ** 2 for i in indices]
    elif pattern == "leading":
        raw = [999.0 * math.sqrt((nfe - 1 - i) / (nfe - 1)) for i in indices]
    else:  # "linear" or fallback
        raw = [999.0 * (nfe - 1 - i) / (nfe - 1) for i in indices]
    # Round and enforce strict descent
    timesteps: list[int] = []
    prev = 1000
    for v in raw:
        t = max(0, min(999, round(v)))
        t = min(t, prev - 1)
        timesteps.append(t)
        prev = t
    return timesteps


# ---- Encode / decode ---------------------------------------------------

def encode_policy(policy: dict[str, Any]) -> list[int]:
    """Encode a policy dict into a binary chromosome."""
    bits = [0] * CHROMOSOME_LENGTH

    method = policy.get("method", "baseline")
    method_idx = METHOD_CHOICES.index(method) if method in METHOD_CHOICES else 0
    for i, b in enumerate(_int_to_bits(method_idx, 2)):
        bits[0 + i] = b

    scheduler = policy.get("scheduler", "ddim")
    sched_idx = SCHEDULER_CHOICES.index(scheduler) if scheduler in SCHEDULER_CHOICES else 0
    bits[2] = sched_idx

    nfe = policy.get("num_inference_steps", 20)
    nfe_idx = NFE_CHOICES.index(nfe) if nfe in NFE_CHOICES else len(NFE_CHOICES) - 1
    for i, b in enumerate(_int_to_bits(nfe_idx, 2)):
        bits[3 + i] = b

    guidance = policy.get("guidance_scale", 7.5)
    guidance_idx = min(range(len(GUIDANCE_CHOICES)), key=lambda j: abs(GUIDANCE_CHOICES[j] - guidance))
    for i, b in enumerate(_int_to_bits(guidance_idx, 2)):
        bits[5 + i] = b

    interval = policy.get("deepcache_interval", 3)
    interval_enc = max(0, min(7, interval - 1))
    for i, b in enumerate(_int_to_bits(interval_enc, 3)):
        bits[7 + i] = b

    branch = policy.get("deepcache_branch_id", 0)
    branch_enc = max(0, min(15, branch))
    for i, b in enumerate(_int_to_bits(branch_enc, 4)):
        bits[10 + i] = b

    pattern = policy.get("timestep_pattern", "linear")
    pattern_idx = TIMESTEP_PATTERNS.index(pattern) if pattern in TIMESTEP_PATTERNS else 0
    for i, b in enumerate(_int_to_bits(pattern_idx, 2)):
        bits[14 + i] = b

    return bits


def decode_policy(bits: list[int]) -> dict[str, Any]:
    """Decode a binary chromosome into a policy dict."""
    if len(bits) != CHROMOSOME_LENGTH:
        raise ValueError(f"expected {CHROMOSOME_LENGTH} bits, got {len(bits)}")

    method_idx = _bits_to_int(bits, 0, 2) % len(METHOD_CHOICES)
    method = METHOD_CHOICES[method_idx]

    sched_idx = bits[2]
    scheduler = SCHEDULER_CHOICES[sched_idx]

    nfe_idx = _bits_to_int(bits, 3, 2) % len(NFE_CHOICES)
    nfe = NFE_CHOICES[nfe_idx]

    guidance_idx = _bits_to_int(bits, 5, 2) % len(GUIDANCE_CHOICES)
    guidance = GUIDANCE_CHOICES[guidance_idx]

    interval_raw = _bits_to_int(bits, 7, 3)
    interval = max(DEEPCACHE_INTERVAL_MIN, min(DEEPCACHE_INTERVAL_MAX, interval_raw + 1))

    branch_raw = _bits_to_int(bits, 10, 4)
    branch_id = max(DEEPCACHE_BRANCH_MIN, min(DEEPCACHE_BRANCH_MAX, branch_raw))

    pattern_idx = _bits_to_int(bits, 14, 2) % len(TIMESTEP_PATTERNS)
    pattern = TIMESTEP_PATTERNS[pattern_idx]

    # Apply method constraints
    if method == "autodiffusion":
        scheduler = "dpm_solver++"

    policy: dict[str, Any] = {
        "method": method,
        "scheduler": scheduler,
        "num_inference_steps": nfe,
        "guidance_scale": guidance,
        "timestep_pattern": pattern,
    }
    if method == "deepcache":
        policy["deepcache_interval"] = interval
        policy["deepcache_branch_id"] = branch_id
    if method == "autodiffusion":
        policy["timesteps"] = generate_timesteps(nfe, pattern)
    return policy


# ---- Random sampling ---------------------------------------------------

def random_chromosome(rng: _random.Random) -> list[int]:
    """Generate a uniformly random chromosome."""
    return [rng.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]


def sample_random_policy(rng: _random.Random) -> dict[str, Any]:
    """Sample a policy uniformly from the search space."""
    return decode_policy(random_chromosome(rng))


# ---- Config construction -----------------------------------------------

def policy_to_config(
    policy: dict[str, Any],
    prompts: list[str],
    seeds: list[int],
    runtime: dict[str, Any] | None = None,
    output_root: str = "artifacts/runs",
    save_images: bool = True,
) -> dict[str, Any]:
    """Convert a decoded policy dict into a validated DiffusionNAS config."""
    if runtime is None:
        runtime = {
            "device": "cuda",
            "precision": "float16",
            "warmup_runs": 1,
            "repetitions": 3,
        }
    config: dict[str, Any] = {
        "method": policy["method"],
        "model": {"id": SD15_MODEL, "height": 512, "width": 512},
        "prompts": prompts,
        "seeds": seeds,
        "inference": {
            "scheduler": policy["scheduler"],
            "num_inference_steps": policy["num_inference_steps"],
            "guidance_scale": policy["guidance_scale"],
        },
        "runtime": runtime,
        "save_images": save_images,
        "output_root": output_root,
    }
    if policy["method"] == "autodiffusion":
        config["inference"]["timesteps"] = policy["timesteps"]
    if policy["method"] == "deepcache":
        config["deepcache"] = {
            "interval": policy["deepcache_interval"],
            "branch_id": policy["deepcache_branch_id"],
            "skip_mode": "uniform",
        }
    return config


# ---- Objective extraction -----------------------------------------------

def read_objectives(run_dir: str | Path) -> dict[str, float] | None:
    """Read objective values from a completed, scored run directory.

    Returns ``None`` when the required fields are missing (e.g. when the
    run was a dry-run or scoring has not been performed yet).
    """
    try:
        summary = json.loads(Path(run_dir).joinpath("summary.json").read_text(encoding="utf-8"))
        return {
            "clip_score": float(summary["quality_metrics"]["clip_score"]["mean"]),
            "latency_p50_ms": float(summary["latency_ms"]["p50"]),
            "peak_allocated_mb": float(summary["peak_allocated_mb"]),
        }
    except (KeyError, TypeError, ValueError, OSError):
        return None


# ---- Search config validation ------------------------------------------

def validate_search_config(config: dict[str, Any]) -> None:
    """Validate a search-algorithm configuration dict.

    Raises :class:`ValueError` with an aggregated error message on any
    violation.
    """
    errors: list[str] = []
    method = config.get("search_method")
    if method not in {"random", "ecad"}:
        errors.append("search_method must be 'random' or 'ecad'")

    budget = config.get("budget")
    if not isinstance(budget, int) or budget < 1:
        errors.append("budget must be a positive integer")

    seed = config.get("seed")
    if not isinstance(seed, int) or seed < 0:
        errors.append("seed must be a non-negative integer")

    prompts = config.get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        errors.append("prompts must be a non-empty list of non-empty strings")

    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) and s >= 0 for s in seeds):
        errors.append("seeds must be a non-empty list of non-negative integers")

    runtime = config.get("runtime", {})
    if runtime.get("device") != "cuda":
        errors.append("runtime.device must be 'cuda'")
    if runtime.get("precision") not in {"float16", "float32"}:
        errors.append("runtime.precision must be 'float16' or 'float32'")

    if method == "ecad":
        pop = config.get("population_size")
        if not isinstance(pop, int) or pop < 2:
            errors.append("population_size must be an integer >= 2")
        cr = config.get("crossover_rate")
        if not isinstance(cr, (int, float)) or not 0.0 <= cr <= 1.0:
            errors.append("crossover_rate must be a float in [0, 1]")
        mr = config.get("mutation_rate")
        if not isinstance(mr, (int, float)) or not 0.0 <= mr <= 1.0:
            errors.append("mutation_rate must be a float in [0, 1]")

    if errors:
        raise ValueError("; ".join(errors))

