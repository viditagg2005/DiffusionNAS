"""DiffNAS-Bench: a real-hardware many-objective benchmark for training-free Stable Diffusion compression.

A benchmark directory (written by ``scripts/export_benchmark.py``) holds:
- ``data.jsonl``: every real evaluated configuration (policy + measured objectives + provenance),
- ``metadata.json``: objectives, directions, reference point, scaling, GPU, surrogate accuracy,
- ``surrogates.pkl``: per-objective regressors fitted on all records (surrogate mode).

Usage::

    from diffusionnas.bench import DiffNASBench
    bench = DiffNASBench.load("artifacts/benchmark")
    policy = bench.sample(random.Random(0))
    f = bench.evaluate([policy], objective_set="M6")   # minimization form, scaled: 0 = best seen, 1 = reference
    exact = bench.lookup(policy)                      # measured objectives if this configuration was evaluated

Objectives (raw units): CLIP prompt alignment (max), CLIP image fidelity to the 20-step DDIM
reference (max), p50 latency in ms normalized to a ~495 ms reference (min), GFLOPs per image (min),
peak allocated VRAM in MB (min), U-Net weight MB (min). Measured on one NVIDIA RTX 6000 Ada,
SD1.5 at 512x512, batch 1, 48 COCO calibration prompts x 2 seeds per configuration.
"""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import random
from typing import Any

import numpy as np

from . import nas_space

OBJECTIVE_SETS = {
    "M4": ["clip_score", "clip_image_fidelity", "latency_p50_ms", "unet_weight_mb"],
    "M6": ["clip_score", "clip_image_fidelity", "latency_p50_ms", "gflops_per_image", "peak_allocated_mb", "unet_weight_mb"],
}
DIRECTIONS = {"clip_score": "max", "clip_image_fidelity": "max", "latency_p50_ms": "min",
              "gflops_per_image": "min", "peak_allocated_mb": "min", "unet_weight_mb": "min"}


class DiffNASBench:
    def __init__(self, records: list[dict[str, Any]], metadata: dict[str, Any], surrogates: dict[str, Any] | None) -> None:
        self.records = records
        self.metadata = metadata
        self.surrogates = surrogates or {}
        self._table = {nas_space.signature(r["policy"]): r for r in records}

    @classmethod
    def load(cls, directory: str | Path) -> "DiffNASBench":
        directory = Path(directory)
        records = [json.loads(line) for line in (directory / "data.jsonl").read_text().splitlines() if line.strip()]
        metadata = json.loads((directory / "metadata.json").read_text())
        surrogates = pickle.loads((directory / "surrogates.pkl").read_bytes()) if (directory / "surrogates.pkl").exists() else None
        return cls(records, metadata, surrogates)

    @staticmethod
    def sample(rng: random.Random) -> dict[str, Any]:
        """A uniformly drawn configuration from the v3 search space."""
        return nas_space.decode(nas_space.random_genome(rng))

    def lookup(self, policy: dict[str, Any]) -> dict[str, float] | None:
        record = self._table.get(nas_space.signature(policy))
        return dict(record["objectives"]) if record else None

    def scale(self, raw: np.ndarray, objective_set: str) -> np.ndarray:
        names = OBJECTIVE_SETS[objective_set]
        sign = np.array([-1.0 if DIRECTIONS[n] == "max" else 1.0 for n in names])
        low = sign * np.array([self.metadata["scaling_low"][n] for n in names])
        high = sign * np.array([self.metadata["reference_point"][n] for n in names])
        return (sign * raw - low) / (high - low)

    def evaluate(self, policies: list[dict[str, Any]], objective_set: str = "M6", scaled: bool = True) -> np.ndarray:
        """Surrogate predictions for any configuration (raw units, or minimization-scaled)."""
        names = OBJECTIVE_SETS[objective_set]
        model = self.surrogates[objective_set]
        raw_min = model.predict(policies)  # minimization form, raw units
        sign = np.array([-1.0 if DIRECTIONS[n] == "max" else 1.0 for n in names])
        raw = raw_min * sign
        return self.scale(raw, objective_set) if scaled else raw
