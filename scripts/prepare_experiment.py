#!/usr/bin/env python3
"""Materialize matched calibration and search configs from a prompt split."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_prompts(path: Path) -> list[str]:
    value = read_json(path)
    if isinstance(value, dict):
        value = value.get("prompts")
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must contain a non-empty JSON list or a prompts object")
    prompts = [item.get("prompt") if isinstance(item, dict) else item for item in value]
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError(f"{path} contains an invalid prompt")
    return prompts


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be a non-empty comma-separated list of non-negative integers")
    return seeds


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-prompts", type=Path, required=True)
    parser.add_argument("--calibration-seeds", default="42,123")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs" / "generated")
    parser.add_argument("--search-budget", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=20)
    args = parser.parse_args()

    if args.search_budget < 1:
        parser.error("--search-budget must be positive")
    if args.population_size < 2 or args.population_size > args.search_budget:
        parser.error("--population-size must be in [2, search-budget]")

    prompts = load_prompts(args.calibration_prompts)
    seeds = parse_seeds(args.calibration_seeds)
    output_dir = args.output_dir.resolve()

    for name in ("baseline", "autodiffusion", "deepcache"):
        config = read_json(ROOT / "configs" / f"{name}_sd15.json")
        config["prompts"] = prompts
        config["seeds"] = seeds
        config["runtime"]["warmup_runs"] = 10
        config["runtime"]["repetitions"] = 1
        config["output_root"] = "artifacts/calibration/runs"
        write_json(output_dir / "calibration" / f"{name}_sd15.json", config)

    for name in ("random_search", "ecad"):
        config = read_json(ROOT / "configs" / f"{name}_sd15.json")
        config["prompts"] = prompts
        config["seeds"] = seeds
        config["budget"] = args.search_budget
        config["runtime"]["warmup_runs"] = 1
        config["runtime"]["repetitions"] = 1
        if name == "ecad":
            config["population_size"] = args.population_size
        write_json(output_dir / "search" / f"{name}_sd15.json", config)

    print(f"Wrote experiment configs to {output_dir}")
    print(f"Calibration samples per policy: {len(prompts) * len(seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

