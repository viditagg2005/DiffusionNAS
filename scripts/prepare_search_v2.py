#!/usr/bin/env python3
"""Materialize repeated v2 searches: random and NSGA-II, several search seeds each."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diffusionnas.search_space import validate_search_config  # noqa: E402
from prepare_experiment import load_prompts, parse_seeds, read_json, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--seeds", default="42,123", help="image seeds per prompt")
    parser.add_argument("--search-seeds", default="1,2,3", help="optimizer RNG seeds; one search per seed")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs" / "generated" / "search_v2")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    seeds = parse_seeds(args.seeds)
    templates = {
        "random": read_json(ROOT / "configs" / "random_search_v2_sd15.json"),
        "nsga2": read_json(ROOT / "configs" / "nsga2_v2_sd15.json"),
    }
    written = 0
    # Interleave optimizers per seed so slow latency drift within one allocation affects both equally.
    for order, search_seed in enumerate(parse_seeds(args.search_seeds)):
        for name, template in templates.items():
            config = deepcopy(template)
            config.update({"prompts": prompts, "seeds": seeds, "seed": search_seed, "budget": args.budget})
            config["runtime"].update({"warmup_runs": 1, "repetitions": 1})
            config["output_root"] = f"artifacts/search_v2/{name}_s{search_seed}"
            validate_search_config(config)
            write_json(args.output_dir / f"{order:02d}_{name}_s{search_seed}.json", config)
            written += 1
    print(f"Wrote {written} search configs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
