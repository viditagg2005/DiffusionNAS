#!/usr/bin/env python3
"""Write the v3 search configs: random and NSGA-II for each optimizer seed, on the calibration split."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diffusionnas.nas_search import validate_nas_config  # noqa: E402
from prepare_experiment import load_prompts, parse_seeds, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompts" / "calibration.json")
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--search-seeds", default="1,2,3")
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs" / "generated" / "nas")
    args = parser.parse_args()

    prompts, seeds = load_prompts(args.prompts), parse_seeds(args.seeds)
    for search_seed in parse_seeds(args.search_seeds):
        for method in ("random", "nsga2"):
            config = deepcopy(json.loads((ROOT / "configs" / "nas" / f"v3_{method}.json").read_text(encoding="utf-8")))
            config.update({"prompts": prompts, "seeds": seeds, "seed": search_seed, "budget": args.budget})
            validate_nas_config(config)
            write_json(args.output_dir / f"seed{search_seed}_{method}.json", config)
    # The fidelity/latency anchor each job measures first, in its own allocation.
    reference = json.loads((ROOT / "configs" / "baseline_sd15.json").read_text(encoding="utf-8"))
    reference.update({"prompts": prompts, "seeds": seeds, "label": "reference"})
    reference["runtime"].update({"warmup_runs": 10, "repetitions": 1, "count_flops": True})
    write_json(args.output_dir / "reference.json", reference)
    print(f"wrote configs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
