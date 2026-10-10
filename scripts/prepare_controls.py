#!/usr/bin/env python3
"""Materialize the matched control benchmark: native DPM-Solver++ and Flexiffusion slice.

All policies run in one allocation (see scripts/slurm/controls_benchmark.sbatch),
so the 20-step DDIM baseline and DeepCache reference are re-measured beside them.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diffusionnas.flexiffusion import (  # noqa: E402
    SD15_BRANCH_CHOICES,
    SD15_INTERVAL_CHOICES,
    SD15_SEGMENT_LENGTH,
    uniform_segments,
)
from diffusionnas.schema import validate_config  # noqa: E402
from prepare_experiment import load_prompts, parse_seeds, read_json, write_json  # noqa: E402

# Smallest, middle and largest N-segment in Flexiffusion's SD1.5 budgets (Table 8).
N_SEGMENTS = (7, 9, 11)


def matched(config: dict, prompts: list[str], seeds: list[int], output_root: str) -> dict:
    config = deepcopy(config)
    config["prompts"] = prompts
    config["seeds"] = seeds
    config["runtime"]["warmup_runs"] = 10
    config["runtime"]["repetitions"] = 1
    config["output_root"] = output_root
    validate_config(config)
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs" / "generated" / "controls")
    parser.add_argument("--output-root", default="artifacts/controls/runs")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    seeds = parse_seeds(args.seeds)
    baseline = read_json(ROOT / "configs" / "baseline_sd15.json")
    configs: dict[str, dict] = {
        "00_baseline_ddim20": baseline,
        "01_deepcache_i3_b0": read_json(ROOT / "configs" / "deepcache_sd15.json"),
        "02_autodiffusion_dpm4": read_json(ROOT / "configs" / "autodiffusion_sd15.json"),
    }
    # Native scheduler control: same solver and step count as the AutoDiffusion point,
    # default Diffusers timestep spacing instead of the explicit list.
    native = deepcopy(baseline)
    native["inference"].update({"scheduler": "dpm_solver++", "num_inference_steps": 4})
    configs["03_native_dpm4"] = native

    flex_template = read_json(ROOT / "configs" / "flexiffusion_sd15.json")
    for n_segment, branch, interval in product(N_SEGMENTS, SD15_BRANCH_CHOICES, SD15_INTERVAL_CHOICES):
        config = deepcopy(flex_template)
        config["inference"]["num_inference_steps"] = n_segment * SD15_SEGMENT_LENGTH
        config["flexiffusion"] = {
            "segment_length": SD15_SEGMENT_LENGTH,
            "segments": uniform_segments(n_segment, branch, interval),
        }
        configs[f"10_flex_n{n_segment:02d}_b{branch}_i{interval}"] = config

    for name, config in configs.items():
        write_json(args.output_dir / f"{name}.json", matched(config, prompts, seeds, args.output_root))
    print(f"Wrote {len(configs)} control configs to {args.output_dir}")
    print(f"Samples per policy: {len(prompts) * len(seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
