#!/usr/bin/env python3
"""Configs for the single-unit sensitivity probe (calibration split only).

One 20-step DDIM reference, the same model with int8 weights, each of the 24
removable U-Net units removed alone, and six resolution/stage groups. Results
size the v3 architecture search space; they never touch validation data.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diffusionnas.architecture import UNIT_NAMES  # noqa: E402
from diffusionnas.schema import validate_config  # noqa: E402
from prepare_experiment import load_prompts, parse_seeds, read_json, write_json  # noqa: E402

GROUPS = {
    "attn_64px": ["down0.attn0", "down0.attn1", "up3.attn0", "up3.attn1", "up3.attn2"],
    "attn_32px": ["down1.attn0", "down1.attn1", "up2.attn0", "up2.attn1", "up2.attn2"],
    "attn_16px": ["down2.attn0", "down2.attn1", "up1.attn0", "up1.attn1", "up1.attn2"],
    "mid_all": ["mid.attn0", "mid.res0", "mid.res1"],
    "down3_res": ["down3.res0", "down3.res1"],
    "down_res_all": ["down0.res0", "down0.res1", "down1.res1", "down2.res1", "down3.res0", "down3.res1"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompts" / "calibration.json")
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs" / "generated" / "probe")
    args = parser.parse_args()

    base = read_json(ROOT / "configs" / "baseline_sd15.json")
    base.update({"prompts": load_prompts(args.prompts), "seeds": parse_seeds(args.seeds),
                 "output_root": "artifacts/probe/runs"})
    base["runtime"].update({"warmup_runs": 3, "repetitions": 1, "count_flops": True})

    variants: dict[str, dict] = {"00_reference": {}, "01_int8": {"weight_precision": "int8"}}
    for name in UNIT_NAMES:
        variants[f"10_rm_{name}"] = {"removed_units": [name]}
    for name, units in GROUPS.items():
        variants[f"20_group_{name}"] = {"removed_units": units}

    for label, architecture in variants.items():
        config = deepcopy(base)
        config["label"] = label
        if architecture:
            config["architecture"] = {"removed_units": [], "weight_precision": "fp16", **architecture}
        validate_config(config)
        write_json(args.output_dir / f"{label}.json", config)
    print(f"Wrote {len(variants)} probe configs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
