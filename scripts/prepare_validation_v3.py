#!/usr/bin/env python3
"""Build the complete validation run list from the locked v3 finalists.

Committed before the finalists were locked, so the comparison set is fixed in
advance. Runs (validation split, 128 prompts x seeds 303/404, 10 warm-ups,
FLOP counting):

- the locked finalists;
- architecture-ablation twins: each finalist's execution policy with no unit
  removal in fp16, and for int8 finalists with removals also no removal in int8.
  These isolate what the architecture genes add on top of sampler/cache tuning;
- baselines: DDIM 6/8/10/15/25/50 steps, DPM-Solver++ 4/6/8/10/15/20 steps,
  DeepCache (20-step DDIM, interval 3, branch 0), int8 weights at 20-step DDIM,
  two Flexiffusion schedules chosen from the calibration controls by a stated
  rule (fastest; highest calibration CLIP), and BK-SDM base/small/tiny
  (external, distilled from SD-v1.4) at 20-step DDIM;
- the 20-step DDIM reference three times (start, middle, end) to measure drift.

Non-reference runs are shuffled with a fixed seed so slow drift is not
confounded with policy type. Duplicate execution signatures are dropped.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diffusionnas.nas_space import signature  # noqa: E402
from diffusionnas.schema import SD15_MODEL, validate_config  # noqa: E402
from diffusionnas.search_space import policy_to_config  # noqa: E402
from prepare_experiment import load_prompts  # noqa: E402

ORDER_SEED = 20261112
SEEDS = [303, 404]
# Calibration controls (2026-10-10): fastest uniform Flexiffusion schedule and highest-CLIP one.
FLEX_FASTEST = (7, 1, 2)
FLEX_BEST_CLIP = (11, 6, 2)


def plain(method="baseline", scheduler="ddim", steps=20, guidance=7.5, precision="fp16", **extra):
    policy = {"method": method, "scheduler": scheduler, "num_inference_steps": steps, "guidance_scale": guidance,
              "architecture": {"removed_units": [], "weight_precision": precision}}
    policy.update(extra)
    return policy


def flex(n_segment: int, branch: int, interval: int) -> dict:
    return plain("flexiffusion", "ddim", 3 * n_segment,
                 flexiffusion={"segment_length": 3, "segments": [{"branch": branch, "interval": interval}] * n_segment})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalists", type=Path, default=ROOT / "artifacts" / "nas" / "finalists.json")
    parser.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompts" / "validation.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs" / "generated" / "validation_v3")
    args = parser.parse_args()

    entries: list[tuple[str, str, dict, str | None]] = []  # (label, role, policy, model id override)
    for finalist in json.loads(args.finalists.read_text(encoding="utf-8"))["finalists"]:
        policy = finalist["policy"]
        entries.append((finalist["name"], "finalist", policy, None))
        architecture = policy["architecture"]
        if architecture["removed_units"] or architecture["weight_precision"] == "int8":
            twin = deepcopy(policy)
            twin["architecture"] = {"removed_units": [], "weight_precision": "fp16"}
            entries.append((f"{finalist['name']}_twin_fp16", "twin", twin, None))
        if architecture["removed_units"] and architecture["weight_precision"] == "int8":
            twin = deepcopy(policy)
            twin["architecture"] = {"removed_units": [], "weight_precision": "int8"}
            entries.append((f"{finalist['name']}_twin_int8", "twin", twin, None))
    for steps in (6, 8, 10, 15, 25, 50):
        entries.append((f"ddim{steps}", "baseline", plain(steps=steps), None))
    for steps in (4, 6, 8, 10, 15, 20):
        entries.append((f"dpm{steps}", "baseline", plain(scheduler="dpm_solver++", steps=steps), None))
    entries.append(("deepcache_i3b0", "baseline", plain("deepcache", deepcache_interval=3, deepcache_branch_id=0), None))
    entries.append(("int8_ddim20", "baseline", plain(precision="int8"), None))
    entries.append(("flex_n7_b1_i2", "baseline", flex(*FLEX_FASTEST), None))
    entries.append(("flex_n11_b6_i2", "baseline", flex(*FLEX_BEST_CLIP), None))
    for size in ("base", "small", "tiny"):
        entries.append((f"bksdm_{size}", "external", plain(), f"nota-ai/bk-sdm-{size}"))

    unique, seen = [], {signature(plain())}
    for label, role, policy, model in entries:
        key = signature(policy) + (model or "")
        if key in seen:
            print(f"drop duplicate {label}")
            continue
        seen.add(key)
        unique.append((label, role, policy, model))
    random.Random(ORDER_SEED).shuffle(unique)
    middle = len(unique) // 2
    ordered = ([("ref_a", "reference", plain(), None)] + unique[:middle] + [("ref_b", "reference", plain(), None)]
               + unique[middle:] + [("ref_c", "reference", plain(), None)])

    prompts = load_prompts(args.prompts)
    runtime = {"device": "cuda", "precision": "float16", "warmup_runs": 10, "repetitions": 1, "count_flops": True}
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    order = []
    for index, (label, role, policy, model) in enumerate(ordered):
        config = policy_to_config(policy, prompts, SEEDS, dict(runtime), "artifacts/validation_v3/runs", save_images=True)
        if policy["architecture"] == {"removed_units": [], "weight_precision": "fp16"}:
            config.pop("architecture")
        if model:
            config["model"]["id"] = model
            config.pop("architecture", None)
        config.update({"label": label, "role": role})
        validate_config(config)
        name = f"{index:02d}_{label}.json"
        (args.output_dir / name).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        order.append({"file": name, "label": label, "role": role, "model": model or SD15_MODEL})
    (args.output_dir / "order.json").write_text(json.dumps(order, indent=2) + "\n", encoding="utf-8")
    roles = {r: sum(o["role"] == r for o in order) for r in ("reference", "finalist", "twin", "baseline", "external")}
    print(f"{len(order)} validation runs -> {args.output_dir}: {roles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
