#!/usr/bin/env python3
"""Lock each arm's finalists with the pre-registered rule and write the C3 run list (configs/nas/c3_protocol.json)."""

from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diffusionnas.nas_space import signature  # noqa: E402
from diffusionnas.schema import validate_config  # noqa: E402
from diffusionnas.search_space import policy_to_config  # noqa: E402
from prepare_experiment import load_prompts  # noqa: E402

OUT = ROOT / "configs" / "generated" / "c3"
ROOTS = {"sa_lhfid": ["artifacts/nas_c2"], "sa_nsga2": ["artifacts/nas_c2"],
         "random": ["artifacts/nas", "artifacts/nas_c2"], "nsga2": ["artifacts/nas", "artifacts/nas_c2"]}


def main() -> int:
    protocol = json.loads((ROOT / "configs/nas/c3_protocol.json").read_text())
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; C3 finalists are locked once")
    OUT.mkdir(parents=True)
    locked = {}
    for arm, roots in ROOTS.items():
        command = [sys.executable, str(ROOT / "scripts/lock_nas_finalists.py"), "--arms", arm, "--max-finalists", "5",
                   "--output", str(OUT / f"locked_{arm}.json")]
        for root in roots:
            command += ["--nas-root", str(ROOT / root)]
        subprocess.run(command, check=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
        locked[arm] = json.loads((OUT / f"locked_{arm}.json").read_text())["finalists"]

    prompts = load_prompts(ROOT / "data/prompts/test2.json")
    runtime = {"device": "cuda", "precision": "float16", "warmup_runs": 10, "repetitions": 1, "count_flops": True}
    by_signature: dict[str, str] = {}
    configs: dict[str, dict] = {}
    assignments: dict[str, list[str]] = {}
    for arm, finalists in locked.items():
        for finalist in finalists:
            key = signature(finalist["policy"])
            if key not in by_signature:
                label = f"c3_{len(by_signature):02d}"
                by_signature[key] = label
                config = policy_to_config(finalist["policy"], prompts, [protocol["image_seed"]], dict(runtime),
                                          "artifacts/c3/runs", save_images=True)
                if config.get("architecture") == {"removed_units": [], "weight_precision": "fp16"}:
                    config.pop("architecture")
                config.update({"label": label, "role": "c3"})
                validate_config(config)
                configs[label] = config
            assignments.setdefault(arm, []).append(by_signature[key])
    reference = json.loads((ROOT / "configs/baseline_sd15.json").read_text())
    reference.update({"prompts": prompts, "seeds": [protocol["image_seed"]], "output_root": "artifacts/c3/runs"})
    reference["runtime"].update(runtime)
    order = list(configs)
    random.Random(20261116).shuffle(order)
    middle = len(order) // 2
    sequence = ["ref_a"] + order[:middle] + ["ref_b"] + order[middle:] + ["ref_c"]
    for index, label in enumerate(sequence):
        config = dict(reference, label=label, role="reference") if label.startswith("ref_") else configs[label]
        validate_config(config)
        (OUT / f"{index:02d}_{label}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (OUT / "assignments.json").write_text(json.dumps({"arms": assignments, "order": sequence}, indent=2) + "\n")
    shutil.copy(ROOT / "configs/nas/c3_protocol.json", OUT / "protocol_at_lock.json")
    print(f"{len(configs)} unique locked configurations + 3 references -> {OUT}")
    for arm, members in assignments.items():
        print(f"  {arm:9s} {members}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
