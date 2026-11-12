#!/usr/bin/env python3
"""Write the C1 confirmatory run configs (protocol: configs/nas/c1_protocol.json)."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diffusionnas.schema import validate_config  # noqa: E402
from prepare_experiment import load_prompts  # noqa: E402

OUT = ROOT / "configs" / "generated" / "c1"


def main() -> int:
    protocol = json.loads((ROOT / "configs/nas/c1_protocol.json").read_text())
    validation = ROOT / "configs/generated/validation_v3"
    by_label = {o["label"]: json.loads((validation / o["file"]).read_text())
                for o in json.loads((validation / "order.json").read_text())}
    posthoc = {json.loads(p.read_text())["label"]: json.loads(p.read_text())
               for p in (ROOT / "configs/generated/posthoc_bksdm").glob("*.json")}
    sources = {
        "reference": by_label["ref_a"], "nas_04": by_label["nas_04"], "nas_04_twin_int8": by_label["nas_04_twin_int8"],
        "nas_01": by_label["nas_01"], "int8_ddim20": by_label["int8_ddim20"], "dpm8": by_label["dpm8"],
        "deepcache_i3b0": by_label["deepcache_i3b0"],
        "bksdm_small_ddim10_int8": posthoc["posthoc_bksdm_small_ddim10_int8"],
        "bksdm_base_ddim10_int8": posthoc["posthoc_bksdm_base_ddim10_int8"],
    }
    assert set(sources) == set(protocol["policies"])
    prompts = load_prompts(ROOT / "data/prompts/test.json")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for index, (label, source) in enumerate(sources.items()):
        config = json.loads(json.dumps(source))
        config.update({"prompts": prompts, "seeds": [protocol["image_seed"]], "label": label, "role": "c1",
                       "output_root": "artifacts/c1/runs"})
        config["runtime"].update({"warmup_runs": 10, "repetitions": 1, "count_flops": True})
        validate_config(config)
        (OUT / f"{index:02d}_{label}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"{len(sources)} C1 configs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
