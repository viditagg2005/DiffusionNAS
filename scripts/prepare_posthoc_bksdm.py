#!/usr/bin/env python3
"""Post-hoc (after validation was inspected) BK-SDM controls on the validation split.

The locked validation ran BK-SDM at 20-step DDIM, while the best NAS finalist
(nas_04) uses 10-step DDIM, guidance 10.5 and int8 weights; its apparent
dominance mixes step count and int8 (both orthogonal to BK-SDM's architecture).
These runs put BK-SDM on the same sampler and precision. They are labelled
post-hoc and never feed back into any selection. One allocation, with a 20-step
reference and a re-run of nas_04 for direct pairing.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas.schema import validate_config  # noqa: E402

SOURCE = ROOT / "configs" / "generated" / "validation_v3"
OUT = ROOT / "configs" / "generated" / "posthoc_bksdm"


def main() -> int:
    order = json.loads((SOURCE / "order.json").read_text())
    by_label = {o["label"]: json.loads((SOURCE / o["file"]).read_text()) for o in order}
    nas04 = by_label["nas_04"]
    configs = {"00_posthoc_ref": dict(by_label["ref_a"], label="posthoc_ref"),
               "01_posthoc_nas_04": dict(nas04, label="posthoc_nas_04")}
    for size in ("tiny", "small", "base"):
        for precision in ("fp16", "int8"):
            config = json.loads(json.dumps(by_label[f"bksdm_{size}"]))
            config["inference"].update({"num_inference_steps": 10, "guidance_scale": nas04["inference"]["guidance_scale"]})
            if precision == "int8":
                config["architecture"] = {"removed_units": [], "weight_precision": "int8"}
            config["label"] = f"posthoc_bksdm_{size}_ddim10_{precision}"
            configs[f"1{len(configs)}_{config['label']}"] = config
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name, config in configs.items():
        config = dict(config, role="posthoc", output_root="artifacts/posthoc_bksdm/runs")
        validate_config(config)
        (OUT / f"{name}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"{len(configs)} post-hoc configs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
