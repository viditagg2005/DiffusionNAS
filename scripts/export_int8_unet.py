#!/usr/bin/env python3
"""Export the SD1.5 U-Net as safetensors in fp16 and with int8 weights; report on-disk sizes.

Backs the compression claim with a measured checkpoint size (not only resident bytes).
Output: artifacts/export/ (git-ignored) and artifacts/analysis/export_sizes.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import torch
    from diffusers import UNet2DConditionModel
    from safetensors.torch import save_file

    from diffusionnas.architecture import quantize_weights_int8, remove_units, unet_size

    out = ROOT / "artifacts" / "export"
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    variants = {"fp16": ([], False), "int8": ([], True), "int8_minus_up2.attn2": (["up2.attn2"], True)}
    for name, (removed, int8) in variants.items():
        unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet",
                                                    torch_dtype=torch.float16)
        remove_units(unet, removed)
        if int8:
            quantize_weights_int8(unet)
        state = {k: v.contiguous() for k, v in unet.state_dict().items()}
        path = out / f"unet_{name}.safetensors"
        save_file(state, str(path))
        sizes[name] = {"file_mb": path.stat().st_size / 2**20, "unet_parameters_m": unet_size(unet)["parameters"] / 1e6,
                       "stored_weight_mb": unet_size(unet)["bytes"] / 2**20, "removed_units": removed}
        print(name, {k: round(v, 1) if isinstance(v, float) else v for k, v in sizes[name].items()})
        del unet, state
    (ROOT / "artifacts" / "analysis" / "export_sizes.json").write_text(json.dumps(sizes, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
