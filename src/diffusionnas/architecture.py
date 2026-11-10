"""Training-free static architecture edits for the SD1.5 U-Net.

Two compression operators, both applied to frozen pretrained weights:

1. **Unit removal.** A removable unit is a module whose output can be replaced
   by its input without breaking tensor shapes: every Transformer2D attention
   block (they are residual), and every ResNet block whose input and output
   channel counts match. Removed modules are deleted, so parameters and
   resident weight memory really shrink (unlike per-step skipping or caching,
   which keep every weight). Up-block ResNets consume a concatenated skip
   connection (in != out channels) and are therefore not removable without
   retraining.
2. **Weight-only int8.** Linear and Conv2d weights are stored as int8 with a
   per-output-channel fp16 scale and dequantized on the fly. Resident weight
   memory roughly halves; activations stay fp16.

Unit names follow the module path, e.g. ``down0.attn1`` or ``mid.res0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WEIGHT_PRECISIONS = ("fp16", "int8")


@dataclass(frozen=True)
class Unit:
    name: str
    kind: str  # "attn" or "res"
    path: tuple[str, int, str, int]  # (container, block index, list attribute, layer index)


def _sd15_units() -> tuple[Unit, ...]:
    units: list[Unit] = []
    # Down blocks 0-2 have 2 attentions; channels change only in layer 0 of blocks 1 and 2.
    for block in range(3):
        for layer in range(2):
            units.append(Unit(f"down{block}.attn{layer}", "attn", ("down_blocks", block, "attentions", layer)))
    for block, layer in ((0, 0), (0, 1), (1, 1), (2, 1), (3, 0), (3, 1)):
        units.append(Unit(f"down{block}.res{layer}", "res", ("down_blocks", block, "resnets", layer)))
    units.append(Unit("mid.attn0", "attn", ("mid_block", 0, "attentions", 0)))
    for layer in range(2):
        units.append(Unit(f"mid.res{layer}", "res", ("mid_block", 0, "resnets", layer)))
    # Up blocks 1-3 have 3 attentions each; up ResNets take concatenated skips and stay.
    for block in range(1, 4):
        for layer in range(3):
            units.append(Unit(f"up{block}.attn{layer}", "attn", ("up_blocks", block, "attentions", layer)))
    return tuple(units)


SD15_UNITS = _sd15_units()
UNIT_NAMES = tuple(unit.name for unit in SD15_UNITS)
_BY_NAME = {unit.name: unit for unit in SD15_UNITS}


def validate_architecture(architecture: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    removed = architecture.get("removed_units", [])
    if not isinstance(removed, list) or not all(isinstance(name, str) for name in removed):
        errors.append("architecture.removed_units must be a list of unit names")
    else:
        unknown = sorted(set(removed) - set(UNIT_NAMES))
        if unknown:
            errors.append(f"unknown architecture units: {unknown}")
        if len(set(removed)) != len(removed):
            errors.append("architecture.removed_units contains duplicates")
    if architecture.get("weight_precision", "fp16") not in WEIGHT_PRECISIONS:
        errors.append(f"architecture.weight_precision must be one of {WEIGHT_PRECISIONS}")
    return errors


def _container(unet: Any, unit: Unit) -> Any:
    root, block, attribute, _ = unit.path
    parent = unet.mid_block if root == "mid_block" else getattr(unet, root)[block]
    return getattr(parent, attribute)


def _identity_modules():
    import torch
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    class RemovedResnet(torch.nn.Module):
        def forward(self, input_tensor, temb=None, *args, **kwargs):
            return input_tensor

    class RemovedAttention(torch.nn.Module):
        def forward(self, hidden_states, *args, return_dict: bool = True, **kwargs):
            return Transformer2DModelOutput(sample=hidden_states) if return_dict else (hidden_states,)

    return RemovedResnet, RemovedAttention


def remove_units(unet: Any, names: list[str]) -> None:
    """Replace the named units by identity modules and free their weights."""
    removed_resnet, removed_attention = _identity_modules()
    for name in names:
        unit = _BY_NAME[name]
        modules = _container(unet, unit)
        layer = unit.path[3]
        current = modules[layer]
        if unit.kind == "res" and getattr(current, "in_channels", None) != getattr(current, "out_channels", None):
            raise ValueError(f"{name} changes channels and cannot be removed without retraining")
        modules[layer] = removed_resnet() if unit.kind == "res" else removed_attention()


def quantize_weights_int8(module: Any) -> int:
    """Store every Linear/Conv2d weight as int8 + per-output-channel scale; return layers converted."""
    import torch
    import torch.nn.functional as F

    class Int8Linear(torch.nn.Module):
        def __init__(self, layer: torch.nn.Linear) -> None:
            super().__init__()
            weight = layer.weight.detach().float()
            scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
            self.register_buffer("qweight", torch.round(weight / scale).to(torch.int8))
            self.register_buffer("scale", scale.to(layer.weight.dtype))
            self.bias = layer.bias

        def forward(self, x, *args, **kwargs):
            return F.linear(x, self.qweight.to(x.dtype) * self.scale.to(x.dtype), self.bias)

    class Int8Conv2d(torch.nn.Module):
        def __init__(self, layer: torch.nn.Conv2d) -> None:
            super().__init__()
            weight = layer.weight.detach().float()
            scale = weight.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8) / 127.0
            self.register_buffer("qweight", torch.round(weight / scale).to(torch.int8))
            self.register_buffer("scale", scale.to(layer.weight.dtype))
            self.bias = layer.bias
            self.stride, self.padding, self.dilation, self.groups = layer.stride, layer.padding, layer.dilation, layer.groups

        def forward(self, x, *args, **kwargs):
            weight = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
            return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    converted = 0
    for parent in list(module.modules()):
        for child_name, child in list(parent.named_children()):
            # LoRA-compatible diffusers layers subclass nn.Linear/nn.Conv2d, so isinstance covers them.
            if isinstance(child, torch.nn.Linear):
                setattr(parent, child_name, Int8Linear(child))
                converted += 1
            elif isinstance(child, torch.nn.Conv2d):
                setattr(parent, child_name, Int8Conv2d(child))
                converted += 1
    return converted


def apply_architecture(pipe: Any, architecture: dict[str, Any] | None) -> dict[str, Any]:
    """Edit ``pipe.unet`` in place; return size statistics measured after the edit."""
    import torch

    architecture = architecture or {}
    before = unet_size(pipe.unet)
    remove_units(pipe.unet, list(architecture.get("removed_units", [])))
    converted = 0
    if architecture.get("weight_precision", "fp16") == "int8":
        converted = quantize_weights_int8(pipe.unet)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after = unet_size(pipe.unet)
    return {
        "removed_units": list(architecture.get("removed_units", [])),
        "weight_precision": architecture.get("weight_precision", "fp16"),
        "int8_layers": converted,
        "unet_parameters_original": before["parameters"],
        "unet_parameters": after["parameters"],
        "unet_weight_bytes_original": before["bytes"],
        "unet_weight_bytes": after["bytes"],
    }


def unet_size(unet: Any) -> dict[str, int]:
    """Count stored weights (parameters plus int8/scale buffers) and their bytes."""
    tensors = list(unet.parameters()) + [b for name, b in unet.named_buffers() if name.endswith(("qweight", "scale"))]
    return {
        "parameters": sum(t.numel() for t in tensors),
        "bytes": sum(t.numel() * t.element_size() for t in tensors),
    }
