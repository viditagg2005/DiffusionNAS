"""CPU tests for training-free U-Net unit removal and int8 weight storage on a tiny SD1.5-shaped U-Net."""
import copy
import importlib.util
from pathlib import Path
import unittest

from diffusionnas.architecture import UNIT_NAMES, validate_architecture
from diffusionnas.schema import ConfigError, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]
HAS_DIFFUSERS = importlib.util.find_spec("diffusers") is not None and importlib.util.find_spec("torch") is not None


def tiny_unet():
    from diffusers import UNet2DConditionModel

    # Same block types, layer counts and channel pattern as SD1.5 (320, 640, 1280, 1280), scaled down.
    return UNet2DConditionModel(
        sample_size=8,
        block_out_channels=(32, 64, 128, 128),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=2,
        cross_attention_dim=16,
        attention_head_dim=4,
        norm_num_groups=8,
    ).eval()


class ArchitectureValidationTests(unittest.TestCase):
    def test_unit_inventory_matches_sd15_layout(self):
        self.assertEqual(len(UNIT_NAMES), 24)
        self.assertEqual(sum(name.split(".")[1].startswith("attn") for name in UNIT_NAMES), 16)

    def test_rejects_unknown_and_duplicate_units(self):
        self.assertTrue(validate_architecture({"removed_units": ["up0.res0"]}))
        self.assertTrue(validate_architecture({"removed_units": ["mid.attn0", "mid.attn0"]}))
        self.assertTrue(validate_architecture({"weight_precision": "int4"}))
        self.assertEqual(validate_architecture({"removed_units": ["mid.attn0"], "weight_precision": "int8"}), [])

    def test_schema_checks_architecture(self):
        config = load_config(ROOT / "configs" / "baseline_sd15.json")
        valid = copy.deepcopy(config)
        valid["architecture"] = {"removed_units": ["mid.attn0"], "weight_precision": "fp16"}
        validate_config(valid)
        invalid = copy.deepcopy(config)
        invalid["architecture"] = {"removed_units": ["down1.res0"]}
        with self.assertRaises(ConfigError):
            validate_config(invalid)


@unittest.skipUnless(HAS_DIFFUSERS, "diffusers/torch not installed")
class ArchitectureEditTests(unittest.TestCase):
    def setUp(self):
        import torch

        torch.manual_seed(0)
        self.torch = torch
        self.unet = tiny_unet()
        self.sample = torch.randn(1, 4, 8, 8)
        self.text = torch.randn(1, 5, 16)

    def forward(self, unet):
        with self.torch.no_grad():
            return unet(self.sample, 10, encoder_hidden_states=self.text).sample

    def test_removing_every_unit_keeps_shapes_and_frees_parameters(self):
        from diffusionnas.architecture import remove_units, unet_size

        before = unet_size(self.unet)["parameters"]
        remove_units(self.unet, list(UNIT_NAMES))
        out = self.forward(self.unet)
        self.assertEqual(tuple(out.shape), (1, 4, 8, 8))
        self.assertTrue(self.torch.isfinite(out).all())
        self.assertLess(unet_size(self.unet)["parameters"], before * 0.8)

    def test_single_removal_changes_output_only_a_little_for_attention(self):
        from diffusionnas.architecture import remove_units

        reference = self.forward(self.unet)
        edited = copy.deepcopy(self.unet)
        remove_units(edited, ["mid.attn0"])
        changed = self.forward(edited)
        self.assertFalse(self.torch.equal(reference, changed))

    def test_int8_weights_are_close_and_smaller(self):
        from diffusionnas.architecture import quantize_weights_int8, unet_size

        reference = self.forward(self.unet)
        bytes_before = unet_size(self.unet)["bytes"]
        quantized = copy.deepcopy(self.unet)
        converted = quantize_weights_int8(quantized)
        self.assertGreater(converted, 50)
        out = self.forward(quantized)
        relative = (out - reference).norm() / reference.norm()
        self.assertLess(float(relative), 0.05)
        # fp32 test model: int8 weights are ~4x smaller (norm/bias parameters stay full precision).
        self.assertLess(unet_size(quantized)["bytes"], bytes_before * 0.35)


if __name__ == "__main__":
    unittest.main()
