import copy
import json
from pathlib import Path
import tempfile
import unittest

from diffusionnas.runner import execute
from diffusionnas.schema import ConfigError, load_config, validate_config
from diffusionnas.sd15_worker import _percentile


ROOT = Path(__file__).resolve().parents[1]


class SchemaTests(unittest.TestCase):
    def test_p90_uses_nearest_rank(self):
        self.assertEqual(_percentile([1.0, 2.0, 9.0], 0.9), 9.0)

    def test_all_example_configs_validate(self):
        for path in sorted((ROOT / "configs").glob("*.json")):
            with self.subTest(path=path.name):
                load_config(path)

    def test_autodiffusion_timesteps_must_descend(self):
        config = load_config(ROOT / "configs" / "autodiffusion_sd15.json")
        invalid = copy.deepcopy(config)
        invalid["inference"]["timesteps"] = [1, 2, 3, 4]
        with self.assertRaises(ConfigError):
            validate_config(invalid)

    def test_autodiffusion_requires_custom_timestep_capable_scheduler(self):
        config = load_config(ROOT / "configs" / "autodiffusion_sd15.json")
        invalid = copy.deepcopy(config)
        invalid["inference"]["scheduler"] = "ddim"
        with self.assertRaises(ConfigError):
            validate_config(invalid)

    def test_model_is_locked_to_sd15(self):
        config = load_config(ROOT / "configs" / "baseline_sd15.json")
        invalid = copy.deepcopy(config)
        invalid["model"]["id"] = "some/other-model"
        with self.assertRaises(ConfigError):
            validate_config(invalid)


class RunnerTests(unittest.TestCase):
    def test_every_adapter_dry_run_writes_standard_artifacts(self):
        expected_fidelity = {
            "baseline_sd15.json": "reference",
            "autodiffusion_sd15.json": "adapted",
            "deepcache_sd15.json": "paper_implementation",
        }
        with tempfile.TemporaryDirectory() as temporary:
            for filename, fidelity in expected_fidelity.items():
                with self.subTest(config=filename):
                    config = load_config(ROOT / "configs" / filename)
                    config["output_root"] = temporary
                    run_dir = execute(config, ROOT, dry_run=True)
                    manifest = json.loads((run_dir / "manifest.json").read_text())
                    summary = json.loads((run_dir / "summary.json").read_text())
                    self.assertEqual(manifest["status"], "dry_run")
                    self.assertEqual(summary["status"], "dry_run")
                    self.assertEqual(manifest["model_id"], "stable-diffusion-v1-5/stable-diffusion-v1-5")
                    self.assertEqual(manifest["implementation_fidelity"], fidelity)
                    self.assertTrue((run_dir / "stdout.log").exists())
                    self.assertTrue((run_dir / "stderr.log").exists())


if __name__ == "__main__":
    unittest.main()
