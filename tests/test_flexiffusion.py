import copy
import importlib.util
import json
from pathlib import Path
import unittest

from diffusionnas.flexiffusion import (
    FULL,
    NULL,
    PARTIAL,
    expand_schedule,
    schedule_stats,
    uniform_segments,
)
from diffusionnas.schema import ConfigError, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]
HAS_DIFFUSERS = importlib.util.find_spec("diffusers") is not None and importlib.util.find_spec("torch") is not None
SD15_SCHEDULER = {
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "num_train_timesteps": 1000,
    "steps_offset": 1,
    "clip_sample": False,
    "set_alpha_to_one": False,
}


class ScheduleTests(unittest.TestCase):
    def test_segment_layout_is_full_then_partial_then_null(self):
        actions = expand_schedule([{"branch": 3, "interval": 2}, {"branch": 6, "interval": 3}], 3)
        self.assertEqual([a.kind for a in actions], [FULL, PARTIAL, NULL, FULL, PARTIAL, PARTIAL])
        self.assertEqual([a.branch for a in actions if a.kind == PARTIAL], [3, 6, 6])

    def test_stats_count_null_steps_out_of_nfe(self):
        stats = schedule_stats(expand_schedule(uniform_segments(9, 3, 2), 3))
        self.assertEqual(stats, {"steps": 27, "full_steps": 9, "partial_steps": 9, "null_steps": 9, "nfe": 18})

    def test_interval_cannot_exceed_segment_length(self):
        with self.assertRaises(ValueError):
            expand_schedule([{"branch": 1, "interval": 4}], 3)


class FlexiffusionSchemaTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs" / "flexiffusion_sd15.json")

    def test_step_count_must_match_segments(self):
        invalid = copy.deepcopy(self.config)
        invalid["inference"]["num_inference_steps"] = 20
        with self.assertRaises(ConfigError):
            validate_config(invalid)

    def test_requires_deterministic_ddim(self):
        invalid = copy.deepcopy(self.config)
        invalid["inference"]["scheduler"] = "dpm_solver++"
        with self.assertRaises(ConfigError):
            validate_config(invalid)

    def test_branch_range(self):
        invalid = copy.deepcopy(self.config)
        invalid["flexiffusion"]["segments"][0] = {"branch": 12, "interval": 2}
        with self.assertRaises(ConfigError):
            validate_config(invalid)


@unittest.skipUnless(HAS_DIFFUSERS, "diffusers/torch not installed")
class SamplerEquivalenceTests(unittest.TestCase):
    def test_null_step_equals_skipping_timestep_for_ddim(self):
        import torch
        from diffusers import DDIMScheduler

        scheduler = DDIMScheduler.from_config(SD15_SCHEDULER)
        scheduler.set_timesteps(9)
        actions = expand_schedule(uniform_segments(3, 1, 2), 3)
        alphas = scheduler.alphas_cumprod.double()

        def eps(x, t):
            return torch.tanh(x * (1.0 + int(t) / 1000.0))

        generator = torch.Generator().manual_seed(0)
        start = torch.randn(1, 4, 8, 8, generator=generator, dtype=torch.float64)

        # Path A: scheduler steps every timestep; null steps reuse the last prediction.
        x, last = start.clone(), None
        for t, action in zip(scheduler.timesteps, actions):
            if action.kind != NULL:
                last = eps(x, t)
            x = scheduler.step(last, t, x).prev_sample

        # Path B: deterministic DDIM jumps directly over the null timesteps.
        timesteps = [int(t) for t in scheduler.timesteps]
        active = [t for t, a in zip(timesteps, actions) if a.kind != NULL]
        targets = active[1:] + [timesteps[-1] - 1000 // 9]
        y = start.clone()
        for t, prev in zip(active, targets):
            e = eps(y, t)
            x0 = (y - (1 - alphas[t]).sqrt() * e) / alphas[t].sqrt()
            a_prev = alphas[prev] if prev >= 0 else scheduler.final_alpha_cumprod.double()
            y = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
        self.assertTrue(torch.allclose(x, y, atol=1e-6), float((x - y).abs().max()))

    def test_autodiffusion_list_is_native_trailing_spacing(self):
        from diffusers import DPMSolverMultistepScheduler

        config = json.loads((ROOT / "configs" / "autodiffusion_sd15.json").read_text())
        trailing = DPMSolverMultistepScheduler.from_config({**SD15_SCHEDULER, "timestep_spacing": "trailing"})
        trailing.set_timesteps(4)
        self.assertEqual(trailing.timesteps.tolist(), config["inference"]["timesteps"])
        default = DPMSolverMultistepScheduler.from_config(SD15_SCHEDULER)
        default.set_timesteps(4)
        self.assertEqual(default.timesteps.tolist(), [999, 749, 500, 250])


if __name__ == "__main__":
    unittest.main()
