import json
from pathlib import Path
import tempfile
import unittest

from diffusionnas.holdout import load_prompts, spearman, validate_holdout
from diffusionnas.pareto import construct_front, dominates
from diffusionnas.quality import image_records, summarize_clip_scores


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(root: Path, run_id: str, method: str, clip: float, latency: float, memory: float) -> Path:
    run_dir = root / run_id
    (run_dir / "images").mkdir(parents=True)
    config = json.loads((ROOT / "configs" / f"{method}_sd15.json").read_text())
    write_json(run_dir / "config.json", config)
    write_json(
        run_dir / "manifest.json",
        {
            "status": "succeeded",
            "run_id": run_id,
            "method": method,
            "implementation_fidelity": "adapted" if method == "autodiffusion" else "paper_implementation",
        },
    )
    write_json(
        run_dir / "summary.json",
        {
            "latency_ms": {"p50": latency},
            "peak_allocated_mb": memory,
            "energy_joules_per_image": None,
            "quality_metrics": {"clip_score": {"mean": clip}},
        },
    )
    return run_dir


class QualityTests(unittest.TestCase):
    def test_clip_summary(self):
        summary = summarize_clip_scores([10.0, 20.0, 30.0], "clip/test")
        self.assertEqual(summary["mean"], 20.0)
        self.assertEqual(summary["count"], 3)

    def test_image_records_follow_prompt_seed_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "images").mkdir()
            write_json(run_dir / "config.json", {"prompts": ["one"], "seeds": [7, 9]})
            for seed in (7, 9):
                (run_dir / "images" / f"p0000_s{seed}.png").touch()
            records = image_records(run_dir)
            self.assertEqual([(record.prompt, record.seed) for record in records], [("one", 7), ("one", 9)])


class ParetoTests(unittest.TestCase):
    def test_dominance_respects_mixed_directions(self):
        objectives = (("quality", "max"), ("latency", "min"))
        self.assertTrue(dominates({"quality": 8, "latency": 2}, {"quality": 7, "latency": 3}, objectives))
        self.assertFalse(dominates({"quality": 8, "latency": 4}, {"quality": 7, "latency": 3}, objectives))

    def test_construct_front_excludes_dominated_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            best = make_run(root, "best", "deepcache", clip=30, latency=100, memory=2000)
            dominated = make_run(root, "dominated", "baseline", clip=20, latency=200, memory=2500)
            tradeoff = make_run(root, "tradeoff", "autodiffusion", clip=31, latency=250, memory=2100)
            result = construct_front([best, dominated, tradeoff], root / "front.json")
            self.assertEqual({point["run_id"] for point in result["points"]}, {"best", "tradeoff"})
            self.assertTrue((root / "front.csv").exists())


class HoldoutTests(unittest.TestCase):
    def test_prompt_formats_and_spearman(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.json"
            write_json(path, {"prompts": ["a", {"prompt": "b"}]})
            self.assertEqual(load_prompts(path), ["a", "b"])
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [30, 20, 10]), -1.0)

    def test_holdout_dry_run_materializes_selected_policies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_run(root, "source", "deepcache", clip=30, latency=100, memory=2000)
            front_path = root / "front.json"
            write_json(
                front_path,
                {
                    "objectives": [
                        {"name": "clip_score", "direction": "max"},
                        {"name": "latency_p50_ms", "direction": "min"},
                    ],
                    "points": [
                        {
                            "run_id": "source",
                            "run_dir": str(source),
                            "objectives": {"clip_score": 30, "latency_p50_ms": 100},
                        }
                    ],
                },
            )
            prompts = root / "holdout.json"
            write_json(prompts, {"prompts": ["held out prompt"]})
            report = validate_holdout(
                front_path,
                prompts,
                seeds=[101, 202],
                project_root=ROOT,
                output_dir=root / "validation",
                dry_run=True,
            )
            self.assertEqual(report["status"], "planned")
            self.assertEqual(report["holdout_prompt_count"], 1)
            holdout_config = json.loads(
                (Path(report["runs"][0]["holdout_run_dir"]) / "config.json").read_text()
            )
            self.assertEqual(holdout_config["prompts"], ["held out prompt"])
            self.assertEqual(holdout_config["seeds"], [101, 202])


if __name__ == "__main__":
    unittest.main()
