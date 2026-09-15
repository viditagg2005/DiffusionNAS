from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from .pareto import construct_front
from .quality import ClipScorer, DEFAULT_CLIP_MODEL
from .runner import execute
from .schema import validate_config


def load_prompts(path: str | Path) -> list[str]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("prompts")
    if not isinstance(value, list) or not value:
        raise ValueError("prompt file must contain a non-empty JSON list or {'prompts': [...]} object")
    prompts: list[str] = []
    for item in value:
        prompt = item.get("prompt") if isinstance(item, dict) else item
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("every hold-out prompt must be a non-empty string")
        prompts.append(prompt)
    return prompts


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks, right_ranks = _ranks(left), _ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left_ranks))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right_ranks))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def validate_holdout(
    front_path: str | Path,
    prompts_path: str | Path,
    seeds: list[int],
    project_root: Path,
    output_dir: str | Path,
    dry_run: bool = False,
    clip_model_id: str = DEFAULT_CLIP_MODEL,
    score_device: str = "cuda",
    batch_size: int = 8,
) -> dict[str, Any]:
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("hold-out seeds must be non-negative integers")
    front = json.loads(Path(front_path).read_text(encoding="utf-8"))
    prompts = load_prompts(prompts_path)
    destination = Path(output_dir).resolve()
    runs_root = destination / "runs"
    destination.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[dict[str, Any], Path]] = []
    for point in front.get("points", []):
        source_run = Path(point["run_dir"])
        config = deepcopy(json.loads((source_run / "config.json").read_text(encoding="utf-8")))
        config["prompts"] = prompts
        config["seeds"] = seeds
        config["runtime"]["repetitions"] = 1
        config["output_root"] = str(runs_root)
        config["holdout"] = {
            "source_run_id": point["run_id"],
            "prompt_file": str(Path(prompts_path).resolve()),
        }
        validate_config(config)
        generated.append((point, execute(config, project_root, dry_run=dry_run)))
    if not generated:
        raise ValueError("Pareto front contains no policies")

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned" if dry_run else "succeeded",
        "calibration_front": str(Path(front_path).resolve()),
        "holdout_prompt_file": str(Path(prompts_path).resolve()),
        "holdout_prompt_count": len(prompts),
        "seeds": seeds,
        "runs": [
            {"source_run_id": point["run_id"], "holdout_run_dir": str(run_dir)}
            for point, run_dir in generated
        ],
    }
    if dry_run:
        (destination / "holdout_validation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report

    scorer = ClipScorer(model_id=clip_model_id, device=score_device, batch_size=batch_size)
    for _, run_dir in generated:
        scorer.score_run(run_dir)
    holdout_front_path = destination / "holdout_pareto_front.json"
    holdout_front = construct_front(
        [run_dir for _, run_dir in generated],
        holdout_front_path,
        include_energy=any(item["name"] == "energy_joules_per_image" for item in front["objectives"]),
    )
    source_by_holdout = {
        run_dir.name: point["run_id"] for point, run_dir in generated
    }
    retained = [source_by_holdout[point["run_id"]] for point in holdout_front["points"]]
    calibration_clip: list[float] = []
    holdout_clip: list[float] = []
    for source_point, run_dir in generated:
        calibration_clip.append(float(source_point["objectives"]["clip_score"]))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        holdout_clip.append(float(summary["quality_metrics"]["clip_score"]["mean"]))
    report.update(
        {
            "holdout_front": str(holdout_front_path),
            "retained_source_run_ids": retained,
            "retention_fraction": len(retained) / len(generated),
            "clip_spearman": spearman(calibration_clip, holdout_clip),
        }
    )
    (destination / "holdout_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report

