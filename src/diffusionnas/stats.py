"""Paired, prompt-level bootstrap used for every quality comparison in the study."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def per_prompt_scores(run_dir: str | Path, field: str = "clip_score") -> dict[int, float]:
    """Mean of a per-image metric over seeds, keyed by prompt index."""
    by_prompt: dict[int, list[float]] = defaultdict(list)
    for line in Path(run_dir, "quality_measurements.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_prompt[row["prompt_index"]].append(float(row[field]))
    return {index: statistics.fmean(values) for index, values in by_prompt.items()}


def paired_bootstrap(
    candidate: dict[int, float], reference: dict[int, float], draws: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Mean candidate-minus-reference difference with a 95% percentile interval over prompts."""
    shared = sorted(set(candidate) & set(reference))
    if not shared:
        raise ValueError("no shared prompts")
    diffs = [candidate[i] - reference[i] for i in shared]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(draws))
    return {
        "mean_difference": statistics.fmean(diffs),
        "ci95_low": means[int(0.025 * draws)],
        "ci95_high": means[int(0.975 * draws) - 1],
        "prompts": len(shared),
    }
