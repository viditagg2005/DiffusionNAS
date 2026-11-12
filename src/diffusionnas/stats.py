"""Paired, prompt-level bootstrap used for every quality comparison in the study."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


FIELD_FILES = {"clip_score": "quality_measurements.jsonl", "black_image": "quality_measurements.jsonl",
               "clip_image_fidelity": "fidelity_measurements.jsonl"}


def per_prompt_scores(run_dir: str | Path, field: str = "clip_score") -> dict[int, float]:
    """Mean of a per-image metric over seeds, keyed by prompt index."""
    by_prompt: dict[int, list[float]] = defaultdict(list)
    for line in Path(run_dir, FIELD_FILES[field]).read_text(encoding="utf-8").splitlines():
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
    at_or_below = sum(m <= 0 for m in means) / draws
    at_or_above = sum(m >= 0 for m in means) / draws
    return {
        "mean_difference": statistics.fmean(diffs),
        "ci95_low": means[int(0.025 * draws)],
        "ci95_high": means[int(0.975 * draws) - 1],
        # Two-sided bootstrap p-value for a zero mean difference.
        "p_value": min(1.0, 2 * min(at_or_below, at_or_above) + 1 / draws),
        "prompts": len(shared),
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[key] = running
    return adjusted
