"""Method-aware inference-policy search space (``search_space: "v2"``).

Fixes the redundancy of the v1 16-bit encoding, where 65,536 genotypes
decoded to ~2,000 distinct policies and half of random draws were one of
32 baselines. Here:

- A method is drawn uniformly, then only that method's genes; inactive
  genes are zeroed so every distinct genome is a distinct policy.
- Mutation resamples one *active* gene to a different value, so it always
  changes the executed policy; crossover only mixes genes of the same method.
- Duplicates are detected on the execution signature (scheduler, guidance,
  timesteps and per-step full/partial/null actions), which also catches
  cross-method equivalents such as a uniform Flexiffusion schedule that
  equals a DeepCache policy.

Methods: native ``baseline`` (DDIM or DPM-Solver++), ``deepcache`` (uniform
interval >= 2; interval 1 executes no caching) and ``flexiffusion``
(per-segment branch/interval, segment length 3, DDIM).
"""
from __future__ import annotations

import json
import random
from typing import Any

from .flexiffusion import NULL, PARTIAL, SD15_BRANCH_CHOICES, SD15_SEGMENT_LENGTH, expand_schedule

METHODS = ("baseline", "deepcache", "flexiffusion")
SCHEDULERS = ("ddim", "dpm_solver++")
STEP_CHOICES = (4, 6, 8, 10, 12, 15, 20, 25)
GUIDANCE_CHOICES = (5.0, 6.5, 7.5, 9.0, 10.5, 12.0)
DEEPCACHE_INTERVALS = (2, 3, 4, 5)
DEEPCACHE_BRANCHES = tuple(range(12))
FLEX_N_SEGMENTS = (4, 5, 6, 7, 8, 9, 10, 11)
FLEX_BRANCHES = SD15_BRANCH_CHOICES
FLEX_INTERVALS = (1, 2, 3)
MAX_SEGMENTS = max(FLEX_N_SEGMENTS)

METHOD, SCHEDULER, STEPS, GUIDANCE, DC_INTERVAL, DC_BRANCH, FLEX_N = range(7)
FLEX_BRANCH0 = 7
FLEX_INTERVAL0 = FLEX_BRANCH0 + MAX_SEGMENTS
GENOME_LENGTH = FLEX_INTERVAL0 + MAX_SEGMENTS
CHOICES: dict[int, tuple] = {
    METHOD: METHODS,
    SCHEDULER: SCHEDULERS,
    STEPS: STEP_CHOICES,
    GUIDANCE: GUIDANCE_CHOICES,
    DC_INTERVAL: DEEPCACHE_INTERVALS,
    DC_BRANCH: DEEPCACHE_BRANCHES,
    FLEX_N: FLEX_N_SEGMENTS,
    **{FLEX_BRANCH0 + i: FLEX_BRANCHES for i in range(MAX_SEGMENTS)},
    **{FLEX_INTERVAL0 + i: FLEX_INTERVALS for i in range(MAX_SEGMENTS)},
}


def active_genes(genome: list[int]) -> list[int]:
    """Genes that change execution for this genome's method (method gene excluded)."""
    method = METHODS[genome[METHOD]]
    if method == "baseline":
        return [SCHEDULER, STEPS, GUIDANCE]
    if method == "deepcache":
        return [SCHEDULER, STEPS, GUIDANCE, DC_INTERVAL, DC_BRANCH]
    n_segment = FLEX_N_SEGMENTS[genome[FLEX_N]]
    genes = [GUIDANCE, FLEX_N]
    for i in range(n_segment):
        genes.append(FLEX_INTERVAL0 + i)
        # A segment with interval 1 has no partial step, so its branch is inert.
        if FLEX_INTERVALS[genome[FLEX_INTERVAL0 + i]] > 1:
            genes.append(FLEX_BRANCH0 + i)
    return genes


def canonical(genome: list[int]) -> list[int]:
    if len(genome) != GENOME_LENGTH:
        raise ValueError(f"expected {GENOME_LENGTH} genes, got {len(genome)}")
    keep = set(active_genes(genome)) | {METHOD}
    return [value if index in keep else 0 for index, value in enumerate(genome)]


def random_genome(rng: random.Random) -> list[int]:
    genome = [0] * GENOME_LENGTH
    genome[METHOD] = rng.randrange(len(METHODS))
    # Sample every gene so the draw count is method-independent, then zero inactive ones.
    for index in range(1, GENOME_LENGTH):
        genome[index] = rng.randrange(len(CHOICES[index]))
    return canonical(genome)


def decode(genome: list[int]) -> dict[str, Any]:
    """Decode a genome into the shared policy dict consumed by ``policy_to_config``."""
    genome = canonical(genome)
    method = METHODS[genome[METHOD]]
    guidance = GUIDANCE_CHOICES[genome[GUIDANCE]]
    if method == "flexiffusion":
        n_segment = FLEX_N_SEGMENTS[genome[FLEX_N]]
        segments = []
        for i in range(n_segment):
            interval = FLEX_INTERVALS[genome[FLEX_INTERVAL0 + i]]
            # Inert branch is stored as the first choice so equal schedules decode identically.
            branch = FLEX_BRANCHES[genome[FLEX_BRANCH0 + i]]
            segments.append({"branch": branch, "interval": interval})
        return {
            "method": method,
            "scheduler": "ddim",
            "num_inference_steps": n_segment * SD15_SEGMENT_LENGTH,
            "guidance_scale": guidance,
            "flexiffusion": {"segment_length": SD15_SEGMENT_LENGTH, "segments": segments},
        }
    policy: dict[str, Any] = {
        "method": method,
        "scheduler": SCHEDULERS[genome[SCHEDULER]],
        "num_inference_steps": STEP_CHOICES[genome[STEPS]],
        "guidance_scale": guidance,
    }
    if method == "deepcache":
        policy["deepcache_interval"] = DEEPCACHE_INTERVALS[genome[DC_INTERVAL]]
        policy["deepcache_branch_id"] = DEEPCACHE_BRANCHES[genome[DC_BRANCH]]
    return policy


def step_actions(policy: dict[str, Any]) -> list[str]:
    """Per-step execution as strings: ``F``, ``P<branch>`` or ``N``."""
    steps = policy["num_inference_steps"]
    if policy["method"] == "baseline":
        return ["F"] * steps
    if policy["method"] == "deepcache":
        interval, branch = policy["deepcache_interval"], policy["deepcache_branch_id"]
        return ["F" if i % interval == 0 else f"P{branch}" for i in range(steps)]
    flex = policy["flexiffusion"]
    labels = []
    for action in expand_schedule(flex["segments"], flex["segment_length"]):
        labels.append("N" if action.kind == NULL else f"P{action.branch}" if action.kind == PARTIAL else "F")
    return labels


def signature(policy: dict[str, Any]) -> str:
    """Execution identity: two policies with equal signatures produce identical images."""
    return json.dumps(
        {
            "scheduler": policy["scheduler"],
            "steps": policy["num_inference_steps"],
            "guidance": policy["guidance_scale"],
            "actions": step_actions(policy),
        },
        sort_keys=True,
    )


def mutate(genome: list[int], rng: random.Random, method_switch_rate: float = 0.1) -> list[int]:
    """Change exactly one executed property; occasionally switch method."""
    child = list(canonical(genome))
    if rng.random() < method_switch_rate:
        child[METHOD] = rng.choice([m for m in range(len(METHODS)) if m != child[METHOD]])
        for index in active_genes(child):
            child[index] = rng.randrange(len(CHOICES[index]))
        return canonical(child)
    index = rng.choice(active_genes(child))
    child[index] = rng.choice([v for v in range(len(CHOICES[index])) if v != child[index]])
    return canonical(child)


def crossover(left: list[int], right: list[int], rng: random.Random) -> tuple[list[int], list[int]]:
    """Uniform crossover over active genes of parents that share a method."""
    a, b = canonical(left), canonical(right)
    if a[METHOD] != b[METHOD]:
        return a, b
    genes = sorted(set(active_genes(a)) | set(active_genes(b)))
    for index in genes:
        if rng.random() < 0.5:
            a[index], b[index] = b[index], a[index]
    return canonical(a), canonical(b)


def encode(policy: dict[str, Any]) -> list[int]:
    genome = [0] * GENOME_LENGTH
    genome[METHOD] = METHODS.index(policy["method"])
    genome[GUIDANCE] = GUIDANCE_CHOICES.index(policy["guidance_scale"])
    if policy["method"] == "flexiffusion":
        segments = policy["flexiffusion"]["segments"]
        genome[FLEX_N] = FLEX_N_SEGMENTS.index(len(segments))
        for i, segment in enumerate(segments):
            genome[FLEX_BRANCH0 + i] = FLEX_BRANCHES.index(segment["branch"])
            genome[FLEX_INTERVAL0 + i] = FLEX_INTERVALS.index(segment["interval"])
        return canonical(genome)
    genome[SCHEDULER] = SCHEDULERS.index(policy["scheduler"])
    genome[STEPS] = STEP_CHOICES.index(policy["num_inference_steps"])
    if policy["method"] == "deepcache":
        genome[DC_INTERVAL] = DEEPCACHE_INTERVALS.index(policy["deepcache_interval"])
        genome[DC_BRANCH] = DEEPCACHE_BRANCHES.index(policy["deepcache_branch_id"])
    return canonical(genome)


def initial_genomes() -> list[list[int]]:
    """Known reference policies used to seed the evolutionary population."""
    return [
        encode({"method": "baseline", "scheduler": "ddim", "num_inference_steps": 20, "guidance_scale": 7.5}),
        encode({"method": "baseline", "scheduler": "dpm_solver++", "num_inference_steps": 4, "guidance_scale": 7.5}),
        encode({
            "method": "deepcache", "scheduler": "ddim", "num_inference_steps": 20, "guidance_scale": 7.5,
            "deepcache_interval": 3, "deepcache_branch_id": 0,
        }),
        encode({
            "method": "flexiffusion", "scheduler": "ddim", "num_inference_steps": 27, "guidance_scale": 7.5,
            "flexiffusion": {"segment_length": 3, "segments": [{"branch": 3, "interval": 2}] * 9},
        }),
    ]


def novel(
    propose,
    seen: set[str],
    rng: random.Random,
    max_tries: int = 1000,
) -> list[int]:
    """Return the first proposal whose execution signature is unseen, else a novel random genome."""
    for _ in range(max_tries):
        genome = propose()
        if signature(decode(genome)) not in seen:
            return genome
    for _ in range(max_tries):
        genome = random_genome(rng)
        if signature(decode(genome)) not in seen:
            return genome
    raise RuntimeError("search space exhausted: no novel policy found")


def config_signature(config: dict[str, Any]) -> str:
    """Execution signature of a run config, including custom-timestep (AutoDiffusion) runs."""
    inference = config["inference"]
    policy: dict[str, Any] = {
        "method": "baseline" if config["method"] == "autodiffusion" else config["method"],
        "scheduler": inference["scheduler"],
        "num_inference_steps": inference["num_inference_steps"],
        "guidance_scale": inference["guidance_scale"],
    }
    if config["method"] == "deepcache":
        policy["deepcache_interval"] = config["deepcache"]["interval"]
        policy["deepcache_branch_id"] = config["deepcache"]["branch_id"]
    if config["method"] == "flexiffusion":
        policy["flexiffusion"] = config["flexiffusion"]
    identity = json.loads(signature(policy))
    identity["timesteps"] = inference.get("timesteps")
    return json.dumps(identity, sort_keys=True)
