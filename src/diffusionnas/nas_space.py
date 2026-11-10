"""v3 search space: static U-Net architecture + weight precision + execution policy.

Genome = v2 execution genome (method, sampler, steps, guidance, DeepCache,
Flexiffusion; see ``policy_space``) followed by one removal bit per searchable
unit and one weight-precision gene. The searchable units come from
``configs/nas/units.json``, chosen from the calibration sensitivity probe by a
rule fixed before any search ran.

Sampling draws the number of removed units uniformly from 0..len(units) and
then a uniform subset, so random search covers light and heavy compression
equally instead of concentrating on removing half the units.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from . import policy_space as exec_space
from .architecture import UNIT_NAMES

UNITS_FILE = Path(__file__).resolve().parents[2] / "configs" / "nas" / "units.json"
PRECISIONS = ("fp16", "int8")


def load_units(path: Path = UNITS_FILE) -> tuple[str, ...]:
    units = tuple(json.loads(path.read_text(encoding="utf-8"))["units"])
    # Keep the canonical architecture order so equal subsets serialize identically.
    return tuple(name for name in UNIT_NAMES if name in units)


UNITS = load_units()
EXEC_LENGTH = exec_space.GENOME_LENGTH
MASK0 = EXEC_LENGTH
PRECISION = MASK0 + len(UNITS)
GENOME_LENGTH = PRECISION + 1


def split(genome: list[int]) -> tuple[list[int], list[int], int]:
    if len(genome) != GENOME_LENGTH:
        raise ValueError(f"expected {GENOME_LENGTH} genes, got {len(genome)}")
    return list(genome[:EXEC_LENGTH]), list(genome[MASK0:PRECISION]), genome[PRECISION]


def canonical(genome: list[int]) -> list[int]:
    exec_genes, mask, precision = split(genome)
    return exec_space.canonical(exec_genes) + [1 if bit else 0 for bit in mask] + [precision]


def random_mask(rng: random.Random) -> list[int]:
    count = rng.randint(0, len(UNITS))
    chosen = set(rng.sample(range(len(UNITS)), count))
    return [1 if i in chosen else 0 for i in range(len(UNITS))]


def random_genome(rng: random.Random) -> list[int]:
    return exec_space.random_genome(rng) + random_mask(rng) + [rng.randrange(len(PRECISIONS))]


def decode(genome: list[int]) -> dict[str, Any]:
    exec_genes, mask, precision = split(canonical(genome))
    policy = exec_space.decode(exec_genes)
    policy["architecture"] = {
        "removed_units": [unit for unit, bit in zip(UNITS, mask) if bit],
        "weight_precision": PRECISIONS[precision],
    }
    return policy


def signature(policy: dict[str, Any]) -> str:
    identity = json.loads(exec_space.signature(policy))
    architecture = policy.get("architecture") or {}
    identity["removed_units"] = sorted(architecture.get("removed_units", []))
    identity["weight_precision"] = architecture.get("weight_precision", "fp16")
    return json.dumps(identity, sort_keys=True)


def mutate(genome: list[int], rng: random.Random, method_switch_rate: float = 0.1) -> list[int]:
    """Change exactly one property: half the time the architecture, otherwise the execution policy."""
    exec_genes, mask, precision = split(canonical(genome))
    if rng.random() < 0.5:
        position = rng.randrange(len(UNITS) + 1)
        if position == len(UNITS):
            precision = 1 - precision
        else:
            mask[position] = 1 - mask[position]
    else:
        exec_genes = exec_space.mutate(exec_genes, rng, method_switch_rate)
    return canonical(exec_genes + mask + [precision])


def crossover(left: list[int], right: list[int], rng: random.Random) -> tuple[list[int], list[int]]:
    """Uniform crossover on architecture genes; execution genes mix only within one method."""
    exec_a, mask_a, prec_a = split(canonical(left))
    exec_b, mask_b, prec_b = split(canonical(right))
    exec_a, exec_b = exec_space.crossover(exec_a, exec_b, rng)
    for i in range(len(UNITS)):
        if rng.random() < 0.5:
            mask_a[i], mask_b[i] = mask_b[i], mask_a[i]
    if rng.random() < 0.5:
        prec_a, prec_b = prec_b, prec_a
    return canonical(exec_a + mask_a + [prec_a]), canonical(exec_b + mask_b + [prec_b])


def novel(propose, seen: set[str], rng: random.Random, max_tries: int = 1000) -> list[int]:
    for _ in range(max_tries):
        genome = propose()
        if signature(decode(genome)) not in seen:
            return genome
    for _ in range(max_tries):
        genome = random_genome(rng)
        if signature(decode(genome)) not in seen:
            return genome
    raise RuntimeError("search space exhausted: no novel policy found")
