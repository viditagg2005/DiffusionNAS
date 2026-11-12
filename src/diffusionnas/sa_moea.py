"""Surrogate-assisted many-objective search for expensive, noisy diffusion-policy evaluation.

Real evaluations are scarce (150 per run). Each round:
1. refit an internal surrogate (ExtraTrees on policy features) to every point evaluated so far,
2. run a many-objective optimizer (LHFiD or NSGA-II) for thousands of *virtual*
   evaluations on the surrogate, excluding already evaluated policies,
3. evaluate a batch of novel candidates for real, chosen from the optimizer's final
   population: predicted-nondominated first, spread by per-objective best then
   farthest-point in predicted objective space.

The noise-aware variant passes the measured per-objective noise SD to LHFiD so
that quality differences within the noise count as ties in hf-dominance.
Objectives are minimized and already on a common fixed scale.
"""
from __future__ import annotations

import random
from typing import Callable

import numpy as np

from . import moea, nas_space
from .surrogate import features

Observe = Callable[[list[dict]], np.ndarray]


class InternalSurrogate:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.model = None

    def fit(self, policies: list[dict], f: np.ndarray) -> "InternalSurrogate":
        from sklearn.ensemble import ExtraTreesRegressor

        self.model = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, random_state=self.seed, n_jobs=1)
        self.model.fit(np.array([features(p) for p in policies]), f)
        return self

    def predict(self, policies: list[dict]) -> np.ndarray:
        return self.model.predict(np.array([features(p) for p in policies]))


def spread_pick(f: np.ndarray, count: int) -> list[int]:
    """Best point on each objective, then farthest-point fill (indices into f)."""
    chosen: list[int] = []
    for j in range(f.shape[1]):
        i = int(np.argmin(f[:, j]))
        if i not in chosen:
            chosen.append(i)
    while len(chosen) < min(count, len(f)):
        distance = np.array([min(np.sum((f[i] - f[c]) ** 2) for c in chosen) if i not in chosen else -1.0
                             for i in range(len(f))])
        chosen.append(int(np.argmax(distance)))
    return chosen[:count]


def run_sa(inner: str, observe: Observe, budget: int, seed: int, vectors: np.ndarray, initial: int = 50,
           batch: int = 20, inner_population: int = 56, inner_generations: int = 30,
           tie_tolerance: np.ndarray | None = None) -> dict:
    rng = random.Random(seed)
    seen: set[str] = set()
    policies: list[dict] = []
    observed: list[np.ndarray] = []

    def evaluate(new: list[dict]) -> None:
        for policy in new:
            seen.add(nas_space.signature(policy))
        policies.extend(new)
        observed.extend(observe(new))

    start = []
    for _ in range(min(initial, budget)):
        genome = nas_space.novel(lambda: nas_space.random_genome(rng), seen | {nas_space.signature(p) for p in start}, rng)
        start.append(nas_space.decode(genome))
    evaluate(start)
    round_index = 0
    while len(policies) < budget:
        round_index += 1
        model = InternalSurrogate(seed * 100 + round_index).fit(policies, np.array(observed))
        result = moea.run(inner, model.predict, inner_population * inner_generations, inner_population, vectors,
                          seed=seed * 100 + round_index, tie_tolerance=tie_tolerance, exclude=set(seen))
        # Candidates come from the optimizer's final population, so its selection scheme matters;
        # its wider virtual archive is used only if the population has too few novel members.
        take = min(batch, budget - len(policies))
        new: list[dict] = []
        for source in (result["population_policies"], result["policies"]):
            taken = seen | {nas_space.signature(p) for p in new}
            unique: dict[str, dict] = {}
            for policy in source:
                key = nas_space.signature(policy)
                if key not in taken:
                    unique.setdefault(key, policy)
            pool = list(unique.values())
            if not pool or len(new) >= take:
                continue
            predicted = model.predict(pool)
            front = moea.nondominated_sort(predicted)[0]
            order = [front[i] for i in spread_pick(predicted[front], take - len(new))]
            rest = [i for i in range(len(pool)) if i not in set(order)]
            if len(order) < take - len(new) and rest:
                order += [rest[i] for i in spread_pick(predicted[rest], take - len(new) - len(order))]
            new += [pool[i] for i in order[: take - len(new)]]
        while len(new) < take:  # degenerate case: top up with novel random policies
            taken = seen | {nas_space.signature(p) for p in new}
            new.append(nas_space.decode(nas_space.novel(lambda: nas_space.random_genome(rng), taken, rng)))
        evaluate(new)
    return {"policies": policies, "observed": np.array(observed)}
