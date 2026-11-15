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
    """ExtraTrees (default, as in C2); "randomforest", "knn" and "gp" exist for the ablation study."""

    def __init__(self, seed: int, family: str = "extratrees") -> None:
        self.seed = seed
        self.family = family
        self.model = None

    def fit(self, policies: list[dict], f: np.ndarray) -> "InternalSurrogate":
        x = np.array([features(p) for p in policies])
        if self.family == "extratrees":
            from sklearn.ensemble import ExtraTreesRegressor

            self.model = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, random_state=self.seed, n_jobs=1)
        elif self.family == "randomforest":
            from sklearn.ensemble import RandomForestRegressor

            self.model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=self.seed, n_jobs=1)
        elif self.family == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            self.model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance"))
        elif self.family == "gp":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
            self.model = make_pipeline(StandardScaler(), MultiOutputRegressor(
                GaussianProcessRegressor(kernel, normalize_y=True, random_state=self.seed)))
        else:
            raise ValueError(self.family)
        self.model.fit(x, f)
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
           tie_tolerance: np.ndarray | None = None, surrogate_family: str = "extratrees",
           pick: str = "spread") -> dict:
    """``surrogate_family`` and ``pick`` ("spread" as in C2, or "hssp": greedy hypervolume subset selection in
    predicted space, reference 1.1 on the fixed scale) exist for the ablation study; defaults reproduce C2."""
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
        model = InternalSurrogate(seed * 100 + round_index, surrogate_family).fit(policies, np.array(observed))
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
            if pick == "hssp":
                from .final_selection import greedy_hssp

                order = [front[i] for i in greedy_hssp(predicted[front], np.full(predicted.shape[1], 1.1), take - len(new))]
                if len(order) < take - len(new):
                    spare = [i for i in front if i not in set(order)]
                    order += [spare[i] for i in spread_pick(predicted[spare], take - len(new) - len(order))] if spare else []
            else:
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
