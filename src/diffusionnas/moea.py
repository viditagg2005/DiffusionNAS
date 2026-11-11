"""Many-objective selection schemes compared on identical variation operators.

All four optimizers draw candidates from ``nas_space`` with the same crossover
and mutation and never re-evaluate an execution signature; they differ only in
mating and environmental selection:

- random search (no selection),
- NSGA-II (Deb et al., 2002): nondominated rank + crowding distance, binary tournament,
- NSGA-III (Deb and Jain, 2014): Das-Dennis reference vectors, normalization by
  extreme-point intercepts, niche-preserving selection, random mating,
- LHFiD (Saxena, Mittal, Kapoor and Deb, COIN Report 2021020): reference-vector
  clusters, Pareto filtering, then localized high-fidelity dominance with
  objective preferences p_m = 1/w_m from each vector, axis vectors keep the
  alpha solution, empty vectors filled by the nearest leftovers; nadir point
  first estimated once the population stabilizes.

LHFiD is a reimplementation. Its stabilization tracking (Saxena and Kapoor,
EMO 2019) is approximated as described in the LHFiD paper: D_t is the mean
per-vector change between successive populations; the population is stable
when the running mean and standard deviation of D agree to ``np`` decimals for
``ns`` successive generations. Self-termination is disabled so every optimizer
gets the same evaluation budget. Objectives are minimized.
"""
from __future__ import annotations

from itertools import combinations
import random
from typing import Callable

import numpy as np

from . import nas_space

Evaluate = Callable[[list[dict]], np.ndarray]


# ---- geometry -------------------------------------------------------------

def das_dennis(m: int, p: int) -> np.ndarray:
    """All m-dimensional weight vectors with components in {0, 1/p, ..., 1} summing to 1."""
    points = []
    for bars in combinations(range(p + m - 1), m - 1):
        counts, previous = [], -1
        for bar in bars:
            counts.append(bar - previous - 1)
            previous = bar
        counts.append(p + m - 2 - previous)
        points.append([c / p for c in counts])
    return np.array(points)


def two_layer(m: int, p_boundary: int, p_inner: int) -> np.ndarray:
    """NSGA-III two-layer reference set: boundary layer plus an inner layer shrunk halfway to the centroid."""
    inner = 0.5 * das_dennis(m, p_inner) + 0.5 / m
    return np.vstack([das_dennis(m, p_boundary), inner])


def perpendicular_distance(f: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Distance of each objective vector (rows of f) to each reference line (rows of vectors)."""
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    projection = f @ unit.T
    return np.sqrt(np.maximum((f ** 2).sum(axis=1, keepdims=True) - projection ** 2, 0.0))


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_sort(f: np.ndarray) -> list[list[int]]:
    n = len(f)
    better = (f[:, None, :] <= f[None, :, :]).all(axis=2) & (f[:, None, :] < f[None, :, :]).any(axis=2)
    dominated_by = better.sum(axis=0)
    fronts, current = [], [i for i in range(n) if dominated_by[i] == 0]
    remaining = dominated_by.copy()
    while current:
        fronts.append(current)
        nxt = []
        for i in current:
            for j in np.nonzero(better[i])[0]:
                remaining[j] -= 1
                if remaining[j] == 0:
                    nxt.append(int(j))
        current = nxt
    return fronts


def hf_dominates(x: np.ndarray, y: np.ndarray, preference: np.ndarray) -> bool:
    """High-fidelity dominance: x better in at least as many objectives and smaller weighted change."""
    return bool((x < y).sum() >= (y < x).sum() and (preference * (x - y)).sum() < 0)


def nsga3_normalize(f: np.ndarray, ideal: np.ndarray, front0: list[int]) -> np.ndarray:
    """Nadir estimate from extreme-point hyperplane intercepts (NSGA-III), with a fallback."""
    m = f.shape[1]
    translated = f - ideal
    extremes = []
    for axis in range(m):
        weights = np.full(m, 1e-6)
        weights[axis] = 1.0
        extremes.append(int(np.argmin((translated / weights).max(axis=1))))
    fallback = translated[front0].max(axis=0)
    try:
        intercepts = 1.0 / np.linalg.solve(translated[extremes], np.ones(m))
        if np.any(~np.isfinite(intercepts)) or np.any(intercepts <= 1e-6):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = fallback
    intercepts = np.where(intercepts <= 1e-6, np.maximum(fallback, 1e-6), intercepts)
    return ideal + intercepts


# ---- environmental selection -----------------------------------------------

def crowding(f: np.ndarray) -> np.ndarray:
    n, m = f.shape
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(f[:, j])
        span = f[order[-1], j] - f[order[0], j]
        distance[order[0]] = distance[order[-1]] = np.inf
        if span > 0:
            distance[order[1:-1]] += (f[order[2:], j] - f[order[:-2], j]) / span
    return distance


def select_nsga2(f: np.ndarray, n: int, state: dict, rng: random.Random) -> list[int]:
    chosen: list[int] = []
    rank = np.zeros(len(f), dtype=int)
    crowd = np.zeros(len(f))
    for r, front in enumerate(nondominated_sort(f)):
        rank[front] = r
        crowd[front] = crowding(f[front])
        if len(chosen) + len(front) <= n:
            chosen.extend(front)
        else:
            order = sorted(front, key=lambda i: -crowd[i])
            chosen.extend(order[: n - len(chosen)])
            break
    state["rank"], state["crowd"] = rank, crowd
    return chosen


def select_nsga3(f: np.ndarray, n: int, state: dict, rng: random.Random) -> list[int]:
    vectors = state["vectors"]
    fronts = nondominated_sort(f)
    pool: list[int] = []
    for front in fronts:
        pool.extend(front)
        if len(pool) >= n:
            last = front
            break
    if len(pool) == n:
        return pool
    chosen = [i for i in pool if i not in set(last)]
    ideal = f[pool].min(axis=0)
    nadir = nsga3_normalize(f[pool], ideal, [pool.index(i) for i in fronts[0] if i in pool])
    normalized = (f[pool] - ideal) / np.maximum(nadir - ideal, 1e-12)
    distance = perpendicular_distance(normalized, vectors)
    niche = distance.argmin(axis=1)
    dist = distance[np.arange(len(pool)), niche]
    position = {i: k for k, i in enumerate(pool)}
    counts = np.zeros(len(vectors), dtype=int)
    for i in chosen:
        counts[niche[position[i]]] += 1
    candidates = list(last)
    excluded: set[int] = set()
    while len(chosen) < n:
        open_niches = [j for j in range(len(vectors)) if j not in excluded]
        least = min(counts[j] for j in open_niches)
        j = rng.choice([k for k in open_niches if counts[k] == least])
        members = [i for i in candidates if niche[position[i]] == j]
        if not members:
            excluded.add(j)
            continue
        pick = min(members, key=lambda i: dist[position[i]]) if counts[j] == 0 else rng.choice(members)
        chosen.append(pick)
        candidates.remove(pick)
        counts[j] += 1
    return chosen


def lhfid_preferences(vectors: np.ndarray, mode: str) -> np.ndarray:
    """Per-vector objective preferences: p_m = 1/w_m (paper's recommendation) or equal p_m = 1/M.

    Zero vector components make 1/w_m undefined; they are floored at the smallest
    positive component in the reference set (our choice, not specified in the paper).
    """
    if mode == "equal":
        return np.full(vectors.shape, 1.0 / vectors.shape[1])
    floor = vectors[vectors > 0].min()
    return 1.0 / np.maximum(vectors, floor)


def select_lhfid(f: np.ndarray, n: int, state: dict, rng: random.Random) -> list[int]:
    vectors = state["vectors"]
    preferences = state.setdefault("preferences", lhfid_preferences(vectors, state.get("preference_mode", "inverse_w")))
    state["ideal"] = np.minimum(state.get("ideal", f.min(axis=0)), f.min(axis=0))
    ideal, nadir = state["ideal"], state.get("nadir")
    scaled = (f - ideal) / np.maximum(nadir - ideal, 1e-12) if nadir is not None else f - ideal
    distance = perpendicular_distance(np.maximum(scaled, 0.0), vectors)
    nearest = distance.argmin(axis=1)
    axis_vector = (vectors > 0).sum(axis=1) == 1
    selected: list[int] = []
    taken: set[int] = set()
    empty: list[int] = []
    for j in range(len(vectors)):
        cluster = [i for i in range(len(f)) if nearest[i] == j]
        if not cluster:
            empty.append(j)
            continue
        cluster = [i for i in cluster if not any(dominates(scaled[k], scaled[i]) for k in cluster if k != i)]
        alpha = min(cluster, key=lambda i: distance[i, j])
        if not axis_vector[j]:
            beta = [i for i in cluster if i != alpha and hf_dominates(scaled[i], scaled[alpha], preferences[j])]
            if beta:
                alpha = min(beta, key=lambda i: distance[i, j])
        selected.append(alpha)
        taken.add(alpha)
    for j in empty:
        leftovers = [i for i in range(len(f)) if i not in taken]
        if not leftovers:
            break
        pick = min(leftovers, key=lambda i: distance[i, j])
        selected.append(pick)
        taken.add(pick)
    return selected[:n]


SELECTORS = {"nsga2": select_nsga2, "nsga3": select_nsga3, "lhfid": select_lhfid, "lhfid_equal": select_lhfid}


# ---- stabilization tracking (LHFiD nadir timing) ----------------------------

class Stabilization:
    def __init__(self, decimals: int = 2, generations: int = 10) -> None:
        self.decimals, self.generations = decimals, generations
        self.history: list[float] = []
        self.signatures: list[tuple[float, float]] = []

    def update(self, previous: np.ndarray, current: np.ndarray, vectors: np.ndarray) -> bool:
        """D_t: mean distance between each vector's nearest member in successive (scaled) populations."""
        def representatives(f):
            d = perpendicular_distance(np.maximum(f, 0.0), vectors)
            return f[d.argmin(axis=0)]
        self.history.append(float(np.linalg.norm(representatives(previous) - representatives(current), axis=1).mean()))
        signature = (round(float(np.mean(self.history)), self.decimals), round(float(np.std(self.history)), self.decimals))
        self.signatures.append(signature)
        recent = self.signatures[-self.generations:]
        return len(recent) == self.generations and len(set(recent)) == 1


# ---- optimization loop -------------------------------------------------------

def run(algorithm: str, evaluate: Evaluate, budget: int, population: int, vectors: np.ndarray | None,
        seed: int, crossover_rate: float = 0.9, mutation_rate: float = 1.0) -> dict:
    """Return every evaluated objective vector in evaluation order, the final population and metadata.

    ``evaluate`` must return objectives already on a common, fixed scale (LHFiD's
    pre-nadir translation and hf weighted sums are not scale-invariant).
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    archive_f: list[np.ndarray] = []

    def novel_batch(propose, count):
        genomes = []
        for _ in range(count):
            genome = nas_space.novel(propose, seen, rng)
            seen.add(nas_space.signature(nas_space.decode(genome)))
            genomes.append(genome)
        return genomes

    def evaluate_genomes(genomes):
        f = evaluate([nas_space.decode(g) for g in genomes])
        archive_f.extend(f)
        return f

    if algorithm == "random":
        while len(archive_f) < budget:
            batch = min(population, budget - len(archive_f))
            evaluate_genomes(novel_batch(lambda: nas_space.random_genome(rng), batch))
        return {"objectives": np.array(archive_f), "population": np.array(archive_f), "nadir_generations": []}

    select = SELECTORS[algorithm]
    state: dict = {"vectors": vectors, "preference_mode": "equal" if algorithm == "lhfid_equal" else "inverse_w"}
    genomes = novel_batch(lambda: nas_space.random_genome(rng), population)
    f = evaluate_genomes(genomes)
    stabilization = Stabilization()
    nadir_generations: list[int] = []
    generation = 0
    if algorithm == "nsga2":
        select(f, population, state, rng)
    while len(archive_f) < budget:
        generation += 1

        def mate():
            if algorithm == "nsga2":
                def tournament():
                    a, b = rng.sample(range(len(genomes)), 2)
                    key = lambda i: (state["rank"][i], -state["crowd"][i])
                    return a if key(a) <= key(b) else b
                p1, p2 = genomes[tournament()], genomes[tournament()]
            else:
                p1, p2 = rng.sample(genomes, 2)
            child = p1
            if rng.random() < crossover_rate:
                child, _ = nas_space.crossover(p1, p2, rng)
            if rng.random() < mutation_rate:
                child = nas_space.mutate(child, rng)
            return child

        offspring = novel_batch(mate, min(population, budget - len(archive_f)))
        child_f = evaluate_genomes(offspring)
        union_g, union_f = genomes + offspring, np.vstack([f, child_f])
        keep = select(union_f, population, state, rng)
        if algorithm.startswith("lhfid"):
            scale = lambda x: x - state["ideal"] if state.get("nadir") is None else (x - state["ideal"]) / np.maximum(state["nadir"] - state["ideal"], 1e-12)
            # Nadir (re-)estimated at every generation where mild stabilization is detected, as in the paper.
            if stabilization.update(scale(f), scale(union_f[keep]), vectors):
                fronts = nondominated_sort(union_f[keep])
                state["nadir"] = nsga3_normalize(union_f[keep], state["ideal"], fronts[0])
                nadir_generations.append(generation)
        genomes, f = [union_g[i] for i in keep], union_f[keep]
        if algorithm == "nsga2":
            select(f, population, state, rng)  # refresh rank/crowding for tournament on survivors
    return {"objectives": np.array(archive_f), "population": f, "nadir_generations": nadir_generations}


# ---- indicators ----------------------------------------------------------------

def hypervolume_mc(f: np.ndarray, reference: np.ndarray, ideal: np.ndarray, samples: np.ndarray) -> float:
    """Monte Carlo hypervolume fraction of the box [ideal, reference] dominated by f (minimization).

    ``samples`` are fixed uniform points in the unit box, shared by every run so
    comparisons are paired and deterministic.
    """
    inside = f[(f < reference).all(axis=1)]
    if len(inside) == 0:
        return 0.0
    inside = inside[[i for i in nondominated_sort(inside)[0]]]
    points = ideal + samples * (reference - ideal)
    dominated = np.zeros(len(points), dtype=bool)
    for start in range(0, len(inside), 64):
        block = inside[start:start + 64]
        dominated |= (block[None, :, :] <= points[:, None, :]).all(axis=2).any(axis=1)
    return float(dominated.mean())


def igd_plus(f: np.ndarray, reference_front: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """IGD+ of the nondominated part of f to a reference front, in normalized objective space."""
    scale = np.maximum(high - low, 1e-12)
    a = (f[nondominated_sort(f)[0]] - low) / scale
    z = (reference_front - low) / scale
    gaps = np.sqrt((np.maximum(a[None, :, :] - z[:, None, :], 0.0) ** 2).sum(axis=2))
    return float(gaps.min(axis=1).mean())
