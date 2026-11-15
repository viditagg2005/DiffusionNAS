"""Continuous test problems and a continuous-variable MOEA loop for benchmarking the selection schemes.

The selectors (NSGA-II/III, LHFiD, RVEA, MOEA/D-style) and LHFiD's stabilization-timed nadir estimation
are imported unchanged from ``moea``; only variation differs from ``moea.run`` (SBX crossover and
polynomial mutation on [0, 1]^D instead of the discrete diffusion-policy operators). ``moea.run`` is not
modified, so every earlier result that used it is unaffected.

Problems follow the standard DTLZ definitions (Deb, Thiele, Laumanns and Zitzler 2005), identical to the
PlatEMO/DDMTOLab implementations: DTLZ1 and DTLZ3 multimodal, DTLZ2 spherical, DTLZ4 biased (alpha 100),
DTLZ7 disconnected. Objectives are minimized.
"""
from __future__ import annotations

import random
from typing import Callable

import numpy as np

from . import moea

# ---- problems -----------------------------------------------------------------


def _sphere(x: np.ndarray, g: np.ndarray, m: int) -> np.ndarray:
    f = np.ones((x.shape[0], m)) * (1 + g)[:, None]
    for i in range(m):
        for j in range(m - i - 1):
            f[:, i] *= np.cos(x[:, j] * np.pi / 2)
        if i > 0:
            f[:, i] *= np.sin(x[:, m - i - 1] * np.pi / 2)
    return f


def dtlz(name: str, x: np.ndarray, m: int) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=float))
    xm = x[:, m - 1:]
    d = x.shape[1]
    if name in ("dtlz1", "dtlz3"):
        g = 100 * (d - m + 1 + np.sum((xm - 0.5) ** 2 - np.cos(20 * np.pi * (xm - 0.5)), axis=1))
    if name == "dtlz1":
        f = np.ones((x.shape[0], m)) * (0.5 * (1 + g))[:, None]
        for i in range(m):
            for j in range(m - i - 1):
                f[:, i] *= x[:, j]
            if i > 0:
                f[:, i] *= 1 - x[:, m - i - 1]
        return f
    if name == "dtlz2":
        return _sphere(x, np.sum((xm - 0.5) ** 2, axis=1), m)
    if name == "dtlz3":
        return _sphere(x, g, m)
    if name == "dtlz4":
        y = x.copy()
        y[:, : m - 1] = x[:, : m - 1] ** 100
        return _sphere(y, np.sum((xm - 0.5) ** 2, axis=1), m)
    if name == "dtlz7":
        g = 1 + 9 * np.mean(xm, axis=1)
        f = np.zeros((x.shape[0], m))
        f[:, : m - 1] = x[:, : m - 1]
        h = m - np.sum(f[:, : m - 1] / (1 + g[:, None]) * (1 + np.sin(3 * np.pi * f[:, : m - 1])), axis=1)
        f[:, m - 1] = (1 + g) * h
        return f
    raise ValueError(name)


def reference_front(name: str, m: int, points: int = 10_000, seed: int = 0) -> np.ndarray:
    """Dense Pareto-front samples (for IGD+ and ideal/nadir)."""
    if name in ("dtlz1", "dtlz2", "dtlz3", "dtlz4"):
        p = 1
        while len(moea.das_dennis(m, p + 1)) <= points:
            p += 1
        w = moea.das_dennis(m, p)
        if name == "dtlz1":
            return 0.5 * w
        return w / np.linalg.norm(w, axis=1, keepdims=True)
    if name == "dtlz7":
        rng = np.random.default_rng(seed)
        x = rng.random((points * 20, m - 1))
        f = np.zeros((len(x), m))
        f[:, : m - 1] = x
        f[:, m - 1] = 2 * (m - np.sum(x / 2 * (1 + np.sin(3 * np.pi * x)), axis=1))
        keep = moea.nondominated_sort(f[: min(len(f), 40_000)])[0]
        return f[keep]
    raise ValueError(name)


def front_bounds(front: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return front.min(axis=0), front.max(axis=0)


def reference_vectors(m: int) -> np.ndarray:
    """Das-Dennis sets used throughout (LHFiD paper Table IV where it overlaps; two layers for M >= 8)."""
    table = {2: (99, None), 3: (13, None), 4: (7, None), 5: (6, None), 6: (4, None), 8: (3, 2), 10: (3, 2), 15: (2, 1)}
    p, inner = table[m]
    return moea.das_dennis(m, p) if inner is None else moea.two_layer(m, p, inner)


# ---- variation ------------------------------------------------------------------


def sbx(p1: np.ndarray, p2: np.ndarray, eta: float, rng: np.random.Generator, rate: float = 0.9):
    """Simulated binary crossover on [0, 1] (bounded, Deb and Agrawal 1995; as in PlatEMO)."""
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > rate:
        return c1, c2
    u = rng.random(len(p1))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (2 - 2 * u) ** (-1 / (eta + 1)))
    beta *= np.where(rng.random(len(p1)) < 0.5, 1.0, -1.0)  # random sign
    beta[rng.random(len(p1)) < 0.5] = 1.0  # each variable crossed with probability 0.5
    mean, half = (p1 + p2) / 2, (p1 - p2) / 2
    c1, c2 = mean + beta * half, mean - beta * half
    return np.clip(c1, 0.0, 1.0), np.clip(c2, 0.0, 1.0)


def polynomial_mutation(x: np.ndarray, eta: float, rng: np.random.Generator, rate: float | None = None) -> np.ndarray:
    """Polynomial mutation on [0, 1] (Deb and Goyal 1996, bounded form as in PlatEMO), rate 1/D by default."""
    y = x.copy()
    rate = 1.0 / len(x) if rate is None else rate
    mask = rng.random(len(x)) < rate
    u = rng.random(len(x))
    d1, d2 = y, 1 - y
    low = u <= 0.5
    delta = np.where(low,
                     (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** (1 / (eta + 1)) - 1,
                     1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** (1 / (eta + 1)))
    y[mask] = y[mask] + delta[mask]
    return np.clip(y, 0.0, 1.0)


def offspring(parents: np.ndarray, count: int, rng: np.random.Generator, eta_c: float, eta_m: float,
              pick: Callable[[], int] | None = None) -> np.ndarray:
    pick = pick or (lambda: int(rng.integers(len(parents))))
    children = []
    while len(children) < count:
        a, b = parents[pick()], parents[pick()]
        c1, c2 = sbx(a, b, eta_c, rng)
        children += [polynomial_mutation(c1, eta_m, rng), polynomial_mutation(c2, eta_m, rng)]
    return np.array(children[:count])


# ---- evolutionary loop --------------------------------------------------------------


def run(algorithm: str, evaluate: Callable[[np.ndarray], np.ndarray], dim: int, population: int,
        generations: int, vectors: np.ndarray, seed: int, eta_c: float = 20.0, eta_m: float = 20.0,
        initial: np.ndarray | None = None, tie_tolerance: np.ndarray | None = None) -> dict:
    """Generational MOEA with ``moea`` selectors; ``evaluate`` maps an (n, D) array to (n, M) objectives.

    Mating is random for NSGA-III, LHFiD (paper's Algorithm 1), RVEA and MOEA/D-style, and binary
    tournament on (rank, crowding) for NSGA-II. LHFiD re-estimates its nadir point at each generation in
    which stabilization tracking detects mild stability, exactly as in ``moea.run``.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    x = rng.random((population, dim)) if initial is None else np.asarray(initial, dtype=float)
    f = evaluate(x)
    select = moea.SELECTORS[algorithm]
    state: dict = {"vectors": vectors, "preference_mode": "equal" if algorithm == "lhfid_equal" else "inverse_w",
                   "tie_tolerance": tie_tolerance}
    if algorithm == "nsga2":
        select(f, len(f), state, py_rng)
    stabilization = moea.Stabilization()
    nadir_generations: list[int] = []
    evaluations = len(f)
    for generation in range(1, generations + 1):
        pick = None
        if algorithm == "nsga2":
            def pick():
                a, b = py_rng.sample(range(len(x)), 2)
                key = lambda i: (state["rank"][i], -state["crowd"][i])
                return a if key(a) <= key(b) else b
        children = offspring(x, population, rng, eta_c, eta_m, pick)
        child_f = evaluate(children)
        evaluations += len(children)
        union_x, union_f = np.vstack([x, children]), np.vstack([f, child_f])
        state["progress"] = generation / generations
        keep = select(union_f, population, state, py_rng)
        if algorithm.startswith("lhfid"):
            scale = lambda v: v - state["ideal"] if state.get("nadir") is None else (v - state["ideal"]) / np.maximum(state["nadir"] - state["ideal"], 1e-12)
            if stabilization.update(scale(f), scale(union_f[keep]), vectors):
                fronts = moea.nondominated_sort(union_f[keep])
                state["nadir"] = moea.nsga3_normalize(union_f[keep], state["ideal"], fronts[0])
                nadir_generations.append(generation)
        x, f = union_x[keep], union_f[keep]
        if algorithm == "nsga2":
            select(f, len(f), state, py_rng)
    return {"x": x, "f": f, "evaluations": evaluations, "nadir_generations": nadir_generations}
