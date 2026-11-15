"""Surrogate-assisted many-objective search on continuous problems (SA-LHFiD and its ablations).

Same loop as ``sa_moea.run_sa`` on [0, 1]^D: start from an initial design, then in each round
1. scale the observed objectives by the observed minimum and maximum (pre-registered online rule; no
   knowledge of the true front),
2. fit one surrogate per objective,
3. run the inner MOEA (``continuous.run`` with a ``moea`` selector) on the surrogate, starting from the
   best observed points,
4. evaluate ``batch`` novel candidates from the inner final population: predicted-nondominated first,
   chosen by ``pick`` ("spread": best per objective then farthest point, as in C2; "hssp": greedy
   hypervolume subset selection in predicted scaled space, reference 1.1).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import continuous, final_selection, moea
from .sa_moea import spread_pick


class Surrogate:
    """Per-objective regressors on x in [0, 1]^D: "gp" (sklearn Matern-5/2 ARD + white noise),
    "gp_ddmto" (DDMTOLab's Kriging, the same model its K-RVEA/KTA2 use), "et" (ExtraTrees, as in C2)."""

    def __init__(self, family: str, seed: int) -> None:
        self.family, self.seed = family, seed

    def fit(self, x: np.ndarray, f: np.ndarray) -> "Surrogate":
        self.m = f.shape[1]
        if self.family == "gp_ddmto":
            import torch
            from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build

            torch.manual_seed(self.seed)
            self.models = mo_gp_build(x, f, torch.float)
        elif self.family == "gp":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

            d = x.shape[1]
            self.models = []
            for j in range(self.m):
                kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(d), (1e-2, 1e2), nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e0))
                self.models.append(GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1,
                                                            random_state=self.seed + j).fit(x, f[:, j]))
        elif self.family == "et":
            from sklearn.ensemble import ExtraTreesRegressor

            self.models = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, random_state=self.seed, n_jobs=1).fit(x, f)
        else:
            raise ValueError(self.family)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.family == "gp_ddmto":
            import torch
            from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_predict

            return np.asarray(mo_gp_predict(self.models, x, torch.float, mse=False))
        if self.family == "gp":
            return np.column_stack([model.predict(x) for model in self.models])
        return self.models.predict(x)


def novel(x: np.ndarray, seen: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Rows of x farther than ``tolerance`` (Chebyshev) from every row of ``seen`` and from each other."""
    keep: list[int] = []
    for i, row in enumerate(x):
        if len(seen) and (np.abs(seen - row).max(axis=1) < tolerance).any():
            continue
        if keep and (np.abs(x[keep] - row).max(axis=1) < tolerance).any():
            continue
        keep.append(i)
    return np.array(keep, dtype=int)


def run_sa(inner: str, observe: Callable[[np.ndarray], np.ndarray], initial_x: np.ndarray, budget: int, seed: int,
           vectors: np.ndarray, batch: int = 20, inner_generations: int = 30, surrogate: str = "gp_ddmto",
           pick: str = "spread", eta_c: float = 20.0, eta_m: float = 20.0) -> dict:
    rng = np.random.default_rng(seed)
    x = np.asarray(initial_x, dtype=float)[:budget]
    f = np.asarray(observe(x), dtype=float)
    dim, population = x.shape[1], len(vectors)
    rounds = 0
    while len(x) < budget:
        rounds += 1
        lo, hi = f.min(axis=0), f.max(axis=0)
        scaled = (f - lo) / np.maximum(hi - lo, 1e-12)
        model = Surrogate(surrogate, seed * 1000 + rounds).fit(x, scaled)
        order = [i for front in moea.nondominated_sort(scaled) for i in front][:population]
        start = x[order]
        if len(start) < population:
            start = np.vstack([start, rng.random((population - len(start), dim))])
        result = continuous.run(inner, model.predict, dim, population, inner_generations, vectors,
                                seed=seed * 1000 + rounds, eta_c=eta_c, eta_m=eta_m, initial=start)
        take = min(batch, budget - len(x))
        pool_x = result["x"][novel(result["x"], x)]
        new = np.empty((0, dim))
        if len(pool_x):
            predicted = model.predict(pool_x)
            front = moea.nondominated_sort(predicted)[0]
            if pick == "hssp":
                chosen = [front[i] for i in final_selection.greedy_hssp(predicted[front], np.full(predicted.shape[1], 1.1), take)]
                rest = [i for i in front if i not in chosen]
                chosen += [rest[i] for i in spread_pick(predicted[rest], take - len(chosen))] if rest and len(chosen) < take else []
            else:
                chosen = [front[i] for i in spread_pick(predicted[front], take)]
            if len(chosen) < take:
                others = [i for i in range(len(pool_x)) if i not in set(chosen)]
                if others:
                    chosen += [others[i] for i in spread_pick(predicted[others], take - len(chosen))]
            new = pool_x[chosen[:take]]
        if len(new) < take:  # degenerate case: top up with uniform random points
            new = np.vstack([new, rng.random((take - len(new), dim))])
        x = np.vstack([x, new])
        f = np.vstack([f, np.asarray(observe(new), dtype=float)])
    return {"x": x, "f": f, "rounds": rounds}
