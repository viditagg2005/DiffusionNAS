"""Surrogate models of the six v3 objectives, fitted on real calibration evaluations.

Used only for the many-objective optimizer benchmark (``scripts/surrogate_benchmark.py``),
where each optimizer gets thousands of cheap evaluations instead of 150 real ones.
Every claim drawn from it is conditional on surrogate accuracy, which is
reported (cross-validated and leave-one-search-seed-out).
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from . import nas_space
from .flexiffusion import NULL, PARTIAL, expand_schedule

# All objectives as minimization: alignment and fidelity are negated.
OBJECTIVES = (
    ("clip_score", "max"),
    ("clip_image_fidelity", "max"),
    ("latency_p50_ms", "min"),
    ("gflops_per_image", "min"),
    ("peak_allocated_mb", "min"),
    ("unet_weight_mb", "min"),
)
SEARCH4 = OBJECTIVES[:3] + OBJECTIVES[5:]


def features(policy: dict[str, Any]) -> list[float]:
    method = policy["method"]
    steps = policy["num_inference_steps"]
    full = partial = null = 0
    branch_sum = 0.0
    if method == "baseline":
        full = steps
    elif method == "deepcache":
        interval, branch = policy["deepcache_interval"], policy["deepcache_branch_id"]
        full = sum(1 for i in range(steps) if i % interval == 0)
        partial = steps - full
        branch_sum = partial * branch
    else:
        flex = policy["flexiffusion"]
        for action in expand_schedule(flex["segments"], flex["segment_length"]):
            if action.kind == NULL:
                null += 1
            elif action.kind == PARTIAL:
                partial += 1
                branch_sum += action.branch
            else:
                full += 1
    architecture = policy["architecture"]
    removed = set(architecture["removed_units"])
    return [
        float(method == "baseline"), float(method == "deepcache"), float(method == "flexiffusion"),
        float(policy["scheduler"] == "dpm_solver++"), float(steps), float(full), float(partial), float(null),
        branch_sum / partial if partial else 0.0, float(full + partial), float(policy["guidance_scale"]),
        float(architecture["weight_precision"] == "int8"), float(len(removed)),
        *[float(unit in removed) for unit in nas_space.UNITS],
    ]


def load_archive(nas_root: Path) -> list[dict[str, Any]]:
    """All v3 candidates with latency normalized by their job's reference (as in the finalist lock)."""
    references = {}
    for seed_dir in sorted(nas_root.glob("seed*")):
        run = next((seed_dir / "reference" / "runs").iterdir())
        references[seed_dir.name] = json.loads((run / "summary.json").read_text())["latency_ms"]["p50"]
    pooled = statistics.median(references.values())
    rows = []
    for seed_dir in sorted(nas_root.glob("seed*")):
        for method in ("random", "nsga2"):
            results = json.loads((seed_dir / method / "search_results.json").read_text())
            for candidate in results["candidates"]:
                objectives = dict(candidate["objectives"])
                objectives["latency_p50_ms"] *= pooled / references[seed_dir.name]
                rows.append({"policy": candidate["policy"], "objectives": objectives,
                             "group": seed_dir.name, "optimizer": method})
    return rows


class Surrogate:
    """One regressor per objective; family and bootstrap resampling make independent surrogates."""

    def __init__(self, objectives=OBJECTIVES, seed: int = 0, family: str = "extratrees", bootstrap: bool = False,
                 n_jobs: int = -1) -> None:
        self.objectives = tuple(objectives)
        self.seed, self.family, self.bootstrap, self.n_jobs = seed, family, bootstrap, n_jobs
        self.models: list[Any] = []

    def _model(self):
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

        if self.family == "randomforest":
            return RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=self.seed, n_jobs=self.n_jobs)
        if self.family == "hgb":
            return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=self.seed)
        return ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=self.seed, n_jobs=self.n_jobs)

    def fit(self, rows: list[dict[str, Any]]) -> "Surrogate":
        if self.bootstrap:
            rng = np.random.default_rng(self.seed)
            rows = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        x = np.array([features(r["policy"]) for r in rows])
        self.models = []
        for name, _ in self.objectives:
            y = np.array([r["objectives"][name] for r in rows])
            self.models.append(self._model().fit(x, y))
        return self

    def predict(self, policies: list[dict[str, Any]]) -> np.ndarray:
        """Objective matrix in minimization form (maximized objectives negated)."""
        x = np.array([features(p) for p in policies])
        columns = [m.predict(x) for m in self.models]
        signs = [(-1.0 if direction == "max" else 1.0) for _, direction in self.objectives]
        return np.stack([s * c for s, c in zip(signs, columns)], axis=1)


def cross_validate(rows: list[dict[str, Any]], objectives=OBJECTIVES, folds: int = 5, seed: int = 0,
                   family: str = "extratrees", region=None) -> dict[str, Any]:
    """K-fold and leave-one-search-seed-out accuracy for each objective.

    ``region`` optionally restricts the scored test points (e.g. to the hypervolume
    reference box), since accuracy there decides the indicator.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score

    def score(train, test):
        if region is not None:
            test = [r for r in test if region(r)]
        model = Surrogate(objectives, seed, family).fit(train)
        predicted = model.predict([r["policy"] for r in test])
        out = {}
        for j, (name, direction) in enumerate(objectives):
            truth = np.array([r["objectives"][name] for r in test])
            guess = predicted[:, j] * (-1.0 if direction == "max" else 1.0)
            out[name] = (truth, guess)
        return out

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    pooled: dict[str, tuple[list, list]] = {name: ([], []) for name, _ in objectives}
    for k in range(folds):
        test_idx = set(order[k::folds].tolist())
        result = score([r for i, r in enumerate(rows) if i not in test_idx], [rows[i] for i in sorted(test_idx)])
        for name, (truth, guess) in result.items():
            pooled[name][0].extend(truth)
            pooled[name][1].extend(guess)
    groups = sorted({r["group"] for r in rows})
    by_group: dict[str, tuple[list, list]] = {name: ([], []) for name, _ in objectives}
    for group in groups:
        result = score([r for r in rows if r["group"] != group], [r for r in rows if r["group"] == group])
        for name, (truth, guess) in result.items():
            by_group[name][0].extend(truth)
            by_group[name][1].extend(guess)
    report = {"scored_points": len(pooled[objectives[0][0]][0])}
    for name, _ in objectives:
        report[name] = {
            "kfold_r2": float(r2_score(*pooled[name])),
            "kfold_spearman": float(spearmanr(*pooled[name]).statistic),
            "leave_seed_out_r2": float(r2_score(*by_group[name])),
            "leave_seed_out_spearman": float(spearmanr(*by_group[name]).statistic),
        }
    return report
