#!/usr/bin/env python3
"""SAEA benchmark on noisy DTLZ problems (protocol: configs/nas/saea_benchmark.json). CPU only; .venv-saea.

Every algorithm sees the same problem object: a logger that records each real evaluation (noise-free and
observed values) and adds Gaussian noise to the pre-registered objectives. Indicators are computed from
the log, identically for all algorithms:
  igd_plus        IGD+ of the nondominated noise-free values of every evaluated point (normalized by the
                  true front's ideal/nadir);
  hv5_deliverable greedy hypervolume subset selection of 5 points on the *observed* (noisy) values, then
                  their hypervolume on the noise-free values (normalized, reference 1.1);
  hv5_oracle      the same 5-point selection on noise-free values (best 5 the archive contains).
Resumable: finished (setting, problem, M, noise, algorithm, seed) rows are skipped.

  python scripts/saea_benchmark.py --setting main --workers 64 [--shard 0 --num-shards 4] [--only k_rvea,sa_lhfid]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas import continuous, final_selection, moea  # noqa: E402

PROTOCOL = json.loads((ROOT / "configs/nas/saea_benchmark.json").read_text())
DDMTO = {"k_rvea": ("ddmtolab.Algorithms.STMO.K_RVEA", "K_RVEA"), "parego": ("ddmtolab.Algorithms.STMO.ParEGO", "ParEGO"),
         "csea": ("ddmtolab.Algorithms.STMO.CSEA", "CSEA"), "moead_ego": ("ddmtolab.Algorithms.STMO.MOEA_D_EGO", "MOEA_D_EGO"),
         "kta2": ("ddmtolab.Algorithms.STMO.KTA2", "KTA2")}
FRONTS: dict = {}


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class Logger:
    """The expensive function: records every real evaluation; noisy objectives get N(0, (sigma * range_j)^2)."""

    def __init__(self, problem: str, m: int, sigma: float, noisy: list[int], span: np.ndarray, seed: int) -> None:
        self.problem, self.m = problem, m
        self.scale = np.zeros(m)
        self.scale[noisy] = sigma * np.asarray(span)[noisy]
        self.rng = np.random.default_rng(seed)
        self.armed = False
        self.x, self.true, self.observed = [], [], []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        true = continuous.dtlz(self.problem, x, self.m)
        if not self.armed:  # DDMTOLab probes the function once when a task is registered; not an evaluation
            return true
        observed = true + self.rng.normal(size=true.shape) * self.scale
        self.x.append(x.copy())
        self.true.append(true)
        self.observed.append(observed)
        return observed

    def arrays(self):
        return np.vstack(self.x), np.vstack(self.true), np.vstack(self.observed)


def front(problem: str, m: int):
    key = (problem, m)
    if key not in FRONTS:
        pf = continuous.reference_front(problem, m)
        FRONTS[key] = (pf, *continuous.front_bounds(pf))
    return FRONTS[key]


def run_task(task: dict) -> dict:
    import warnings

    warnings.filterwarnings("ignore")  # botorch/gpytorch emit per-fit input-scaling warnings
    import torch

    torch.set_num_threads(1)
    s = PROTOCOL["settings"][task["setting"]]
    problem, m, sigma, algorithm, seed = task["problem"], task["m"], task["noise"], task["algorithm"], task["seed"]
    dim, budget, n_init = s["dim"], s["budget"], s["initial"]
    pf, ideal, nadir = front(problem, m)
    noisy = list(range(int(np.ceil(m / 2))))
    logger = Logger(problem, m, sigma, noisy, nadir - ideal, stable_seed("noise", task["setting"], problem, m, sigma, seed))
    run_seed = stable_seed("run", task["setting"], problem, m, sigma, algorithm, seed)
    np.random.seed(run_seed % 2**32)
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    vectors = continuous.reference_vectors(m)
    eta_m = PROTOCOL["eta_m"].get(problem, PROTOCOL["eta_m"]["default"])
    start = time.time()
    if algorithm in DDMTO:
        import importlib

        from ddmtolab.Methods.mtop import MTOP

        mtop = MTOP()
        mtop.add_task(objective_func=logger, dim=dim, lower_bound=np.zeros(dim), upper_bound=np.ones(dim))
        logger.armed = True
        module, name = DDMTO[algorithm]
        cls = getattr(importlib.import_module(module), name)
        cls(mtop, n_initial=n_init, max_nfes=budget, save_data=False, **PROTOCOL["ddmtolab_kwargs"].get(algorithm, {})).optimize()
    else:
        from scipy.stats import qmc

        logger.armed = True
        initial = qmc.LatinHypercube(d=dim, seed=run_seed % 2**32).random(n_init if algorithm != "lhs" else budget)
        if algorithm == "lhs":
            logger(initial)
        else:
            arm = PROTOCOL["our_arms"][algorithm]
            from diffusionnas.sa_continuous import run_sa

            run_sa(arm["inner"], logger, initial, budget, run_seed % 2**31, vectors, batch=arm.get("batch", 20),
                   inner_generations=arm.get("inner_generations", 30), surrogate=arm.get("surrogate", "gp_ddmto"),
                   pick=arm.get("pick", "spread"), eta_m=eta_m)
    runtime = time.time() - start
    x, true, observed = logger.arrays()
    evaluations = len(x)
    x, true, observed = x[:budget], true[:budget], observed[:budget]  # count only the budget
    scale = np.maximum(nadir - ideal, 1e-12)
    true_n, obs_n = (true - ideal) / scale, (observed - ideal) / scale
    ref = np.full(m, 1.1)
    chosen = final_selection.greedy_hssp(obs_n, ref, 5)
    oracle = final_selection.greedy_hssp(true_n, ref, 5)
    out = ROOT / "artifacts" / "saea_benchmark" / task["setting"] / "runs"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{problem}_m{m}_n{sigma}_{algorithm}_s{seed}"
    np.savez_compressed(out / f"{tag}.npz", x=x.astype(np.float32), true=true, observed=observed)
    return {**task, "evaluations_logged": evaluations, "runtime_s": runtime,
            "igd_plus": moea.igd_plus(true, pf, ideal, nadir),
            "hv5_deliverable": final_selection.hypervolume(true_n[chosen], ref),
            "hv5_oracle": final_selection.hypervolume(true_n[oracle], ref)}


class RunTimeout(Exception):
    pass


def run_task_guarded(task: dict) -> dict:
    """run_task with the protocol's wall-clock limit; a run that exceeds it is recorded, never dropped."""
    import signal

    def alarm(signum, frame):
        raise RunTimeout()

    limit = int(PROTOCOL["run_time_limit_s"])
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(limit)
    try:
        return {**run_task(task), "status": "ok"}
    except RunTimeout:
        return {**task, "status": "timeout", "time_limit_s": limit}
    except Exception as error:  # recorded so a crash is visible in the results, not silently retried
        return {**task, "status": "error", "error": repr(error)[:500]}
    finally:
        signal.alarm(0)


def tasks_for(setting: str, only: list[str] | None) -> list[dict]:
    s = PROTOCOL["settings"][setting]
    algorithms = only or PROTOCOL["algorithms"]
    return [{"setting": setting, "problem": p, "m": m, "noise": n, "algorithm": a, "seed": seed}
            for p, m, n, a, seed in itertools.product(s["problems"], s["objectives"], s["noise"], algorithms, range(s["seeds"]))]


def key(row: dict) -> tuple:
    return (row["setting"], row["problem"], row["m"], row["noise"], row["algorithm"], row["seed"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", default="main", choices=sorted(PROTOCOL["settings"]))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--only", help="comma-separated algorithms (default: all in the protocol)")
    parser.add_argument("--limit", type=int, help="run at most this many tasks (pilot)")
    parser.add_argument("--filter", help="problem:m, e.g. dtlz2:10 (pilot)")
    args = parser.parse_args()
    tasks = tasks_for(args.setting, args.only.split(",") if args.only else None)
    tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    if args.filter:
        p, m = args.filter.split(":")
        tasks = [t for t in tasks if t["problem"] == p and t["m"] == int(m)]
    results = ROOT / "artifacts" / "saea_benchmark" / args.setting / f"results_shard{args.shard}.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    for path in results.parent.glob("results_shard*.jsonl"):
        done |= {key(json.loads(line)) for line in path.read_text().splitlines() if line.strip()}
    tasks = [t for t in tasks if key(t) not in done]
    if args.limit:
        tasks = tasks[: args.limit]
    # longest first: many objectives and model-heavy algorithms at the front of the queue
    tasks.sort(key=lambda t: (-t["m"], t["algorithm"] not in ("parego", "kta2", "moead_ego")))
    print(f"{len(tasks)} tasks on {args.workers} workers -> {results}", flush=True)
    start = time.time()
    with get_context("spawn").Pool(args.workers, maxtasksperchild=20) as pool, results.open("a") as sink:
        for i, row in enumerate(pool.imap_unordered(run_task_guarded, tasks), 1):
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  {(time.time() - start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
