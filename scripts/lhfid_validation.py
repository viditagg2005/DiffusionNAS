#!/usr/bin/env python3
"""Validate our LHFiD / NSGA-III reimplementations against the LHFiD paper (configs/nas/lhfid_validation.json).

  python scripts/lhfid_validation.py --workers 64
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
from multiprocessing import get_context
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas import continuous  # noqa: E402

PROTOCOL = json.loads((ROOT / "configs/nas/lhfid_validation.json").read_text())
P_GAPS = {3: 13, 5: 6, 10: 3}
SEEDS = 31


def hypervolume(f: np.ndarray, ref: np.ndarray, samples: int = 1_000_000, seed: int = 12345) -> float:
    """Exact (moocore) for M <= 5; for M >= 6 Monte Carlo over the box [0, ref] with fixed samples (PlatEMO's
    convention for many objectives), counting samples not dominated by f."""
    import moocore

    inside = f[(f < ref).all(axis=1)]
    if not len(inside):
        return 0.0
    if len(ref) <= 5:
        return float(moocore.hypervolume(inside, ref=ref))
    rng = np.random.default_rng(seed)
    free = 0
    for start in range(0, samples, 10_000):
        u = rng.random((min(10_000, samples - start), len(ref))) * ref
        dominated = np.zeros(len(u), dtype=bool)
        for block in range(0, len(inside), 64):
            dominated |= (inside[None, block:block + 64, :] <= u[:, None, :]).all(axis=2).any(axis=1)
        free += int((~dominated).sum())
    return float(np.prod(ref) * (1 - free / samples))


def run_task(task: tuple) -> dict:
    problem, m, algorithm, seed = task
    generations = PROTOCOL["targets"][problem][str(m)]["generations"]
    vectors = continuous.reference_vectors(m)
    dim = m - 1 + 20
    start = time.time()
    result = continuous.run(algorithm, lambda x: continuous.dtlz(problem, x, m), dim, len(vectors), generations, vectors,
                            seed=10_000 * m + seed, eta_c=20.0, eta_m=20.0)
    ref = np.full(m, 1 + 1 / P_GAPS[m])
    f = result["f"]  # true ideal 0 and nadir 1 for DTLZ2/4: already normalized
    hv = hypervolume(f, ref)
    return {"problem": problem, "m": m, "algorithm": algorithm, "seed": seed, "hv": hv, "generations": generations,
            "population": len(vectors), "first_nadir_generation": (result["nadir_generations"] or [None])[0],
            "runtime_s": time.time() - start}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    out = ROOT / "artifacts" / "lhfid_validation"
    out.mkdir(parents=True, exist_ok=True)
    sink_path = out / "runs.jsonl"
    done = set()
    if sink_path.exists():
        done = {(r["problem"], r["m"], r["algorithm"], r["seed"]) for r in map(json.loads, sink_path.read_text().splitlines())}
    tasks = [(p, int(m), a, s) for p, block in PROTOCOL["targets"].items() for m in block for a in ("lhfid", "nsga3")
             for s in range(SEEDS) if (p, int(m), a, s) not in done]
    tasks.sort(key=lambda t: -t[1])
    print(f"{len(tasks)} runs on {args.workers} workers", flush=True)
    with get_context("spawn").Pool(args.workers) as pool, sink_path.open("a") as sink:
        for i, row in enumerate(pool.imap_unordered(run_task, tasks), 1):
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
    rows = [json.loads(line) for line in sink_path.read_text().splitlines()]
    summary = {}
    for problem, block in PROTOCOL["targets"].items():
        for m, target in block.items():
            ref = (1 + 1 / P_GAPS[int(m)]) ** int(m)
            cell = {}
            for algorithm in ("nsga3", "lhfid"):
                values = [r["hv"] for r in rows if r["problem"] == problem and r["m"] == int(m) and r["algorithm"] == algorithm]
                ours = statistics.median(values)
                cell[algorithm] = {"ours_median": ours, "paper_median": target[algorithm], "n": len(values),
                                   "difference": ours - target[algorithm], "box_volume": ref}
            cell["ordering_reproduced"] = (cell["nsga3"]["ours_median"] > cell["lhfid"]["ours_median"]) == \
                                          (target["nsga3"] > target["lhfid"])
            summary[f"{problem}_m{m}"] = cell
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, cell in summary.items():
        print(f"{name:10s} " + "  ".join(f"{a}: ours {cell[a]['ours_median']:.6f} paper {cell[a]['paper_median']:.6f}"
                                         for a in ("nsga3", "lhfid")) + f"  ordering reproduced {cell['ordering_reproduced']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
