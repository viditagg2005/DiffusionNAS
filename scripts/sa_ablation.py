#!/usr/bin/env python3
"""SA-LHFiD ablations on the simulated real problem (protocol: configs/nas/sa_ablation.json). CPU only.

Reuses scripts/sa_simulation.py: truths fitted on the 900 v3 evaluations (artifacts/nas; no SA-arm data),
calibrated observation noise and scaling. The noise stream depends only on (truth, seed), so arms are paired.

  python scripts/sa_ablation.py --workers 64
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import sa_simulation as sim  # noqa: E402
from diffusionnas import final_selection, moea  # noqa: E402
from diffusionnas.sa_moea import run_sa  # noqa: E402
from diffusionnas.stats import holm  # noqa: E402
from diffusionnas.surrogate import load_archive  # noqa: E402

PROTOCOL = json.loads((ROOT / "configs/nas/sa_ablation.json").read_text())


def task(args):
    truth_index, seed, arm = args
    truth = sim.STATE["truths"][truth_index]
    lo, hi = sim.STATE["lo"], sim.STATE["hi"]
    scale = lambda raw: (raw - lo) / (hi - lo)
    noise_rng = np.random.default_rng(10_000 * truth_index + seed)  # same stream as sa_simulation for this seed

    observed_log: list[np.ndarray] = []

    def observe(policies):
        raw = truth["model"].predict(policies)
        out = scale(raw + noise_rng.normal(size=raw.shape) * truth["sigma_raw"])
        observed_log.extend(out)
        return out

    settings = {**PROTOCOL["base"], **PROTOCOL["arms"][arm]}
    result = run_sa(settings["inner"], observe, PROTOCOL["budget"], seed, moea.das_dennis(4, 5),
                    initial=settings["initial"], batch=settings["batch"], inner_population=settings["inner_population"],
                    inner_generations=settings["inner_generations"], surrogate_family=settings["surrogate_family"],
                    pick=settings["pick"])
    true = scale(truth["model"].predict(result["policies"]))
    observed = np.array(observed_log)
    ref = np.ones(true.shape[1])
    chosen = final_selection.greedy_hssp(observed, ref, 5)
    return {"truth": truth_index, "seed": seed, "arm": arm, "true_hv": final_selection.hypervolume(true, ref),
            "deliverable_hv5": final_selection.hypervolume(true[chosen], ref),
            "inside_box": int(final_selection.in_box(true, ref).sum()), "evaluated": len(true)}


def main() -> int:
    from scipy.stats import wilcoxon

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seeds", type=int, default=PROTOCOL["seeds_per_truth"])
    args = parser.parse_args()
    out = ROOT / "artifacts" / "sa_ablation"
    out.mkdir(parents=True, exist_ok=True)
    state = sim.build(load_archive(ROOT / "artifacts/nas"))
    state_path = out / "truths.pkl"
    state_path.write_bytes(pickle.dumps(state))
    sink_path = out / "runs.jsonl"
    done = set()
    if sink_path.exists():
        done = {(r["truth"], r["seed"], r["arm"]) for r in map(json.loads, sink_path.read_text().splitlines())}
    arms = list(PROTOCOL["arms"])
    tasks = [(t, s, a) for t in range(len(state["truths"])) for s in range(args.seeds) for a in arms if (t, s, a) not in done]
    print(f"{len(tasks)} runs on {args.workers} workers", flush=True)
    with get_context("spawn").Pool(args.workers, initializer=sim.load_state, initargs=(str(state_path),)) as pool, \
            sink_path.open("a") as sink:
        for i, row in enumerate(pool.imap_unordered(task, tasks), 1):
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
    rows = [json.loads(line) for line in sink_path.read_text().splitlines()]
    summary = {}
    for t in range(len(state["truths"])):
        family = state["truths"][t]["family"]
        by = {a: {r["seed"]: r for r in rows if r["truth"] == t and r["arm"] == a} for a in arms}
        block = {}
        for indicator in ("true_hv", "deliverable_hv5"):
            raw, entries = {}, {}
            for arm in arms[1:]:
                seeds = sorted(set(by["base"]) & set(by[arm]))
                a = np.array([by[arm][s][indicator] for s in seeds])
                b = np.array([by["base"][s][indicator] for s in seeds])
                p = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
                raw[arm] = p
                entries[arm] = {"median_arm": float(np.median(a)), "median_base": float(np.median(b)),
                                "median_paired_difference": float(np.median(a - b)), "p": p}
            for arm, p in holm(raw).items():
                entries[arm]["p_holm"] = p
            block[indicator] = entries
        summary[family] = block
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for family, block in summary.items():
        print(f"\ntruth {family}")
        for indicator, entries in block.items():
            for arm, e in entries.items():
                print(f"  {indicator:16s} {arm:22s} d {e['median_paired_difference']:+.4f}  pHolm {e['p_holm']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
