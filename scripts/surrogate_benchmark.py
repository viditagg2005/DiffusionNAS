#!/usr/bin/env python3
"""Run the pre-registered many-objective optimizer benchmark (configs/nas/surrogate_benchmark.json).

CPU only. Outputs (git-ignored) under artifacts/surrogate_benchmark/:
results.json (accuracy, per-run indicators, statistics) and the confirmation
point set; confirmation run configs go to configs/generated/confirmation/.
"""

from __future__ import annotations

import os

# One thread per worker process (set before numpy/sklearn load): many workers with
# multithreaded BLAS/OpenMP each otherwise oversubscribe the cores badly.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from copy import deepcopy
import json
from multiprocessing import get_context
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diffusionnas import moea  # noqa: E402
from diffusionnas.nas_space import signature  # noqa: E402
from diffusionnas.schema import validate_config  # noqa: E402
from diffusionnas.search_space import policy_to_config  # noqa: E402
from diffusionnas.surrogate import OBJECTIVES, Surrogate, cross_validate, load_archive  # noqa: E402
from prepare_experiment import load_prompts  # noqa: E402

PROTOCOL = json.loads((ROOT / "configs" / "nas" / "surrogate_benchmark.json").read_text())
DIRECTION = dict(OBJECTIVES)
STATE: dict = {}


def minimization(names, values: dict) -> np.ndarray:
    return np.array([-values[n] if DIRECTION[n] == "max" else values[n] for n in names])


def load_state(path: str) -> None:
    """Worker initializer: reuse the surrogates fitted once in the parent."""
    import pickle

    STATE.update(pickle.loads(Path(path).read_bytes()))


def setup(rows) -> None:
    for key, names in PROTOCOL["objective_sets"].items():
        objectives = [(n, DIRECTION[n]) for n in names]
        lo = np.min([minimization(names, r["objectives"]) for r in rows], axis=0)
        hi = minimization(names, PROTOCOL["reference_point"])
        surrogates = [Surrogate(objectives, s["seed"], s["family"], s["bootstrap"], n_jobs=1).fit(rows)
                      for s in PROTOCOL["surrogates"]]
        m = len(names)
        vectors = moea.das_dennis(4, 5) if m == 4 else moea.two_layer(6, 2, 2)
        STATE[key] = {"names": names, "lo": lo, "hi": hi, "surrogates": surrogates, "vectors": vectors}


def task(args):
    key, surrogate_index, seed, algorithm = args
    state = STATE[key]
    scale = lambda f: (f - state["lo"]) / (state["hi"] - state["lo"])
    evaluate = lambda policies: scale(state["surrogates"][surrogate_index].predict(policies))
    started = time.time()
    result = moea.run(algorithm, evaluate, PROTOCOL["budget"], PROTOCOL["population"][key], state["vectors"],
                      seed=1000 * surrogate_index + seed)
    archive = result["objectives"]
    ref = np.ones(archive.shape[1])
    hv = lambda f: float(__import__("moocore").hypervolume(f[(f < ref).all(axis=1)], ref=ref)) if (f < ref).all(axis=1).any() else 0.0
    record = {
        "objective_set": key, "surrogate": surrogate_index, "seed": seed, "algorithm": algorithm,
        "archive_hv": {str(c): hv(archive[:c]) for c in PROTOCOL["checkpoints"]},
        "final_population_hv": hv(result["population"]) if algorithm != "random" else None,
        "nadir_generations": result["nadir_generations"],
        "seconds": round(time.time() - started, 1),
    }
    if surrogate_index == 0 and key == "M6":
        inside = (archive < ref).all(axis=1)
        record["archive_inside"] = [{"policy": p, "scaled": list(map(float, f))}
                                    for p, f, ok in zip(result["policies"], archive, inside) if ok]
    return record


def holm(pvalues: dict) -> dict:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for rank, (pair, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[pair] = running
    return adjusted


def statistics_for(records, key, indicator):
    from scipy.stats import wilcoxon

    algorithms = PROTOCOL["algorithms"]
    cells = sorted({(r["surrogate"], r["seed"]) for r in records})
    value = {(r["algorithm"], r["surrogate"], r["seed"]): indicator(r) for r in records if r["objective_set"] == key}
    summary = {a: {"median": statistics.median(value[(a, s, d)] for s, d in cells),
                   "mean": statistics.fmean(value[(a, s, d)] for s, d in cells),
                   "sd": statistics.pstdev(value[(a, s, d)] for s, d in cells)} for a in algorithms
               if all((a, s, d) in value and value[(a, s, d)] is not None for s, d in cells)}
    pairs, raw = {}, {}
    names = list(summary)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diffs = [value[(a, s, d)] - value[(b, s, d)] for s, d in cells]
            signs = []
            for surrogate in sorted({s for s, _ in cells}):
                per_surrogate = [value[(a, s, d)] - value[(b, s, d)] for s, d in cells if s == surrogate]
                signs.append(np.sign(statistics.median(per_surrogate)))
            p = float(wilcoxon(diffs).pvalue) if any(diffs) else 1.0
            raw[f"{a} vs {b}"] = p
            pairs[f"{a} vs {b}"] = {"median_difference": statistics.median(diffs), "p": p,
                                    "consistent_sign_across_surrogates": bool(len(set(signs)) == 1 and signs[0] != 0)}
    for pair, p in holm(raw).items():
        pairs[pair]["p_holm"] = p
        pairs[pair]["claimable"] = bool(p < 0.05 and pairs[pair]["consistent_sign_across_surrogates"])
    return {"summary": summary, "pairs": pairs}


def pick_confirmation(points: list[dict], limit: int) -> list[dict]:
    """Best point on each objective, then farthest-point fill (as in lock_nas_finalists)."""
    f = np.array([p["scaled"] for p in points])
    chosen = []
    for j in range(f.shape[1]):
        i = int(np.argmin(f[:, j]))
        if i not in chosen:
            chosen.append(i)
    while len(chosen) < min(limit, len(points)):
        distances = [min(np.sum((f[i] - f[c]) ** 2) for c in chosen) if i not in chosen else -1 for i in range(len(f))]
        chosen.append(int(np.argmax(distances)))
    return [points[i] for i in chosen[:limit]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "surrogate_benchmark")
    parser.add_argument("--quick", action="store_true", help="pilot: 1 seed, surrogate 0 only, budget unchanged")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = load_archive(ROOT / "artifacts" / "nas")
    ref = PROTOCOL["reference_point"]
    in_box = lambda r: all((r["objectives"][n] >= ref[n]) if DIRECTION[n] == "max" else (r["objectives"][n] <= ref[n])
                           for n in PROTOCOL["objective_sets"]["M4"])
    accuracy = {}
    if not args.quick:
        for family in ("extratrees", "randomforest", "hgb"):
            accuracy[family] = {"all": cross_validate(rows, OBJECTIVES, family=family),
                                "reference_box": cross_validate(rows, OBJECTIVES, family=family, region=in_box)}
            print(family, {k: round(v["leave_seed_out_spearman"], 3) for k, v in accuracy[family]["reference_box"].items() if k != "scored_points"})
    setup(rows)
    import pickle

    state_path = args.output / "surrogates.pkl"
    state_path.write_bytes(pickle.dumps(STATE))
    surrogates = [0] if args.quick else range(len(PROTOCOL["surrogates"]))
    seeds = [0] if args.quick else range(PROTOCOL["seeds_per_surrogate"])
    tasks = [(key, s, seed, a) for key in PROTOCOL["objective_sets"] for s in surrogates for seed in seeds
             for a in PROTOCOL["algorithms"]]
    print(f"{len(tasks)} runs on {args.workers} workers")
    # Spawned workers fit their own surrogates: forking after OpenMP-backed models ran in the
    # parent (HistGradientBoosting) deadlocks the children.
    with get_context("spawn").Pool(args.workers, initializer=load_state, initargs=(str(state_path),)) as pool:
        records = []
        for i, record in enumerate(pool.imap_unordered(task, tasks), 1):
            records.append(record)
            if i % 25 == 0 or args.quick:
                print(f"  {i}/{len(tasks)} {record['objective_set']} {record['algorithm']:12s} "
                      f"HV@3360 {record['archive_hv']['3360']:.4f} nadir@{record['nadir_generations'][:3]} {record['seconds']}s", flush=True)

    results = {"protocol": PROTOCOL, "accuracy": accuracy, "statistics": {}, "runs": [
        {k: v for k, v in r.items() if k != "archive_inside"} for r in records]}
    for key in PROTOCOL["objective_sets"]:
        subset = [r for r in records if r["objective_set"] == key]
        results["statistics"][key] = {
            f"archive_hv@{c}": statistics_for(subset, key, lambda r, c=c: r["archive_hv"][str(c)])
            for c in PROTOCOL["checkpoints"]}
        results["statistics"][key]["final_population_hv"] = statistics_for(
            [r for r in subset if r["algorithm"] != "random"], key, lambda r: r["final_population_hv"])
    (args.output / ("pilot.json" if args.quick else "results.json")).write_text(json.dumps(results, indent=2) + "\n")

    if not args.quick:
        prompts = load_prompts(ROOT / "data" / "prompts" / "calibration.json")
        runtime = {"device": "cuda", "precision": "float16", "warmup_runs": 3, "repetitions": 1, "count_flops": True}
        out = ROOT / "configs" / "generated" / "confirmation"
        out.mkdir(parents=True, exist_ok=True)
        confirmation, seen = [], set()
        for algorithm in PROTOCOL["algorithms"]:
            runs = sorted([r for r in records if r["objective_set"] == "M6" and r["surrogate"] == 0 and r["algorithm"] == algorithm],
                          key=lambda r: r["archive_hv"]["3360"])
            median_run = runs[len(runs) // 2]
            for k, point in enumerate(pick_confirmation(median_run["archive_inside"], 5)):
                predicted = (np.array(point["scaled"]) * (STATE["M6"]["hi"] - STATE["M6"]["lo"]) + STATE["M6"]["lo"]).tolist()
                label = f"{algorithm}_{k}"
                key_sig = signature(point["policy"])
                confirmation.append({"label": label, "algorithm": algorithm, "policy": point["policy"],
                                     "predicted_minimization": dict(zip(STATE["M6"]["names"], predicted)),
                                     "duplicate": key_sig in seen})
                seen.add(key_sig)
                config = policy_to_config(point["policy"], prompts, [42, 123], dict(runtime),
                                          "artifacts/confirmation/runs", save_images=True)
                config["label"] = label
                validate_config(config)
                (out / f"{label}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        reference = json.loads((ROOT / "configs" / "generated" / "nas" / "reference.json").read_text())
        reference.update({"label": "reference", "output_root": "artifacts/confirmation/runs"})
        reference["runtime"]["warmup_runs"] = 10
        (out / "00_reference.json").write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")
        (args.output / "confirmation_points.json").write_text(json.dumps(confirmation, indent=2) + "\n")
        print(f"confirmation: {len(confirmation)} points -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
