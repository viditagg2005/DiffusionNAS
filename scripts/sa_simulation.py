#!/usr/bin/env python3
"""Preliminary simulation of surrogate-assisted LHFiD (protocol: configs/nas/sa_simulation.json). CPU only."""

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

from diffusionnas import moea, nas_space  # noqa: E402
from diffusionnas.sa_moea import run_sa  # noqa: E402
from diffusionnas.stats import holm  # noqa: E402
from diffusionnas.surrogate import OBJECTIVES, Surrogate, load_archive  # noqa: E402

PROTOCOL = json.loads((ROOT / "configs/nas/sa_simulation.json").read_text())
NAMES = PROTOCOL["objectives"]
DIRECTION = dict(OBJECTIVES)
RELIABILITY = {"clip_score": 0.50, "clip_image_fidelity": 0.90}
STATE: dict = {}


def minimization(values: dict) -> np.ndarray:
    return np.array([-values[n] if DIRECTION[n] == "max" else values[n] for n in NAMES])


def build(rows) -> dict:
    lo = np.min([minimization(r["objectives"]) for r in rows], axis=0)
    hi = minimization(PROTOCOL["reference_point"])
    truths = []
    in_box = [r for r in rows if (minimization(r["objectives"]) < hi).all()]
    for spec in PROTOCOL["truths"]:
        model = Surrogate([(n, DIRECTION[n]) for n in NAMES], spec["seed"], spec["family"], n_jobs=1).fit(rows)
        truth_box = model.predict([r["policy"] for r in in_box])  # minimization form, raw units
        sigma = np.zeros(len(NAMES))
        for j, name in enumerate(NAMES):
            if name in RELIABILITY:
                # reliability r = var_true / (var_true + sigma^2)  =>  sigma^2 = var_true (1/r - 1)
                sigma[j] = float(np.sqrt(np.var(truth_box[:, j]) * (1 / RELIABILITY[name] - 1)))
        truths.append({"model": model, "sigma_raw": sigma, "family": spec["family"]})
    return {"lo": lo, "hi": hi, "truths": truths, "in_box": len(in_box)}


def load_state(path: str) -> None:
    STATE.update(pickle.loads(Path(path).read_bytes()))


def task(args):
    truth_index, seed, arm = args
    truth = STATE["truths"][truth_index]
    lo, hi = STATE["lo"], STATE["hi"]
    scale = lambda raw: (raw - lo) / (hi - lo)
    noise_rng = np.random.default_rng(10_000 * truth_index + seed)

    def observe(policies):
        raw = truth["model"].predict(policies)
        return scale(raw + noise_rng.normal(size=raw.shape) * truth["sigma_raw"])

    tolerance = truth["sigma_raw"] / (hi - lo)
    budget = PROTOCOL["budget"]
    if arm in ("random", "nsga2", "lhfid"):
        result = moea.run(arm, observe, budget, 20, moea.das_dennis(4, 3), seed=seed)
        policies = result["policies"]
    else:
        inner = "nsga2" if arm == "sa_nsga2" else "lhfid"
        result = run_sa(inner, observe, budget, seed, moea.das_dennis(4, 5),
                        tie_tolerance=tolerance if arm == "sa_lhfid_noise" else None)
        policies = result["policies"]
    import moocore

    true = scale(truth["model"].predict(policies))
    ref = np.ones(len(NAMES))
    inside = true[(true < ref).all(axis=1)]
    hv = float(moocore.hypervolume(inside, ref=ref)) if len(inside) else 0.0
    signatures = {nas_space.signature(p) for p in policies}
    return {"truth": truth_index, "seed": seed, "arm": arm, "true_hv": hv, "evaluated": len(policies),
            "unique": len(signatures), "inside_box": int(len(inside))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=PROTOCOL["seeds_per_truth"])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sa_simulation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_archive(ROOT / "artifacts/nas")
    state = build(rows)
    for t in state["truths"]:
        print(f"truth {t['family']}: noise SD raw {dict(zip(NAMES, np.round(t['sigma_raw'], 3)))}")
    state_path = args.output / "truths.pkl"
    state_path.write_bytes(pickle.dumps(state))
    arms = list(PROTOCOL["arms"])
    tasks = [(t, s, a) for t in range(len(state["truths"])) for s in range(args.seeds) for a in arms]
    print(f"{len(tasks)} runs on {args.workers} workers", flush=True)
    records = []
    with get_context("spawn").Pool(args.workers, initializer=load_state, initargs=(str(state_path),)) as pool:
        for i, record in enumerate(pool.imap_unordered(task, tasks), 1):
            records.append(record)
            if i % 20 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
    from scipy.stats import mannwhitneyu

    statistics_out = {}
    for t in range(len(state["truths"])):
        values = {a: [r["true_hv"] for r in records if r["truth"] == t and r["arm"] == a] for a in arms}
        raw, pairs = {}, {}
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                p = float(mannwhitneyu(values[a], values[b], alternative="two-sided").pvalue)
                raw[f"{a} vs {b}"] = p
                pairs[f"{a} vs {b}"] = {"median_difference": statistics.median(values[a]) - statistics.median(values[b]), "p": p}
        for pair, p in holm(raw).items():
            pairs[pair]["p_holm"] = p
        statistics_out[state["truths"][t]["family"]] = {
            "summary": {a: {"median": statistics.median(v), "mean": statistics.fmean(v), "sd": statistics.pstdev(v)} for a, v in values.items()},
            "pairs": pairs}
    out = {"protocol": PROTOCOL, "noise_sd_raw": {t["family"]: dict(zip(NAMES, t["sigma_raw"].tolist())) for t in state["truths"]},
           "statistics": statistics_out, "runs": records}
    (args.output / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    for family, block in statistics_out.items():
        print(f"\ntruth {family}: " + "  ".join(f"{a} {v['median']:.4f}" for a, v in block["summary"].items()))
        for pair, v in block["pairs"].items():
            if "sa_" in pair:
                print(f"   {pair:28s} dmed {v['median_difference']:+.4f}  pHolm {v['p_holm']:.2g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
