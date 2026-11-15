#!/usr/bin/env python3
"""POST HOC diagnostics of the C3 result (written after artifacts/analysis/c3.json was seen; not pre-registered).

1. Pre-registered C3 HV with latency rescaled to the calibration node (test job ran ~7% slower).
2. Calibration-to-test2 shift of CLIP relative to each split's reference (winner's curse, per arm).
3. Calibration 4-D HV of each arm's 5 locked finalists (does the ordering exist before any test data?).
4. Each arm's calibration front: full HV and greedy hypervolume-subset-selection (k=5) HV inside the box,
   i.e. what a box-aware lock rule would have kept.
5. Cross-fit of (4): choose the HSSP-5 set with CLIP and black counts from one random half of the calibration
   prompts, score its HV on the other half, both directions, 20 splits. Separates "a box-aware lock would have
   favoured SA-LHFiD" from selection on noise in a larger pool. Fidelity uses full-data values for every arm
   (v3 runs in artifacts/nas kept no per-image fidelity; fidelity split-half reliability is 0.90 vs 0.50 for CLIP).
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas import nas_space  # noqa: E402
from diffusionnas.nas_search import SEARCH_OBJECTIVES  # noqa: E402
from diffusionnas.pareto import nondominated  # noqa: E402
from diffusionnas.stats import per_prompt_scores  # noqa: E402

ARMS = ("sa_lhfid", "sa_nsga2", "random", "nsga2")
ROOTS = {"sa_lhfid": ["artifacts/nas_c2"], "sa_nsga2": ["artifacts/nas_c2"],
         "random": ["artifacts/nas", "artifacts/nas_c2"], "nsga2": ["artifacts/nas", "artifacts/nas_c2"]}
K = 5
SPLITS = 20


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def vector(o: dict) -> list[float]:
    return [-o["clip_score"], -o["clip_image_fidelity"], o["latency_p50_ms"], o["unet_weight_mb"]]


def hv(points: np.ndarray, ref: np.ndarray) -> float:
    import moocore
    inside = points[(points < ref).all(axis=1)] if len(points) else points
    return float(moocore.hypervolume(inside, ref=ref)) if len(inside) else 0.0


def calibration_pool(arm: str) -> tuple[dict, dict]:
    """Pool exactly as scripts/lock_nas_finalists.py; also keep each entry's run dirs and job references."""
    refs, pool = {}, {}
    for root in ROOTS[arm]:
        for seed_dir in sorted((ROOT / root).glob("seed*")):
            if not (seed_dir / arm / "search_results.json").exists():
                continue
            ref_run = next((seed_dir / "reference" / "runs").iterdir())
            summary = read(ref_run / "summary.json")
            refs[seed_dir] = (summary["latency_ms"]["p50"], summary["quality_metrics"]["clip_score"]["mean"],
                              summary["quality_metrics"]["black_images"], ref_run)
            for c in read(seed_dir / arm / "search_results.json")["candidates"]:
                entry = pool.setdefault(nas_space.signature(c["policy"]),
                                        {"objectives": dict(c["objectives"]), "rel": [], "runs": []})
                entry["rel"].append(c["objectives"]["latency_p50_ms"] / refs[seed_dir][0])
                entry["runs"].append(seed_dir / arm / "runs" / Path(c["run_dir"]).name)
    return pool, refs


def calibration_front(arm: str) -> tuple[list[dict], float]:
    """Same pooling, latency normalization, merge and eligibility as scripts/lock_nas_finalists.py."""
    pool, refs = calibration_pool(arm)
    latency = statistics.median(r[0] for r in refs.values())
    clip = statistics.median(r[1] for r in refs.values())
    black = max(r[2] for r in refs.values())
    for entry in pool.values():
        entry["objectives"]["latency_p50_ms"] = statistics.median(entry["rel"]) * latency
    eligible = [e for e in pool.values() if e["objectives"]["clip_score"] >= clip - 1.0
                and e["objectives"]["clip_image_fidelity"] >= 75.0 and e["objectives"]["black_images"] <= black + 2]
    return nondominated(eligible, SEARCH_OBJECTIVES), clip


def greedy_hssp(F: np.ndarray, ref: np.ndarray, k: int) -> list[int]:
    chosen: list[int] = []
    for _ in range(min(k, len(F))):
        chosen.append(max((i for i in range(len(F)) if i not in chosen), key=lambda i: hv(F[chosen + [i]], ref)))
    return chosen


def per_prompt(run: Path, field: str) -> dict[int, float]:
    return per_prompt_scores(run, field)


def cross_fit(arm: str) -> dict:
    """Select HSSP-5 on one prompt half, evaluate on the other (CLIP and black split; fidelity full-data)."""
    pool, refs = calibration_pool(arm)
    latency = statistics.median(r[0] for r in refs.values())
    entries = list(pool.values())
    prompts = sorted(per_prompt(next(iter(refs.values()))[3], "clip_score"))
    index = {p: i for i, p in enumerate(prompts)}

    def matrix(run_lists: list[list[Path]], field: str) -> np.ndarray:
        out = np.full((len(run_lists), len(prompts)), np.nan)
        for row, runs in enumerate(run_lists):
            values = [per_prompt(r, field) for r in runs]
            for p in prompts:
                out[row, index[p]] = statistics.fmean(v[p] for v in values)
        return out

    clip = matrix([e["runs"] for e in entries], "clip_score")
    black = matrix([e["runs"] for e in entries], "black_image") * 2  # per-prompt mean over 2 seeds -> count
    ref_clip = matrix([[r[3]] for r in refs.values()], "clip_score")
    ref_black = matrix([[r[3]] for r in refs.values()], "black_image") * 2
    fidelity = np.array([e["objectives"]["clip_image_fidelity"] for e in entries])
    lat = np.array([statistics.median(e["rel"]) * latency for e in entries])
    mb = np.array([e["objectives"]["unet_weight_mb"] for e in entries])
    rng = np.random.default_rng(0)
    results = []
    for _ in range(SPLITS):
        perm = rng.permutation(len(prompts))
        halves = (perm[: len(perm) // 2], perm[len(perm) // 2:])
        for sel, ev in (halves, halves[::-1]):
            def objectives(h):
                c = clip[:, h].mean(axis=1)
                rc = float(np.median(ref_clip[:, h].mean(axis=1)))
                ok = (c >= rc - 1.0) & (fidelity >= 75.0) & (black[:, h].sum(axis=1) <= ref_black[:, h].sum(axis=1).max() + 2)
                box = np.array([-(rc - 1.0), -75.0, 600.0, 1700.0])
                return np.column_stack([-c, -fidelity, lat, mb]), ok, box
            Fs, ok, box_s = objectives(sel)
            Fe, _, box_e = objectives(ev)
            candidates = np.where(ok & (Fs < box_s).all(axis=1))[0]
            front = [i for i in candidates if not any((Fs[j] <= Fs[i]).all() and (Fs[j] < Fs[i]).any() for j in candidates)]
            chosen = [front[i] for i in greedy_hssp(Fs[front], box_s, K)]
            results.append({"selection_hv": hv(Fs[chosen], box_s), "evaluation_hv": hv(Fe[chosen], box_e)})
    sel = np.array([r["selection_hv"] for r in results])
    ev = np.array([r["evaluation_hv"] for r in results])
    return {"selection_hv_mean": float(sel.mean()), "evaluation_hv_mean": float(ev.mean()),
            "evaluation_hv_sd": float(ev.std(ddof=1)), "retained_fraction": float(ev.mean() / sel.mean()),
            "evaluation_hv_by_fold": ev.tolist()}


def main() -> int:
    c3 = read(ROOT / "artifacts/analysis/c3.json")
    assignments = read(ROOT / "configs/generated/c3/assignments.json")["arms"]
    secondary = {label: v for arm in c3["secondary"].values() for label, v in arm.items()}
    locked = {arm: read(ROOT / f"configs/generated/c3/locked_{arm}.json") for arm in ARMS}
    test_ref = np.array([-c3["clip_floor"], -75.0, 600.0, 1700.0])
    test_ref_clip = c3["clip_floor"] + 1.0
    test_ref_latency = statistics.fmean(c3["reference_latency_ms"].values())
    cal_ref_latency = statistics.fmean(locked[a]["pooled_reference_latency_ms"] for a in ARMS)
    scale = cal_ref_latency / test_ref_latency

    label_of, n = {}, 0  # labels follow prepare_c3.py: first appearance of each signature in arm order
    for arm in ARMS:
        for f in locked[arm]["finalists"]:
            key = nas_space.signature(f["policy"])
            if key not in label_of:
                label_of[key] = f"c3_{n:02d}"
                n += 1

    out = {"note": "post hoc; not pre-registered", "latency_scale_to_calibration_node": scale, "arms": {}}
    for arm in ARMS:
        members = assignments[arm]
        test = np.array([[-secondary[m]["clip"], -secondary[m]["fidelity"], secondary[m]["latency_p50_ms"],
                          secondary[m]["unet_weight_mb"]] for m in members])
        test_scaled = test * np.array([1, 1, scale, 1])
        cal_ref_clip = locked[arm]["reference_clip"]
        cal_box = np.array([-(cal_ref_clip - 1.0), -75.0, 600.0, 1700.0])
        shifts = []
        for f in locked[arm]["finalists"]:
            label = label_of[nas_space.signature(f["policy"])]
            shifts.append((secondary[label]["clip"] - test_ref_clip) - (f["objectives"]["clip_score"] - cal_ref_clip))
        cal_locked = np.array([vector(f["objectives"]) for f in locked[arm]["finalists"]])
        front, _ = calibration_front(arm)
        assert len(front) == locked[arm]["front_size"], (arm, len(front), locked[arm]["front_size"])
        F = np.array([vector(p["objectives"]) for p in front])
        F = F[(F < cal_box).all(axis=1)]
        chosen = greedy_hssp(F, cal_box, K)
        out["arms"][arm] = {
            "test2_hv_preregistered": c3["hypervolume"][arm],
            "test2_hv_latency_rescaled": hv(test_scaled, test_ref),
            "test2_in_box": int((test < test_ref).all(axis=1).sum()),
            "clip_shift_test_minus_calibration_relative": {"mean": statistics.fmean(shifts), "values": shifts},
            "calibration_hv_locked5": hv(cal_locked, cal_box),
            "calibration_in_box_locked5": int((cal_locked < cal_box).all(axis=1).sum()),
            "calibration_front_in_box": len(F),
            "calibration_hv_full_front": hv(F, cal_box),
            "calibration_hv_greedy_hssp5": hv(F[chosen], cal_box),
            "cross_fit_greedy_hssp5": cross_fit(arm),
        }
    folds = {arm: np.array(out["arms"][arm]["cross_fit_greedy_hssp5"]["evaluation_hv_by_fold"]) for arm in ARMS}
    out["cross_fit_sa_lhfid_wins_fraction_of_folds"] = {
        other: float((folds["sa_lhfid"] > folds[other]).mean()) for other in ARMS if other != "sa_lhfid"}
    path = ROOT / "artifacts/analysis/c3_diagnostics.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"latency scale {scale:.4f}")
    for arm, d in out["arms"].items():
        print(f"{arm:9s} test2 HV {d['test2_hv_preregistered']:.3g} (rescaled {d['test2_hv_latency_rescaled']:.3g}, in box {d['test2_in_box']}/5) | "
              f"CLIP shift {d['clip_shift_test_minus_calibration_relative']['mean']:+.2f} | cal HV locked5 {d['calibration_hv_locked5']:.3g} "
              f"({d['calibration_in_box_locked5']}/5 in box), greedy-HSSP5 {d['calibration_hv_greedy_hssp5']:.3g}, full front "
              f"{d['calibration_hv_full_front']:.3g} ({d['calibration_front_in_box']} pts)")
        x = d["cross_fit_greedy_hssp5"]
        print(f"{'':9s} cross-fit HSSP5: selection-half HV {x['selection_hv_mean']:.3g} -> other-half {x['evaluation_hv_mean']:.3g} "
              f"(sd {x['evaluation_hv_sd']:.2g}, retained {x['retained_fraction']:.0%})")
    print("SA-LHFiD wins on fraction of cross-fit folds:", out["cross_fit_sa_lhfid_wins_fraction_of_folds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
