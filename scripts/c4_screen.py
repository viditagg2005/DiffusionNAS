#!/usr/bin/env python3
"""C4 stage A (screen 16 per arm) and naive control (5 per arm) from existing calibration archives; writes the
stage B re-measurement run list on calibration2 (configs/nas/c4_protocol.json). Run once; outputs are locked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diffusionnas import final_selection as fs  # noqa: E402
from diffusionnas.nas_space import signature  # noqa: E402
from diffusionnas.schema import validate_config  # noqa: E402
from diffusionnas.search_space import policy_to_config  # noqa: E402
from prepare_experiment import load_prompts  # noqa: E402

ARMS = {"sa_lhfid": ["artifacts/nas_c2"], "sa_nsga2": ["artifacts/nas_c2"],
        "random": ["artifacts/nas", "artifacts/nas_c2"], "nsga2": ["artifacts/nas", "artifacts/nas_c2"]}
N_SCREEN, K = 16, 5
IMAGE_SEED = 707
OUT = ROOT / "configs" / "generated" / "c4"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def pooled_archive(roots: list[Path], arm: str) -> dict:
    """Pool one arm exactly as scripts/lock_nas_finalists.py; return eligible entries and pooled references."""
    refs, pool, gpus = [], {}, set()
    for root in roots:
        for seed_dir in sorted(root.glob("seed*")):
            if not (seed_dir / arm / "search_results.json").exists():
                continue
            ref_run = next(d for d in sorted((seed_dir / "reference" / "runs").iterdir())
                           if read(d / "manifest.json").get("status") == "succeeded")
            summary = read(ref_run / "summary.json")
            ref = {"latency": summary["latency_ms"]["p50"], "clip": summary["quality_metrics"]["clip_score"]["mean"],
                   "black": summary["quality_metrics"]["black_images"]}
            refs.append(ref)
            gpus.add(read(ref_run / "manifest.json")["provenance"]["gpu"]["name"])
            for c in read(seed_dir / arm / "search_results.json")["candidates"]:
                entry = pool.setdefault(signature(c["policy"]), {"policy": c["policy"], "objectives": dict(c["objectives"]),
                                                                 "rel": [], "sources": []})
                entry["rel"].append(c["objectives"]["latency_p50_ms"] / ref["latency"])
                entry["sources"].append(f"{seed_dir.parent.name}/{seed_dir.name}/{arm}/{c['candidate_id']}")
    if len(gpus) != 1:
        raise SystemExit(f"{arm}: searches ran on {sorted(gpus)}")
    latency = statistics.median(r["latency"] for r in refs)
    clip = statistics.median(r["clip"] for r in refs)
    black = statistics.median(r["black"] for r in refs)
    black_max = max(r["black"] for r in refs)
    for e in pool.values():
        e["objectives"]["latency_p50_ms"] = statistics.median(e["rel"]) * latency
    eligible = [e for e in pool.values() if e["objectives"]["clip_score"] >= clip - 1.0
                and e["objectives"]["clip_image_fidelity"] >= 75.0 and e["objectives"]["black_images"] <= black_max + 2]
    front = [eligible[i] for i in fs.nondominated_indices([fs.vector(e["objectives"]) for e in eligible])]
    return {"front": front, "reference_latency": latency, "reference_clip": clip, "reference_black_median": black,
            "pool": len(pool), "eligible": len(eligible)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=ROOT, help="root containing artifacts/nas and artifacts/nas_c2")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; C4 screening is locked once")
    args.out.mkdir(parents=True)
    (args.out / "remeasure").mkdir()

    screen = {}
    for arm, roots in ARMS.items():
        archive = pooled_archive([args.artifacts / r for r in roots], arm)
        lock = read(ROOT / "configs" / "generated" / "c3" / f"locked_{arm}.json")
        assert len(archive["front"]) == lock["front_size"], (arm, len(archive["front"]), lock["front_size"])
        ref = fs.box(archive["reference_clip"], archive["reference_latency"])
        points = np.array([fs.vector(e["objectives"]) for e in archive["front"]])
        order = fs.greedy_hssp(points, ref, N_SCREEN)
        screened = [archive["front"][i] for i in order]
        screen[arm] = {
            "box": ref.tolist(), "reference_latency_ms": archive["reference_latency"],
            "reference_clip": archive["reference_clip"], "reference_black_median": archive["reference_black_median"],
            "pool_unique": archive["pool"], "eligible": archive["eligible"], "front": len(archive["front"]),
            "front_in_box": int(fs.in_box(points, ref).sum()),
            "calibration_hv_naive5": fs.hypervolume(points[order[:K]], ref),
            "screened": [{"rank": r, "signature": signature(e["policy"]), "policy": e["policy"],
                          "calibration_objectives": e["objectives"], "sources": e["sources"]} for r, e in enumerate(screened)],
            "naive": [signature(e["policy"]) for e in screened[:K]],
        }

    prompts = load_prompts(ROOT / "data" / "prompts" / "calibration2.json")
    runtime = {"device": "cuda", "precision": "float16", "warmup_runs": 10, "repetitions": 1, "count_flops": False}
    labels: dict[str, str] = {}
    configs: dict[str, dict] = {}
    for arm in ARMS:
        for item in screen[arm]["screened"]:
            key = item["signature"]
            if key not in labels:
                labels[key] = f"r{len(labels):02d}"
                config = policy_to_config(item["policy"], prompts, [IMAGE_SEED], dict(runtime), "artifacts/c4/remeasure/runs")
                if config.get("architecture") == {"removed_units": [], "weight_precision": "fp16"}:
                    config.pop("architecture")
                config.update({"label": labels[key], "role": "c4_remeasure"})
                validate_config(config)
                configs[labels[key]] = config
            item["label"] = labels[key]
    reference = read(ROOT / "configs" / "baseline_sd15.json")
    reference.update({"prompts": prompts, "seeds": [IMAGE_SEED], "output_root": "artifacts/c4/remeasure/runs"})
    reference["runtime"].update(runtime)
    order = list(configs)
    random.Random(20261117).shuffle(order)
    middle = len(order) // 2
    sequence = ["ref_a"] + order[:middle] + ["ref_b"] + order[middle:] + ["ref_c"]
    for index, label in enumerate(sequence):
        config = dict(reference, label=label, role="reference") if label.startswith("ref_") else configs[label]
        validate_config(config)
        (args.out / "remeasure" / f"{index:02d}_{label}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (args.out / "screen.json").write_text(json.dumps({"arms": screen, "remeasure_order": sequence}, indent=2) + "\n")
    shutil.copy(ROOT / "configs" / "nas" / "c4_protocol.json", args.out / "protocol_at_lock.json")
    print(f"{len(configs)} unique screened configurations + 3 references -> {args.out / 'remeasure'}")
    for arm, s in screen.items():
        print(f"  {arm:9s} pool {s['pool_unique']}, eligible {s['eligible']}, front {s['front']} ({s['front_in_box']} in box); "
              f"screened {[i['label'] for i in s['screened']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
