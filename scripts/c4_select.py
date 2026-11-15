#!/usr/bin/env python3
"""C4 stage C: choose 5 per arm from the calibration2 re-measurements (greedy HSSP), then write the test3
run list with every NFS and naive set (configs/nas/c4_protocol.json). Run once, after stage B is scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diffusionnas import final_selection as fs  # noqa: E402
from diffusionnas.schema import validate_config  # noqa: E402
from diffusionnas.search_space import policy_to_config  # noqa: E402
from prepare_experiment import load_prompts  # noqa: E402

K = 5
IMAGE_SEED = 909


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def succeeded_runs(runs_root: Path) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for manifest in sorted(runs_root.glob("*/manifest.json")):
        if read(manifest).get("status") == "succeeded":
            label = read(manifest.parent / "config.json")["label"]
            if label in runs:
                raise SystemExit(f"two succeeded runs for {label}")
            runs[label] = manifest.parent
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c4", type=Path, default=ROOT / "configs" / "generated" / "c4")
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts" / "c4" / "remeasure" / "runs")
    args = parser.parse_args()
    test_dir = args.c4 / "test"
    if test_dir.exists():
        raise SystemExit(f"{test_dir} exists; C4 selection is locked once")
    screen = read(args.c4 / "screen.json")
    runs = succeeded_runs(args.runs)
    needed = sorted({i["label"] for s in screen["arms"].values() for i in s["screened"]} | {"ref_a", "ref_b", "ref_c"})
    missing = [label for label in needed if label not in runs]
    if missing:
        raise SystemExit(f"missing re-measurement runs: {missing}")
    gpus = {read(runs[label] / "manifest.json")["provenance"]["gpu"]["name"] for label in needed}
    if len(gpus) != 1:
        raise SystemExit(f"re-measurement ran on {sorted(gpus)}")

    summary = {label: read(runs[label] / "summary.json") for label in needed}
    ref_latency = statistics.median(summary[r]["latency_ms"]["p50"] for r in ("ref_a", "ref_b", "ref_c"))
    ref_clip = summary["ref_a"]["quality_metrics"]["clip_score"]["mean"]
    ref_black2 = summary["ref_a"]["quality_metrics"]["black_images"]
    ref = fs.box(ref_clip, ref_latency)

    def fresh(label: str) -> dict:
        q = summary[label]["quality_metrics"]
        return {"clip_score": q["clip_score"]["mean"], "clip_image_fidelity": q["clip_image_fidelity"]["mean"],
                "latency_p50_ms": summary[label]["latency_ms"]["p50"],
                "unet_weight_mb": summary[label]["architecture"]["unet_weight_bytes"] / 2**20,
                "black_images": q["black_images"]}

    selection = {"box": ref.tolist(), "reference_latency_ms": ref_latency, "reference_clip": ref_clip,
                 "reference_black": ref_black2, "gpu": gpus.pop(), "arms": {}}
    for arm, s in screen["arms"].items():
        rows = []
        for item in s["screened"]:
            o = fresh(item["label"])
            black_total = item["calibration_objectives"]["black_images"] + o["black_images"]
            eligible = black_total <= s["reference_black_median"] + ref_black2 + 2
            rows.append({"label": item["label"], "signature": item["signature"], "fresh": o, "black_total": black_total,
                         "eligible": bool(eligible)})
        pool = [r for r in rows if r["eligible"]]
        points = np.array([fs.vector(r["fresh"]) for r in pool]).reshape(-1, 4)
        chosen = [pool[i] for i in fs.greedy_hssp(points, ref, K)]
        selection["arms"][arm] = {"rows": rows, "nfs": [r["signature"] for r in chosen],
                                  "nfs_labels": [r["label"] for r in chosen],
                                  "naive": s["naive"], "naive_labels": [i["label"] for i in s["screened"][:K]]}

    # test3 run list: every NFS and naive set, shared signatures once
    policies = {i["signature"]: i["policy"] for s in screen["arms"].values() for i in s["screened"]}
    prompts = load_prompts(ROOT / "data" / "prompts" / "test3.json")
    runtime = {"device": "cuda", "precision": "float16", "warmup_runs": 10, "repetitions": 1, "count_flops": True}
    labels: dict[str, str] = {}
    assignments: dict[str, dict[str, list[str]]] = {"nfs": {}, "naive": {}}
    for arm, sel in selection["arms"].items():
        for kind in ("nfs", "naive"):
            for key in sel[kind]:
                labels.setdefault(key, f"t{len(labels):02d}")
            assignments[kind][arm] = [labels[key] for key in sel[kind]]
    test_dir.mkdir(parents=True)
    configs = {}
    for key, label in labels.items():
        config = policy_to_config(policies[key], prompts, [IMAGE_SEED], dict(runtime), "artifacts/c4/test/runs")
        if config.get("architecture") == {"removed_units": [], "weight_precision": "fp16"}:
            config.pop("architecture")
        config.update({"label": label, "role": "c4_test"})
        validate_config(config)
        configs[label] = config
    reference = read(ROOT / "configs" / "baseline_sd15.json")
    reference.update({"prompts": prompts, "seeds": [IMAGE_SEED], "output_root": "artifacts/c4/test/runs"})
    reference["runtime"].update(runtime)
    order = list(configs)
    random.Random(20261118).shuffle(order)
    middle = len(order) // 2
    sequence = ["ref_a"] + order[:middle] + ["ref_b"] + order[middle:] + ["ref_c"]
    for index, label in enumerate(sequence):
        config = dict(reference, label=label, role="reference") if label.startswith("ref_") else configs[label]
        validate_config(config)
        (test_dir / f"{index:02d}_{label}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (args.c4 / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (args.c4 / "assignments.json").write_text(json.dumps({**assignments, "order": sequence,
                                                          "signatures": {v: k for k, v in labels.items()}}, indent=2) + "\n")
    print(f"{len(configs)} unique test3 configurations + 3 references -> {test_dir}")
    for arm in selection["arms"]:
        print(f"  {arm:9s} nfs {assignments['nfs'][arm]}  naive {assignments['naive'][arm]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
