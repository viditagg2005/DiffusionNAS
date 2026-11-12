#!/usr/bin/env python3
"""Final validation tables and figures from artifacts/validation_v3 (one allocation, untouched split).

Per run: latency (p50, speedup vs the median of the three in-job references),
GFLOPs, peak VRAM, U-Net parameters and bytes, CLIP alignment and CLIP fidelity
with paired prompt-bootstrap 95% CIs against reference ``ref_a``, black images,
KID/FID relative to the reference and vs real COCO. Also: reference drift,
finalist-vs-twin deltas (what the architecture genes add), and the 6-objective
validation front. Writes artifacts/analysis/validation_v3.json, docs/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas.pareto import nondominated  # noqa: E402
from diffusionnas.stats import paired_bootstrap, per_prompt_scores  # noqa: E402

RUNS = ROOT / "artifacts" / "validation_v3" / "runs"
FIG = ROOT / "docs" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FRONT_OBJECTIVES = (("clip_score", "max"), ("clip_image_fidelity", "max"), ("latency_p50_ms", "min"),
                    ("gflops_per_image", "min"), ("peak_allocated_mb", "min"), ("unet_weight_mb", "min"))


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    runs = {}
    gpus = set()
    for manifest in sorted(RUNS.glob("*/manifest.json")):
        if read(manifest).get("status") != "succeeded":
            continue
        config = read(manifest.parent / "config.json")
        runs[config["label"]] = (manifest.parent, config)
        gpus.add(read(manifest)["provenance"]["gpu"]["name"])
    if len(gpus) != 1:
        raise SystemExit(f"validation spans GPU models {gpus}")
    refs = [runs[k][0] for k in ("ref_a", "ref_b", "ref_c") if k in runs]
    ref_latency = statistics.median(read(r / "summary.json")["latency_ms"]["p50"] for r in refs)
    ref_clip = per_prompt_scores(refs[0])
    ref_fid = per_prompt_scores(refs[0], "clip_image_fidelity")

    table = {}
    for label, (run, config) in runs.items():
        s = read(run / "summary.json")
        q = s["quality_metrics"]
        arch = s["architecture"]
        row = {
            "label": label, "role": config.get("role"), "model": config["model"]["id"], "method": config["method"],
            "scheduler": config["inference"]["scheduler"], "steps": config["inference"]["num_inference_steps"],
            "guidance": config["inference"]["guidance_scale"],
            "removed_units": (config.get("architecture") or {}).get("removed_units", []),
            "weight_precision": arch["weight_precision"],
            "latency_p50_ms": s["latency_ms"]["p50"], "latency_p90_ms": s["latency_ms"]["p90"],
            "speedup": ref_latency / s["latency_ms"]["p50"],
            "gflops_per_image": s.get("gflops_per_image"), "peak_allocated_mb": s["peak_allocated_mb"],
            "unet_parameters_m": arch["unet_parameters"] / 1e6, "unet_weight_mb": arch["unet_weight_bytes"] / 2**20,
            "clip_score": q["clip_score"]["mean"], "clip_image_fidelity": q["clip_image_fidelity"]["mean"],
            "black_images": q["black_images"],
            "kid_relative_x1000": q.get("relative_distribution", {}).get("kid", {}).get("mean_x1000"),
            "fid_relative": q.get("relative_distribution", {}).get("fid"),
            "kid_real_x1000": q.get("real_distribution", {}).get("kid", {}).get("mean_x1000"),
            "fid_real": q.get("real_distribution", {}).get("fid"),
        }
        if run != refs[0]:
            row["clip_vs_reference"] = paired_bootstrap(per_prompt_scores(run), ref_clip)
            row["fidelity_vs_reference"] = paired_bootstrap(per_prompt_scores(run, "clip_image_fidelity"), ref_fid)
        table[label] = row

    twins = {}
    for label, row in table.items():
        if row["role"] != "finalist":
            continue
        for suffix in ("_twin_fp16", "_twin_int8"):
            twin = table.get(label + suffix)
            if twin is None:
                continue
            twins[label + suffix] = {
                "latency_change_pct": 100 * (row["latency_p50_ms"] / twin["latency_p50_ms"] - 1),
                "weight_mb_change": row["unet_weight_mb"] - twin["unet_weight_mb"],
                "peak_mb_change": row["peak_allocated_mb"] - twin["peak_allocated_mb"],
                "gflops_change_pct": 100 * (row["gflops_per_image"] / twin["gflops_per_image"] - 1),
                "clip_vs_twin": paired_bootstrap(per_prompt_scores(runs[label][0]), per_prompt_scores(runs[label + suffix][0])),
                "fidelity_vs_twin": paired_bootstrap(per_prompt_scores(runs[label][0], "clip_image_fidelity"),
                                                     per_prompt_scores(runs[label + suffix][0], "clip_image_fidelity")),
            }
    sd15 = [r for r in table.values() if r["model"].startswith("stable-diffusion") and r["label"] not in ("ref_b", "ref_c")]
    front = nondominated([{"label": r["label"], "objectives": r} for r in sd15], FRONT_OBJECTIVES)
    result = {
        "gpu": gpus.pop(),
        "reference_latency_ms": {k: read(runs[k][0] / "summary.json")["latency_ms"]["p50"] for k in ("ref_a", "ref_b", "ref_c") if k in runs},
        "reference_latency_median_ms": ref_latency,
        "runs": table,
        "finalist_vs_twin": twins,
        "validation_front_6obj": sorted(p["label"] for p in front),
    }
    out = ROOT / "artifacts" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_v3.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"GPU {result['gpu']}; reference p50 {result['reference_latency_ms']}")
    print(f"{'label':22s} {'ms':>6s} {'x':>5s} {'MB':>6s} {'peak':>6s} {'CLIP':>6s} {'dCLIP [CI]':>22s} {'fid':>5s} {'blk':>3s} {'rKID':>6s} {'KIDreal':>7s}")
    for row in sorted(table.values(), key=lambda r: r["latency_p50_ms"]):
        ci = row.get("clip_vs_reference")
        dclip = f"{ci['mean_difference']:+.2f} [{ci['ci95_low']:+.2f},{ci['ci95_high']:+.2f}]" if ci else "ref"
        fmt = lambda v, f: format(v, f) if v is not None else "-"
        print(f"{row['label']:22s} {row['latency_p50_ms']:6.0f} {row['speedup']:5.2f} {row['unet_weight_mb']:6.0f} "
              f"{row['peak_allocated_mb']:6.0f} {row['clip_score']:6.2f} {dclip:>22s} {row['clip_image_fidelity']:5.1f} "
              f"{row['black_images']:3d} {fmt(row['kid_relative_x1000'], '6.2f'):>6s} {fmt(row['kid_real_x1000'], '7.2f'):>7s}")
    print("front:", result["validation_front_6obj"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = (("NAS finalists", lambda r: r["role"] == "finalist", ORANGE),
              ("Training-free baselines and twins", lambda r: r["role"] in ("baseline", "twin", "reference"), BLUE),
              ("BK-SDM (distilled, external)", lambda r: r["role"] == "external", AQUA))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, (x, xlabel) in zip(axes, (("latency_p50_ms", "p50 latency, ms (one allocation)"), ("unet_weight_mb", "U-Net weights, MB"))):
        for name, member, color in groups:
            rows = [r for r in table.values() if member(r)]
            ax.scatter([r[x] for r in rows], [r["clip_score"] for r in rows], s=40 if color == ORANGE else 26, color=color,
                       edgecolors="white", linewidths=1.2, label=name, zorder=3 if color == ORANGE else 2)
        ax.axhline(table["ref_a"]["clip_score"], color=MUTED, linewidth=1, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    axes[0].set_ylabel("CLIP prompt alignment (validation)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "validation_front.png", dpi=200)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
